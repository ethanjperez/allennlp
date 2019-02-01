import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FiLM(torch.nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(1)
        betas = betas.unsqueeze(1)
        return (gammas * x) + betas


@Model.register("bert-qa")
class BertQA(Model):
    """
    This class implements BERT for QA

    The basic layout is pretty simple: encode with BERT, apply a 1D convolution, and then
    do a softmax over span start and span end.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 span_end_encoder: Seq2SeqEncoder = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False) -> None:
        super(BertQA, self).__init__(vocab, regularizer)

        self.judge = judge
        self.is_judge = self.judge is None
        self.reward_method = None if self.is_judge else reward_method
        self.update_judge = update_judge and (self.judge is not None)
        self._detach_value_head = detach_value_head
        self._text_field_embedder = text_field_embedder
        self.answer_type = 'span' if (span_end_encoder is not None) else 'mc'

        span_start_input_dim = text_field_embedder.get_output_dim()
        if not self.is_judge:
            self._value_head = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))  # NB: Can make MLP
            self._turn_film_gen = torch.nn.Linear(1, 2 * span_start_input_dim)
            self._film = FiLM()
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        if self.answer_type == 'span':
            self._span_end_encoder = span_end_encoder  # NOTE: Use low capacity
            span_end_input_dim = text_field_embedder.get_output_dim() + span_end_encoder.get_output_dim()
            self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

            check_dimensions_match(span_end_encoder.get_input_dim(), 3 * text_field_embedder.get_output_dim(),
                                   "span end encoder input dim", "3 * modeling dim")

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],  # Dummy input for BERT
                passage: Dict[str, torch.LongTensor],  # Contains Q
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                store_metrics: bool = True,
                valid_output_mask: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.
        store_metrics : bool
            If true, stores metrics (if applicable) within model metric tracker.
            If false, returns resulting metrics immediately, without updating the model metric tracker.
        valid_output_mask: ``torch.LongTensor``, optional
            The locations for a valid answer. Used to limit the model's output space.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        sep_token = metadata[0]['[SEP]'] if '[SEP]' in metadata[0] else self.vocab._token_to_index['bert']['[SEP]']
        sep_token_mask = (passage['tokens'] == sep_token).long()
        token_type_ids = (sep_token_mask.cumsum(-1) - sep_token_mask).clamp(max=1)
        if not self.is_judge:
            assert(metadata is not None and 'a_turn' in metadata[0])
            a_turn = torch.tensor([sample_metadata['a_turn'] for sample_metadata in metadata]).to(passage['tokens']).unsqueeze(1)
            # TODO: Use boolean variable passed in to determine if A/B should use Frozen Judge BERT or their own updating BERT
            if self._text_field_embedder._token_embedders['tokens'].requires_grad:
                token_type_ids[:, 0] = a_turn.squeeze(1)
            a_turn = a_turn.float()
        # Shape: (batch_size, passage_length, modeling_dim)
        modeled_passage = self._text_field_embedder(passage)
        batch_size, passage_length, modeling_dim = modeled_passage.size()
        passage_mask = util.get_text_field_mask(passage).float()
        if valid_output_mask is None:  # NB: Make this use question make too for normal Judge training
            valid_output_mask = passage_mask

        # Debate: Post-BERT agent-based conditioning
        if not self.is_judge:
            turn_film_params = self._turn_film_gen(a_turn)
            turn_gammas, turn_betas = torch.split(turn_film_params, modeling_dim, dim=-1)
            # NB: Check you need to apply passage_mask here
            modeled_passage = self._film(modeled_passage, 1. + turn_gammas, turn_betas) * passage_mask.unsqueeze(-1)

        span_start_input = self._dropout(modeled_passage)
        if not self.is_judge:
            value_head_input = modeled_passage.detach() if self._detach_value_head else modeled_passage
            # Shape: (batch_size)
            value = (self._value_head(value_head_input).squeeze(-1) * passage_mask).mean(1)
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, valid_output_mask)

        if self.answer_type == 'mc':
            span_end_logits = span_start_logits
            span_end_probs = span_start_probs
        elif self.answer_type == 'span':
            # Shape: (batch_size, modeling_dim)
            span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
            # Shape: (batch_size, passage_length, modeling_dim)
            tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
                                                                                       passage_length,
                                                                                       modeling_dim)

            # Shape: (batch_size, passage_length, modeling_dim * 3)
            span_end_representation = torch.cat([modeled_passage,
                                                 tiled_start_representation,
                                                 modeled_passage * tiled_start_representation],
                                                dim=-1)
            # Shape: (batch_size, passage_length, encoding_dim)
            encoded_span_end = self._dropout(self._span_end_encoder(span_end_representation,
                                                                    passage_mask))
            # Shape: (batch_size, passage_length, encoding_dim + span_end_encoding_dim)
            span_end_input = self._dropout(torch.cat([modeled_passage, encoded_span_end], dim=-1))
            span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
            span_end_probs = util.masked_softmax(span_end_logits, valid_output_mask)
        span_start_logits = util.replace_masked_values(span_start_logits, valid_output_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, valid_output_mask, -1e7)
        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
                "span_start_logits": span_start_logits,
                "span_start_probs": span_start_probs,
                "span_end_logits": span_end_logits,
                "span_end_probs": span_end_probs,
                "best_span": best_span,
                "value": value if not self.is_judge else None,
                }

        # Compute the loss for training.
        if span_start is not None:
            if self.answer_type == 'span':
                span_start[span_start >= valid_output_mask.size(1)] = -100  # Don't add to loss if span not in input. NB: Hacky: Will alter effective batch size
            loss = nll_loss(util.masked_log_softmax(span_start_logits, valid_output_mask), span_start.squeeze(-1))
            if store_metrics:
                self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
            if self.answer_type == 'span':
                span_end[span_end >= valid_output_mask.size(1)] = -100  # Don't add to loss if span not in input. NB: Hacky: Will alter effective batch size
                loss += nll_loss(util.masked_log_softmax(span_end_logits, valid_output_mask), span_end.squeeze(-1))
            if store_metrics:
                self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
                self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        tmp_squad_metrics = None
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                if (len(metadata[i]['passage_tokens']) != len(offsets)):
                    import ipdb; ipdb.set_trace()
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    if store_metrics:
                        self._squad_metrics(best_span_string, answer_texts)
                    else:
                        if tmp_squad_metrics is None:
                            tmp_squad_metrics = SquadEmAndF1()
                        tmp_squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
        if tmp_squad_metrics is not None:
            return output_dict, tmp_squad_metrics
        return output_dict

    def get_metrics(self, reset: bool = False, per_sample: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset, per_sample)
        return {
                'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'em': exact_match,
                'f1': f1_score,
                }

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

        span_start_logits = span_start_logits.detach().cpu().numpy()
        span_end_logits = span_end_logits.detach().cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span
