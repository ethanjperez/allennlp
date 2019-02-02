import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss, relu, softmax, log_softmax

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention import BilinearMatrixAttention
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


@Model.register("bert-mc")
class BertMC(Model):
    """
    This class implements BERT for Multiple-choice QA

    The basic layout is:
    1) Encode P, Q, A with BERT
    2) Use bilinear attentions (PxQ and PxA_i) to get P, Q, A summaries
    3) Additional, global non-linear operations on BERT and summary P, Q, A features
    4) Softmax over the predicted logit for each A_i

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
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False) -> None:
        super(BertMC, self).__init__(vocab, regularizer)

        self.judge = judge
        self.is_judge = self.judge is None
        self.reward_method = None if self.is_judge else reward_method
        self.update_judge = update_judge and (self.judge is not None)
        self._detach_value_head = detach_value_head
        self._text_field_embedder = text_field_embedder
        self.answer_type = 'mc'

        hidden_dim = text_field_embedder.get_output_dim()
        self._passage_question_attention = BilinearMatrixAttention(hidden_dim, hidden_dim, use_input_biases=True)
        self._passage_option_attention = BilinearMatrixAttention(hidden_dim, hidden_dim, use_input_biases=True)

        self._final_passage_question_encoder = TimeDistributed(torch.nn.Linear(2 * hidden_dim, hidden_dim))
        self._final_question_encoder = TimeDistributed(torch.nn.Linear(2 * hidden_dim, hidden_dim))
        self._final_passage_option_encoder = TimeDistributed(torch.nn.Linear(2 * hidden_dim, hidden_dim))
        self._final_option_encoder = TimeDistributed(torch.nn.Linear(2 * hidden_dim, hidden_dim))

        if not self.is_judge:
            self._value_head = TimeDistributed(torch.nn.Linear(hidden_dim, 1))  # NB: Can make MLP
            self._turn_film_gen = torch.nn.Linear(1, 2 * hidden_dim)
            self._film = FiLM()

        self._span_start_accuracy = CategoricalAccuracy()
        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                answer_index: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                store_metrics: bool = True) -> Dict[str, torch.Tensor]:
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
        answer_index : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            index of option that is the true answer.  If this is given, we will compute a loss
            that gets included in the output dictionary.
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
        # Get masks
        passage_mask = util.get_text_field_mask(passage).float()
        question_mask = util.get_text_field_mask(question).float()
        options_mask = util.get_text_field_mask(options).float()

        token_type_ids = torch.zeros_like(passage['tokens'], dtype=torch.long)
        if not self.is_judge:
            assert(metadata is not None and 'a_turn' in metadata[0])
            a_turn = torch.tensor([sample_metadata['a_turn'] for sample_metadata in metadata]).to(passage['tokens']).unsqueeze(1)
            # TODO: Use boolean variable passed in to determine if A/B should use Frozen Judge BERT or their own updating BERT
            if self._text_field_embedder._token_embedders['tokens'].requires_grad:
                token_type_ids[:, 0] = a_turn.squeeze(1)
            a_turn = a_turn.float()
        # Shape: (batch_size, passage_length, hidden_dim)  # TODO: Remove use of offsets from passage/question/options to get full BERT output!
        hidden_passage = self._text_field_embedder(passage) * passage_mask.unsqueeze(-1)
        hidden_question = self._text_field_embedder(question) * question_mask.unsqueeze(-1)
        hidden_options = self._text_field_embedder(options) * options_mask.unsqueeze(-1)

        # Get dimensions
        batch_size, passage_length, hidden_dim = hidden_passage.size()
        _, question_length, _ = hidden_question.size()
        _, num_options, options_length, _ = hidden_options.size()

        # Debate: Post-BERT agent-based conditioning
        if not self.is_judge:  # TODO: Check padding/masking is done correctly here!
            turn_film_params = self._turn_film_gen(a_turn)
            turn_gammas, turn_betas = torch.split(turn_film_params, hidden_dim, dim=-1)
            # NB: Check you need to apply passage_mask here
            hidden_passage = self._film(hidden_passage, 1. + turn_gammas, turn_betas) * passage_mask.unsqueeze(-1)

        ### Cross attentions
        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._passage_question_attention(hidden_passage, hidden_question)
        passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)  # NB: Need to incorporate passage_mask here? (and in other equivalent places?)
        # Shape: (batch_size, passage_length, hidden_dim)
        passage_question_vectors = util.weighted_sum(hidden_question, passage_question_attention) * passage_mask.unsqueeze(-1)  # M^p'
        summary_passage_question = relu(self._final_passage_question_encoder(torch.cat([passage_question_vectors - hidden_passage, passage_question_vectors * hidden_passage], dim=2)))  # S^q
        pooled_summary_passage_question = util.replace_masked_values(summary_passage_question, passage_mask.unsqueeze(-1), -1e7).max(dim=1)[0]
        # Shape: (batch_size, question_length, hidden_dim)
        question_passage_attention = util.masked_softmax(passage_question_similarity.transpose(1, 2), passage_mask)
        question_passage_vectors = util.weighted_sum(hidden_passage, question_passage_attention) * question_mask.unsqueeze(-1)  # M^q
        summary_question = relu(self._final_question_encoder(torch.cat([question_passage_vectors - hidden_question, question_passage_vectors * hidden_question], dim=2)))  # S^p'
        # Shape: (batch_size, hidden_dim)
        pooled_summary_question = util.replace_masked_values(summary_question, question_mask.unsqueeze(-1), -1e7).max(dim=1)[0]
        # Shape: (batch_size, 2 * hidden_dim)
        final_passage_question = torch.cat([pooled_summary_question, pooled_summary_passage_question], dim=1)

        option_logits = []
        for i in range(num_options):
            hidden_option = hidden_options[:, i]
            option_mask = options_mask[:, i]
            # Shape: (batch_size, passage_length, options_length)
            passage_option_similarity = self._passage_option_attention(hidden_passage, hidden_option)
            passage_option_attention = util.masked_softmax(passage_option_similarity, option_mask)
            # Shape: (batch_size, passage_length, hidden_dim)
            passage_option_vectors = util.weighted_sum(hidden_option, passage_option_attention) * passage_mask.unsqueeze(-1)
            summary_passage_option = relu(self._final_passage_option_encoder(torch.cat([passage_option_vectors - hidden_passage, passage_option_vectors * hidden_passage], dim=2)))
            # Shape: (batch_size, hidden_dim)
            pooled_summary_passage_option = util.replace_masked_values(summary_passage_option, passage_mask.unsqueeze(-1), -1e7).max(dim=1)[0]
            # Shape: (batch_size, options_length, hidden_dim)
            option_passage_attention = util.masked_softmax(passage_option_similarity.transpose(1, 2), passage_mask)
            option_passage_vectors = util.weighted_sum(hidden_passage, option_passage_attention) * option_mask.unsqueeze(-1)  # M^a?
            summary_option = relu(self._final_option_encoder(torch.cat([option_passage_vectors - hidden_option, option_passage_vectors * hidden_option], dim=2)))  # S^p?
            # Shape: (batch_size, hidden_dim)
            pooled_summary_option = util.replace_masked_values(summary_option, option_mask.unsqueeze(-1), -1e7).max(dim=1)[0]
            # Shape: (batch_size, 2 * hidden_dim)
            final_passage_option = torch.cat([pooled_summary_option, pooled_summary_passage_option], dim=1)
            option_logits.append((final_passage_question * final_passage_option).mean(dim=1, keepdim=True))
        option_logits = torch.cat(option_logits, dim=1)
        option_probs = softmax(option_logits, dim=1)
        best_answer_index = option_probs.max(dim=1)[1]

        if not self.is_judge:
            value_head_input = hidden_passage.detach() if self._detach_value_head else hidden_passage  # TODO: Fix input
            # Shape: (batch_size)
            value = (self._value_head(value_head_input).squeeze(-1) * passage_mask).mean(1)  # TODO: Don't count masked areas in mean!!

        output_dict = {
                "span_start_logits": option_logits,
                "span_start_probs": option_probs,
                "best_span": best_answer_index,
                "value": value if not self.is_judge else None,
                "accuracy": best_answer_index == answer_index if self.is_judge else None  # TODO: Use this as tmp_squad_metrics in Oracle
                }

        # Compute the loss for training.
        if answer_index is not None:
            loss = nll_loss(log_softmax(option_logits, dim=1), answer_index.squeeze(-1))
            if store_metrics:
                self._span_start_accuracy(option_logits, answer_index.squeeze(-1))
            output_dict["loss"] = loss
        return output_dict

    # TODO: Implement per_sample metrics
    def get_metrics(self, reset: bool = False, per_sample: bool = False) -> Dict[str, float]:
        return {'start_acc': self._span_start_accuracy.get_metric(reset)}
