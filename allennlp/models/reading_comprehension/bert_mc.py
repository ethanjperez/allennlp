import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.functional import nll_loss, relu, softmax, log_softmax

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

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
        self._hidden_dim = text_field_embedder.get_output_dim()
        self.answer_type = 'mc'

        if not self.is_judge:
            self._value_head = TimeDistributed(torch.nn.Linear(self._hidden_dim, 1))  # NB: Can make MLP
            self._turn_film_gen = torch.nn.Linear(1, 2 * self._hidden_dim)
            self._film = FiLM()

        self._span_start_accuracy = CategoricalAccuracy()
        self._initializer = initializer

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
        # Precomputation
        a_turn = None
        if not self.is_judge:
            assert(metadata is not None and 'a_turn' in metadata[0])
            a_turn = torch.tensor([sample_metadata['a_turn'] for sample_metadata in metadata]).to(passage['tokens']).unsqueeze(1).float()
        sep_token = metadata[0]['[SEP]'] if '[SEP]' in metadata[0] else self.vocab._token_to_index['bert']['[SEP]']

        # Calculate answer
        option_logits, value = self.compute_logits_and_value(question, passage, options, sep_token, a_turn)
        option_probs = softmax(option_logits, dim=1)
        best_answer_index = option_probs.max(dim=1)[1]

        # Store results
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

    @staticmethod
    def pack_sequences(tokens_1: torch.LongTensor, tokens_2: torch.LongTensor, sep_token: int = None,
                       maxlen: int = 512) -> torch.LongTensor:
        """
        Packs two BERT-formatted sequences into BERT format: [CLS] seq1 tokens [SEP] seq2 tokens [SEP].
        If packed sequence exceeds BERT's max input length, then the first sequence is always truncated.
        """
        assert (tokens_1.dim() == 2) and (tokens_2.dim() == 2), 'pack_sequences only supports 2-dimensional sequences.'
        batch_size = tokens_1.size(0)
        packed_seqs = torch.zeros(batch_size, maxlen, dtype=torch.long, device=tokens_1.device)
        packed_seq_lengths = []
        for i in range(batch_size):
            truncatable_length = tokens_1[i].nonzero().size(0) - 1  # Exclude terminating [SEP]
            required_length = tokens_2[i].nonzero().size(0)  # Exclude [CLS], include separating [SEP]
            seq1_target_length = min(maxlen - required_length, truncatable_length)
            packed_seq_no_padding = torch.cat([tokens_1[i, :seq1_target_length],
                                               (torch.LongTensor() if sep_token is None else torch.LongTensor([sep_token])).to(tokens_1),
                                               tokens_2[i, 1:required_length]], dim=0)
            packed_seq_length = packed_seq_no_padding.size(0)
            packed_seqs[i, :packed_seq_length] = packed_seq_no_padding
            packed_seq_lengths.append(packed_seq_length)
        return packed_seqs[:, :max(packed_seq_lengths)]  # Truncate extra padding from filling in zero matrix

    @staticmethod
    def get_token_type_ids(tokens, sep_token):
        """
        Returns the token type ids, to be used in BERT's segment embeddings
        """
        assert (tokens.dim() in [2, 3]), 'pack_sequences only supports {2,3}-dimensional sequences.'
        orig_size = tokens.size()
        if tokens.dim() == 3:
            tokens = util.combine_initial_dims(tokens)
        sep_token_mask = (tokens == sep_token).long()
        if sep_token_mask.nonzero().size(0) == tokens.size(0):
            return torch.zeros(orig_size, dtype=torch.long)  # Use all zeros if there's 1 [SEP] per sample
        return (sep_token_mask.cumsum(-1) - sep_token_mask).clamp(max=1).view(orig_size)

    @staticmethod
    def tokens_to_bert_input(tokens, sep_token):
        """
        Converts tokens into a BERT-compatible dictionary format
        """
        assert (tokens.dim() in [2, 3]), 'pack_sequences only supports {2,3}-dimensional sequences.'
        bert_input = {'tokens': tokens, 'token-type-ids': BertMC.get_token_type_ids(tokens, sep_token)}
        bert_input['mask'] = util.get_text_field_mask(bert_input).float()
        bert_input['tokens-offsets'] = None
        return bert_input

    def compute_logits_and_value(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                sep_token: int,
                a_turn: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass. Must be implemented by subclass.
        """
        raise NotImplementedError


@Model.register("bert-mc")
class BertMCDCMN(BertMC):
    """
    The SOTA (1/2019) architecture on RACE, from:
    `Dual Co-Matching Network for Multi-choice Reading Comprehension` (https://arxiv.org/pdf/1901.09381.pdf)
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         initializer=initializer,
                         regularizer=regularizer,
                         judge=judge,
                         update_judge=update_judge,
                         reward_method=reward_method,
                         detach_value_head=detach_value_head)
        self._passage_question_attention = BilinearMatrixAttention(self._hidden_dim, self._hidden_dim, use_input_biases=True)
        self._passage_option_attention = BilinearMatrixAttention(self._hidden_dim, self._hidden_dim, use_input_biases=True)
        self._final_passage_question_encoder = TimeDistributed(torch.nn.Linear(2 * self._hidden_dim, self._hidden_dim))
        self._final_question_encoder = TimeDistributed(torch.nn.Linear(2 * self._hidden_dim, self._hidden_dim))
        self._final_passage_option_encoder = TimeDistributed(torch.nn.Linear(2 * self._hidden_dim, self._hidden_dim))
        self._final_option_encoder = TimeDistributed(torch.nn.Linear(2 * self._hidden_dim, self._hidden_dim))
        self._initializer(self)

    def compute_logits_and_value(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                sep_token: int,
                a_turn: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass
        """
        # Get masks and other variables
        passage_mask = util.get_text_field_mask(passage).float()
        question_mask = util.get_text_field_mask(question).float()
        options_mask = util.get_text_field_mask(options).float()
        num_options = options['tokens'].size(1)

        token_type_ids = BertMC.get_token_type_ids(passage['tokens'], sep_token)  # TODO: Use token_type_ids!
        if not self.is_judge:
            # TODO: Use boolean variable passed in to determine if A/B should use Frozen Judge BERT or their own updating BERT
            if self._text_field_embedder._token_embedders['tokens'].requires_grad:
                token_type_ids[:, 0] = a_turn.long().squeeze(1)
        # Shape: (batch_size, passage_length, hidden_dim)  # TODO: Get full BERT output: Set 'tokens-offsets'=None
        hidden_passage = self._text_field_embedder(passage) * passage_mask.unsqueeze(-1)
        hidden_question = self._text_field_embedder(question) * question_mask.unsqueeze(-1)
        hidden_options = self._text_field_embedder(options) * options_mask.unsqueeze(-1)

        # Get dimensions
        batch_size, passage_length, _ = hidden_passage.size()
        _, question_length, _ = hidden_question.size()
        _, _, options_length, _ = hidden_options.size()

        # Debate: Post-BERT agent-based conditioning
        if not self.is_judge:
            turn_film_params = self._turn_film_gen(a_turn)
            turn_gammas, turn_betas = torch.split(turn_film_params, self._hidden_dim, dim=-1)
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

        value = None
        if not self.is_judge:
            value_head_input = hidden_passage.detach() if self._detach_value_head else hidden_passage  # TODO: Fix input
            # Shape: (batch_size)
            value = (self._value_head(value_head_input).squeeze(-1) * passage_mask).mean(1)  # TODO: Don't count masked areas in mean!!

        return option_logits, value


@Model.register("bert-mc-gpt")
class BertMCGPT(BertMC):
    """
    Bert-for-Multiple-Choice, inspired by OpenAI GPT's RACE model. Used with BERT on RACE here:
    `BERT for Multiple Choice Machine Comprehension`: (https://github.com/NoviScl/BERT-RACE/blob/master/BERT_RACE.pdf)
    Applies BERT to each option to get each softmax logit: Logit_i = BERT([CLS] Passage [SEP] Question + Option_i [SEP])
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         initializer=initializer,
                         regularizer=regularizer,
                         judge=judge,
                         update_judge=update_judge,
                         reward_method=reward_method,
                         detach_value_head=detach_value_head)
        self._logit_predictor = torch.nn.Linear(self._hidden_dim, 1)
        self._initializer(self)

    def compute_logits_and_value(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                sep_token: int,
                a_turn: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass
        """
        if not self.is_judge:
            raise NotImplementedError

        # BERT-formatting input
        batch_size, num_options, _ = options['tokens'].size()
        pqo_tokens_list = []
        pqo_token_maxlens = []
        for i in range(num_options):
            qo_tokens = self.pack_sequences(question['tokens'], options['tokens'][:, i])
            pqo_tokens_list.append(self.pack_sequences(passage['tokens'], qo_tokens, sep_token))
            pqo_token_maxlens.append(pqo_tokens_list[i].size(-1))
        pqo_tokens = torch.zeros(batch_size, num_options, max(pqo_token_maxlens), dtype=torch.long, device=passage['tokens'].device)
        for i in range(num_options):
            pqo_tokens[:, i, :pqo_tokens_list[i].size(-1)] = pqo_tokens_list[i]
        pqo = self.tokens_to_bert_input(pqo_tokens, sep_token)

        hidden_pqo = self._text_field_embedder(pqo)
        pred_hidden_a = hidden_pqo[:, :, 0]
        option_logits = self._logit_predictor(pred_hidden_a).squeeze(-1)
        return option_logits, None


@Model.register("bert-mc-pq2a")
class BertMCPQ2A(BertMC):
    """
    The SOTA (1/2019) architecture on RACE, from:
    `Dual Co-Matching Network for Multi-choice Reading Comprehension` (https://arxiv.org/pdf/1901.09381.pdf)
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         initializer=initializer,
                         regularizer=regularizer,
                         judge=judge,
                         update_judge=update_judge,
                         reward_method=reward_method,
                         detach_value_head=detach_value_head)
        self._initializer(self)

    def compute_logits_and_value(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                sep_token: int,
                a_turn: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass
        """
        # NOTE: Add pre-BERT conditioning for a+b training
        # NB: May need to also calculate passage_question['tokens-offsets']
        # TODO: Test with passage > 512 tokens! Find in race stdout.log files
        passage_question_tokens = self.pack_sequences(passage['tokens'], question['tokens'], sep_token)
        passage_question = self.tokens_to_bert_input(passage_question_tokens, sep_token)
        hidden_passage_question = self._text_field_embedder(passage_question)

        # Debate: Post-BERT agent-based conditioning
        value = None
        if not self.is_judge:
            turn_film_params = self._turn_film_gen(a_turn)
            turn_gammas, turn_betas = torch.split(turn_film_params, self._hidden_dim, dim=-1)
            # NB: Check you need to apply passage_mask here
            hidden_passage_question = self._film(hidden_passage_question, 1. + turn_gammas, turn_betas) * passage_question['mask'].unsqueeze(-1)
            value_head_input = hidden_passage_question.detach() if self._detach_value_head else hidden_passage_question  # TODO: Fix input
            # Shape: (batch_size)
            value = (self._value_head(value_head_input).squeeze(-1) * passage_question['mask']).mean(1)  # TODO: Don't count masked areas in mean!!

        predicted_hidden_answer = hidden_passage_question[:, 0]

        options['tokens-offsets'] = None  # To get full BERT output (per wordpiece not word)
        options['token-type-ids'] = BertMC.get_token_type_ids(options['tokens'], sep_token)
        hidden_options = self._text_field_embedder(options)
        encoded_hidden_options = hidden_options[:, :, 0]
        option_logits = (encoded_hidden_options * predicted_hidden_answer.unsqueeze(1)).mean(-1)
        return option_logits, value


@Model.register("bert-mc-q2a")
class BertMCQ2A(BertMC):
    """
    BERT Baseline which uses only the question and answer options to make a prediction.
    Applies BERT to each option alone (without context) to get each softmax logit: Logit_i = BERT([CLS] Option_i [SEP])
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         initializer=initializer,
                         regularizer=regularizer,
                         judge=judge,
                         update_judge=update_judge,
                         reward_method=reward_method,
                         detach_value_head=detach_value_head)
        self._initializer(self)

    def compute_logits_and_value(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                sep_token: int,
                a_turn: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass
        """
        if not self.is_judge:
            raise NotImplementedError

        question['tokens-offsets'] = None  # To get full BERT output (per wordpiece not word)
        question['token-type-ids'] = None  # No segment embeddings necessary here
        hidden_question = self._text_field_embedder(question)
        encoded_hidden_question = hidden_question[:, 0]

        options['tokens-offsets'] = None  # To get full BERT output (per wordpiece not word)
        options['token-type-ids'] = None  # No segment embeddings necessary here
        hidden_options = self._text_field_embedder(options)
        encoded_hidden_options = hidden_options[:, :, 0]
        option_logits = (encoded_hidden_options * encoded_hidden_question.unsqueeze(1)).mean(-1)
        return option_logits, None


@Model.register("bert-mc-p2a")
class BertMCP2A(BertMC):
    """
    BERT Baseline which uses only the question and answer options to make a prediction.
    Applies BERT to each option alone (without context) to get each softmax logit: Logit_i = BERT([CLS] Option_i [SEP])
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         initializer=initializer,
                         regularizer=regularizer,
                         judge=judge,
                         update_judge=update_judge,
                         reward_method=reward_method,
                         detach_value_head=detach_value_head)
        self._initializer(self)

    def compute_logits_and_value(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                sep_token: int,
                a_turn: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass
        """
        if not self.is_judge:
            raise NotImplementedError

        passage['tokens-offsets'] = None  # To get full BERT output (per wordpiece not word)
        passage['token-type-ids'] = None  # No segment embeddings necessary here
        hidden_passage = self._text_field_embedder(passage)
        encoded_hidden_passage = hidden_passage[:, 0]

        options['tokens-offsets'] = None  # To get full BERT output (per wordpiece not word)
        options['token-type-ids'] = None  # No segment embeddings necessary here
        hidden_options = self._text_field_embedder(options)
        encoded_hidden_options = hidden_options[:, :, 0]
        option_logits = (encoded_hidden_options * encoded_hidden_passage.unsqueeze(1)).mean(-1)
        return option_logits, None


@Model.register("bert-mc-a")
class BertMCA(BertMC):
    """
    BERT Baseline which uses only answer options to make a prediction.
    Applies BERT to each option alone (without context) to get each softmax logit: Logit_i = BERT([CLS] Option_i [SEP])
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         initializer=initializer,
                         regularizer=regularizer,
                         judge=judge,
                         update_judge=update_judge,
                         reward_method=reward_method,
                         detach_value_head=detach_value_head)
        self._logit_predictor = torch.nn.Linear(self._hidden_dim, 1)
        self._initializer(self)

    def compute_logits_and_value(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                sep_token: int,
                a_turn: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass
        """
        if not self.is_judge:
            raise NotImplementedError

        options['tokens-offsets'] = None  # To get full BERT output (per wordpiece not word)
        options['token-type-ids'] = None  # No segment embeddings necessary here
        hidden_options = self._text_field_embedder(options)
        encoded_hidden_options = hidden_options[:, :, 0]
        option_logits = self._logit_predictor(encoded_hidden_options).squeeze(-1)
        return option_logits, None
