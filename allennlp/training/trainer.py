import copy
import json
import logging
import math
import os
import time
import re
import datetime
import traceback
import warnings
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, NamedTuple

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import (dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metrics import Average
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("default")
class Trainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 debate_mode: List[str],
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 eval_mode: bool = False,
                 breakpoint_level: int = 0,
                 id_to_oracle_filename: str = None,
                 accumulation_steps: int = 1) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a ``DataIterator``, and uses the supplied ``Optimizer`` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``int``, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``PytorchLRScheduler``, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.  To support updating the learning
            rate on every batch, this can optionally implement ``step_batch(batch_num_total)`` which
            updates the learning rate given the batch number.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : ``int``, optional, (default = ``None``)
            If defined, how often to log the average batch size.
        """
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.iterator = iterator
        self._debate_mode = debate_mode
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.train_data = train_dataset
        self._validation_data = validation_dataset
        self._eval_mode = eval_mode
        self._breakpoint_level = breakpoint_level
        self._accumulation_steps = accumulation_steps
        self._using_bert = hasattr(self.model, '_text_field_embedder') and \
                   hasattr(self.model._text_field_embedder, 'token_embedder_tokens') and \
                   'bert_token_embedder' in str(type(self.model._text_field_embedder.token_embedder_tokens))
        self._answer_id_tokens = ['1st', '2nd', '3rd', '4th'] if (self.model.answer_type == 'mc') else None
        self._mc_dataset_reader = 'answer_index' in self.train_data[0].fields

        self._id_to_oracle_is_complete = (id_to_oracle_filename is not None)
        self._id_to_oracle = {}
        if id_to_oracle_filename is not None:
            with open(id_to_oracle_filename) as id_to_oracle_file:
                self._id_to_oracle = json.load(id_to_oracle_file)

        self._trainer_metrics = {}
        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError('{} is an invalid value for "patience": it must be a positive integer '
                                     'or None (if you want to disable early stopping)'.format(patience))

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        self._checkpointer = Checkpointer(serialization_dir,
                                          keep_serialized_model_every_num_seconds,
                                          num_serialized_models_to_keep)

        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._tensorboard = TensorboardWriter(
                get_batch_num_total=lambda: self._batch_num_total,
                serialization_dir=serialization_dir,
                summary_interval=summary_interval,
                histogram_interval=histogram_interval,
                should_log_parameter_statistics=should_log_parameter_statistics,
                should_log_learning_rate=should_log_learning_rate)

        self._log_batch_size_period = log_batch_size_period

        self._last_log = 0.0  # time of last logging

        # Enable activation logging.
        if histogram_interval is not None:
            self._tensorboard.enable_activation_logging(self.model)

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def _forward(self, batch_group: List[TensorDict], model: Model) -> Dict[str, Any]:
        """
        Does a forward pass on the appropriate model and device(s) and returns the result.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = model(**batch)
        return output_dict

    def _slice_batch(self, batch: TensorDict, idxs: slice) -> TensorDict:
        """
        Slices and copies an existing batch into a smaller batch. Repeats the slice num_repeat times.
        Use a single integer or a slice for idxs to get sample(s) in the batch.
        """
        sliced_batch = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                sliced_batch[k] = {}
                for inner_k, inner_v in v.items():
                    sliced_batch[k][inner_k] = inner_v[idxs]
            elif isinstance(v, torch.Tensor):
                sliced_batch[k] = v[idxs]
            elif isinstance(v, list):
                sliced_batch[k] = v[idxs]
            elif isinstance(v, bool):
                sliced_batch[k] = v
            else:
                raise NotImplementedError('Unimplemented slice for key, value:', k, v)
        return sliced_batch

    def _create_batch_from_idx(self, batch: TensorDict, idx: int, num_repeat) -> TensorDict:
        """
        Slices and copies an existing batch into a smaller batch. Repeats the slice num_repeat times.
        Use a single integer to get sample in the batch.
        """
        sliced_batch = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                sliced_batch[k] = {}
                for inner_k, inner_v in v.items():
                    sliced_batch[k][inner_k] = inner_v[idx].repeat(num_repeat, *[1 for _ in range(inner_v[idx].dim())])
            elif isinstance(v, torch.Tensor):
                sliced_batch[k] = v[idx].repeat(num_repeat, 1)
            elif isinstance(v, list):
                sliced_batch[k] = [copy.deepcopy(v[idx]) for _ in range(num_repeat)]
            else:
                raise NotImplementedError('Unimplemented slice for key, value:', k, v)
        return sliced_batch

    def _update_trainer_metrics(self, metric_name: str, new_value: torch.tensor) -> None:
        """
        Updates the trainer's metrics for metric_name with new_value
        """
        self._trainer_metrics[metric_name] = self._trainer_metrics.get(metric_name, Average())
        self._trainer_metrics[metric_name](new_value)

    def _modify_input_passage(self, batch, sent_choice_input_masks, mask_tok_val, mod_type):
        """
        Modifies a passage according to sentence selections made elsewhere.
        Supports e.g. masking and deleting portions of the passage.
        Used before passing the batch to the judge.
        """
        mod_type = mod_type.lower()
        has_chars = 'token_characters' in batch['passage'].keys()
        # TODO: BERT: Ensure this works for BERT
        if mod_type == 'delete':
            # NB: SQuAD: Can also make deletion-based. Must modify span_end, span_start, and metadata then.
            # NB: RACE: Better to modify metadata here
            post_delete_toks = torch.zeros_like(batch['passage']['tokens'])
            if has_chars:
                post_delete_tok_chars = torch.zeros_like(batch['passage']['token_characters'])
            for idx in range(batch['passage']['tokens'].size(0)):
                toks = batch['passage']['tokens'][idx]
                reveal_idxs = (toks * (1. - sent_choice_input_masks[idx])).nonzero().squeeze(-1)
                post_delete_toks[idx][:toks[reveal_idxs].size(0)] = toks[reveal_idxs]
                if has_chars:
                    tok_chars = batch['passage']['token_characters'][idx]
                    post_delete_tok_chars[idx][:toks[reveal_idxs].size(0)] = tok_chars[reveal_idxs]
            # Detaching just in case to prevent gradient flow back to agents modifying passage
            batch['passage']['tokens'] = post_delete_toks.detach()
            if has_chars:
                batch['passage']['token_characters'] = post_delete_tok_chars.detach()
        elif mod_type == 'mask':
            batch['passage']['tokens'] = batch['passage']['tokens'].masked_fill(sent_choice_input_masks.byte(), mask_tok_val)
            if has_chars:
                # NB: Check 0 is character-level padding too / check for correctness
                batch['passage']['token_characters'] = batch['passage']['token_characters'].masked_fill(sent_choice_input_masks.byte().unsqueeze(-1), 0)
        else:
            raise NotImplementedError('Modifying passages via mod_type ' + mod_type + ' not supported.')
        return batch

    def _add_debate_metrics(self, output_dict, sent_output_idxs, sent_choice_idxs, num_turns, turn_str):
        """
        Add various metrics related to the batch's debate (excluding losses)
        """
        # Add stats on if J chosen a sentence from A or B
        # NB: Not useful for RACE (except to check if there's a bug), may be incorrect for SQuAD deletion setting
        j_span_start_sent = sent_output_idxs.gather(1, output_dict['best_span'][:, :1].to(sent_output_idxs.device))
        j_span_end_sent = sent_output_idxs.gather(1, output_dict['best_span'][:, 1:].to(sent_output_idxs.device))
        j_num_ab_sents_chosen = torch.zeros_like(j_span_start_sent).float()
        for turn in range(num_turns):
            j_sent_chosen = ((j_span_start_sent <= sent_choice_idxs[turn]) * (sent_choice_idxs[turn] <= j_span_end_sent)).float()
            self._update_trainer_metrics('j_sent_chosen' + turn_str[turn], j_sent_chosen.mean())
            j_num_ab_sents_chosen += j_sent_chosen
        j_chose_no_ab_sents = (j_num_ab_sents_chosen == 0).float()
        self._update_trainer_metrics('j_sent_chosen_not_a_or_b', j_chose_no_ab_sents.mean())

    @staticmethod
    def _print_debate(batch, num_sents, debate_mode, sent_choice_output_masks, sent_choice_idxs, output_dict,
                      j_scores, sc_diffs=None):
        """
        Neatly prints all debates from a batch.
        """
        bsz = batch['passage']['tokens'].size(0)
        for sample_no in range(bsz):
            if bool(num_sents[sample_no] >= 3):
                print('\n***ID***\n', batch['metadata'][sample_no]['id'])
                print('\n***Passage***\n', ' '.join(batch['metadata'][sample_no]['passage_tokens']))
                print('\n***Question***\n', ' '.join(batch['metadata'][sample_no]['question_tokens']))
                toks = batch['metadata'][sample_no]['passage_tokens']
                if 'options' in batch:
                    print('\n***Options***\n', [' '.join(batch['metadata'][sample_no]['options_tokens'][i]) for i in range(4)])
                    true_answer_index = batch['answer_index'][sample_no]
                    print('\n***True Answer***\n', true_answer_index.item(), ' '.join(batch['metadata'][sample_no]['options_tokens'][true_answer_index]))
                    best_answer_index = output_dict['best_answer_index'][sample_no]
                    print('\n***Predicted Answer***\n', best_answer_index.item(), ' '.join(batch['metadata'][sample_no]['options_tokens'][best_answer_index]))
                else:
                    print('\n***Answers***\n', [answer if isinstance(answer, str) else ' '.join(answer) for answer in batch['metadata'][sample_no]['answer_texts']])
                    if 'best_span' in output_dict:
                        print(' '.join(toks[output_dict['best_span'][sample_no][0]:output_dict['best_span'][sample_no][1] + 1]))
                for turn, method in enumerate(debate_mode[0]):
                    turn_sent_output_idxs = sent_choice_output_masks[turn][sample_no].nonzero().squeeze(-1)
                    sent_str = 'None'
                    if len(turn_sent_output_idxs) > 0:
                        turn_sent_start_output_idx = turn_sent_output_idxs.min()
                        turn_sent_end_output_idx = turn_sent_output_idxs.max() + 1
                        sent_str = ' '.join(toks[turn_sent_start_output_idx: turn_sent_end_output_idx])
                    print('\n---', method.upper(), '--- Sentence', int(sent_choice_idxs[turn][sample_no]), '\n', sent_str)
                print('\n--- J --- EM / F1 / SSP / SC_DIFF\n',
                      j_scores.get('em', 'N/A'), '/',
                      j_scores.get('f1', 'N/A'), '/',
                      j_scores.get('ssp', 'N/A'), '/',
                      float(sc_diffs[sample_no]) if sc_diffs is not None else 'N/A')
        return

    def _print_tokens(self, tokens):
        """
        Prints BERT wordpiece tokens from token indices.
        """
        if self._using_bert:
            print(' '.join([self.get_index_token(tok.item()) for tok in tokens]))
        return

    def _print_input_span(self, batch: TensorDict, sample_no: int, input_span: Tuple[int, int]):
        """
        Prints the token strings of the given span defined on the input level (e.g. word or wordpiece level).
        """
        self._print_tokens(batch['passage']['tokens'][sample_no, input_span[0]: input_span[1] + 1])
        return

    @staticmethod
    def _print_output_span(batch: TensorDict, sample_no: int, output_span: Tuple[int, int]):
        """
        Prints the token strings of the given span defined on the output level (word level).
        """
        print(' '.join(batch['metadata'][sample_no]['passage_tokens'][output_span[0]: output_span[1] + 1]))
        return

    @staticmethod
    def _output_to_input_idx(batch: TensorDict, sample_no: int, output_idx: int) -> int:
        """
        Converts a index on the output (always word-level) to one on the input (often sub-word level, shifted, etc.)
        """
        return batch['passage']['tokens-offsets'][sample_no][output_idx].item()

    @staticmethod
    def _output_to_input_span(batch: TensorDict, sample_no: int, output_span: Tuple[int, int]) -> Tuple[int, int]:
        """
        Converts a span over the output (always word-level) to one over the input (often sub-word level, shifted, etc.)
        """
        offsets = batch['passage']['tokens-offsets'][sample_no]
        return offsets[output_span[0]].item(), offsets[output_span[1]].item()

    def get_index_token(self, index):
        """
        Returns the token corresponding to a vocab index for this model.
        Within Training, this should always be used to get the index (i.e., instead of the model directly)
        """
        if self._using_bert:
            return self.model.vocab._index_to_token['bert'][index]
        else:
            raise NotImplementedError  # TODO: Non-BERT models

    def get_token_index(self, token: str):
        """
        Returns the index of a token, regardless of the text field embedder.
        Within Training, this should always be used to get the index (i.e., instead of the model directly)
        """
        if self._using_bert:
            return self.model.vocab._token_to_index['bert'][token]
        else:
            return self.model.vocab.get_token_index(token)

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool, debate_mode: List[str] = None) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        # If overriding default passage choosing method
        if debate_mode is None:
            debate_mode = self._debate_mode

        # Optional debugging sanity check
        if (not self._mc_dataset_reader) and (self._breakpoint_level >= 1) and for_training:
            for batch in batch_group:
                for i in range(batch['passage']['tokens'].size(0)):
                    print('ID:', batch['metadata'][i]['id'], ' ...')
                    char_span_start = batch['metadata'][i]['token_offsets'][batch['span_start'][i]][0]
                    char_span_end = batch['metadata'][i]['token_offsets'][batch['span_end'][i]][1]
                    answer_text = batch['metadata'][i]['answer_texts'][0]
                    post_processing_answer_text = batch['metadata'][i]['original_passage'][char_span_start: char_span_end]
                    answer_processing_error = not (answer_text in post_processing_answer_text)
                    if self.model.answer_type == 'mc':
                        answer_processing_error = (answer_text != post_processing_answer_text) or (answer_text not in self._answer_id_tokens)
                    if answer_processing_error:  # Print: unexpected mismatch with true answer
                        self._print_tokens(batch['passage']['tokens'][i, :])
                        print('answer_text =', answer_text)
                        print('post_processing_answer_text =', post_processing_answer_text)

        # Set output_dict['loss'] to do gradient descent on.
        if debate_mode[0] == "f":  # Full passage training: Normal SL training
            output_dict = self._forward(batch_group, self.model)
        else:  # Training on subset of sentence (judge or debate training)
            outputs = []
            # TODO(Sidd): Distribute this loop across GPUs. See training_util.data_parallel
            for batch in batch_group:
                outputs.append(self.debate_batch_loss(batch, for_training, debate_mode))
            # Taken from training_util.data_parallel
            losses = torch.cat([output['loss'].unsqueeze(0) for output in outputs], 0)
            output_dict = {'loss': losses.mean()}  # NB(Sidd): Are all batches exactly same # of samples regardless of number of GPUs? Otherwise .mean() is incorrect
        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def debate_batch_loss(self, batch: TensorDict, for_training: bool, debate_mode: List[str]) -> torch.Tensor:
        """
        Does a debate-style forward pass on a single batch in the group
        """
        # Set a few useful variables/aliases
        debater = None if self.model.is_judge else self.model
        judge = self.model if self.model.is_judge else self.model.judge
        mc = (self.model.answer_type == 'mc')
        mod_type = 'delete' if mc else 'mask'
        num_rounds = len(debate_mode)
        assert num_rounds <= 1, 'No implementation yet for # rounds =' + str(num_rounds)
        num_turns = len(debate_mode[0])
        bsz, input_dim = batch['passage']['tokens'].size()
        output_dim = batch['passage']['tokens-offsets'].size(1)  # output_dim < input_dim if using e.g. wordpiece
        mask_tok_val = self.get_token_index('[MASK]') if self._using_bert else self.get_token_index('.')
        a_turn = {turn: debate_mode[0][turn] == 'a' for turn in range(len(debate_mode[0]))}
        turn_str = {turn: "_turn_" + str(turn) + "_agent_" + debate_mode[0][turn] for turn in range(num_turns)}
        sl_debate = (debater is not None) and (debater.reward_method.startswith('sl'))
        debate_mode_with_eval_only_turns = debate_mode[0] if not sl_debate else debate_mode[0].replace('b', 'Bb').replace('a', 'Aa')

        # Add token info to batch for BERT
        if self._using_bert:
            for i in range(bsz):
                batch['metadata'][i]['[SEP]'] = self.model.vocab._token_to_index['bert']['[SEP]']

        # NOTE: Move to GPU if slow
        # Ensure Judge receives question
        question_input_mask = torch.zeros(bsz, input_dim, dtype=torch.long)
        question_output_mask = torch.zeros(bsz, output_dim, dtype=torch.long)
        if (not self._mc_dataset_reader) and ('question_span' in batch['metadata'][0]):
            for i in range(bsz):
                question_output_span = batch['metadata'][i]['question_span']
                question_output_mask[i, question_output_span[0]: question_output_span[1]+1] = 1.
                question_input_span = self._output_to_input_span(batch, i, question_output_span)
                question_input_mask[i, question_input_span[0]: question_input_span[1]+1] = 1.
                if self._breakpoint_level >= 1:
                    self._print_output_span(batch, i, question_output_span)
                    self._print_input_span(batch, i, question_input_span)
        required_text_output_mask = question_output_mask
        required_text_input_mask = question_input_mask

        # Ensure Judge receives answers. Limit where Judge can answer (if applicable)
        passage_output_mask = nn_util.get_text_field_mask(batch['passage'], 0)
        judge_output_mask = (passage_output_mask - question_output_mask).clamp(min=0)
        if (not self._mc_dataset_reader) and mc and ('answer_choice_spans' in batch['metadata'][0]):
            judge_output_mask = torch.zeros(bsz, output_dim, dtype=torch.long)
            pos_answer_output_mask = torch.zeros(bsz, output_dim, dtype=torch.long)
            pos_answer_input_mask = torch.zeros(bsz, input_dim, dtype=torch.long)
            for i in range(bsz):
                answer_choice_output_spans = batch['metadata'][i]['answer_choice_spans']
                answer_choice_input_spans = [self._output_to_input_span(batch, i, out_span) for out_span in answer_choice_output_spans]
                assert len(answer_choice_output_spans) == len(self._answer_id_tokens), 'Must provide ' + len(self._answer_id_tokens) + ' answer indices in metadata:' + batch['metadata'][i]
                for answer_choice_output_span, answer_choice_input_span in zip(answer_choice_output_spans, answer_choice_input_spans):
                    judge_output_mask[i, answer_choice_output_span[1]] = 1.  # Judge target is always span end for MC
                    pos_answer_output_mask[i, answer_choice_output_span[0]: answer_choice_output_span[1]+1] = 1.
                    pos_answer_input_mask[i, answer_choice_input_span[0]: answer_choice_input_span[1]+1] = 1.
            required_text_output_mask += pos_answer_output_mask  # Prevent debaters from selecting answers
            required_text_input_mask += pos_answer_input_mask  # Prevent debaters from selecting answers
        if self._using_bert:
            required_bert_tokens = {'[CLS]', '[SEP]'}
            for i in range(bsz):
                for j, output_token in enumerate(batch['metadata'][i]['passage_tokens']):
                    if (output_token in required_bert_tokens) and (j < output_dim):
                        required_text_output_mask[i, j] = 1.  # NB: Mask will change after deletion (for final [SEP]).
                        required_text_input_mask[i, self._output_to_input_idx(batch, i, j)] = 1.
        required_text_output_mask = required_text_output_mask.clamp(max=1)  # In case of double-counting (which shouldn't happen)
        required_text_input_mask = required_text_input_mask.clamp(max=1)  # In case of double-counting (which shouldn't happen)  TODO: Delete if unused

        # Calculate where debaters can select sentences to quote/delete
        punct_tokens = {'.', '?', '!'}
        debate_choice_output_mask = torch.zeros(bsz, output_dim, dtype=torch.long)
        debate_choice_input_mask = torch.zeros(bsz, input_dim, dtype=torch.long)
        for i in range(bsz):
            for j, output_token in enumerate(batch['metadata'][i]['passage_tokens']):
                if (output_token in punct_tokens) and (j < output_dim):
                    debate_choice_output_mask[i, j] = 1.  # IndexError: index 489 is out of bounds for dimension 0 with size 479. 88% through validation, with random sentence removed.
                    debate_choice_input_mask[i, self._output_to_input_idx(batch, i, j)] = 1.
        # Force last non-padding token to be an eos token in the mask
        for i in range(bsz):
            last_output_token_idx = min(len(batch['metadata'][i]['passage_tokens']), output_dim) - 1
            debate_choice_output_mask[i, last_output_token_idx] = 1.
            last_input_token_idx = self._output_to_input_idx(batch, i, last_output_token_idx)
            debate_choice_input_mask[i, last_input_token_idx] = 1.
        debate_choice_output_mask *= (1. - required_text_output_mask)
        debate_choice_input_mask *= (1. - required_text_input_mask)

        # Calculate number of choosable input/passage sentences
        num_output_sents = debate_choice_output_mask.sum(1)
        num_input_sents = debate_choice_input_mask.sum(1)
        assert nn_util.tensors_equal(num_output_sents, num_input_sents), 'Error: Discrepancy in # of output and input sentences:' + str(num_output_sents) + ', ' + str(num_input_sents)
        num_sents = num_output_sents
        self._update_trainer_metrics('num_sents', num_sents.float().mean())

        # NOTE: Padding regions have sent_idxs == num_sents. Required regions have -1.
        sent_output_idxs = debate_choice_output_mask.cumsum(1) - debate_choice_output_mask
        sent_output_idxs = sent_output_idxs.masked_fill(required_text_output_mask.byte(), -1.)
        sent_output_idxs = sent_output_idxs.masked_fill(sent_output_idxs == num_sents.unsqueeze(-1), -1.)
        sent_input_idxs = debate_choice_input_mask.cumsum(1) - debate_choice_input_mask
        sent_input_idxs = sent_input_idxs.masked_fill(required_text_input_mask.byte(), -1.)
        sent_input_idxs = sent_input_idxs.masked_fill(sent_input_idxs == num_sents.unsqueeze(-1), -1.)
        if not mc:
            sent_answer_idx = sent_output_idxs.gather(1, batch['span_start'].to(sent_input_idxs.device))  # TODO: Verify

        # Execute player turns to determine mask.
        sent_choice_idxs = []
        sent_choice_probs = []
        values = []  # Add -1 * torch.ones(bsz) if no value prediction made
        for debate_mode_with_eval_only_turns_idx, method in enumerate(debate_mode_with_eval_only_turns):
            # NB: Refactor a player turn into one function
            turn = len(sent_choice_idxs)  # Excludes eval only turns
            next_method = ''
            if (debate_mode_with_eval_only_turns_idx + 1) < len(debate_mode_with_eval_only_turns):
                next_method = debate_mode_with_eval_only_turns[debate_mode_with_eval_only_turns_idx + 1]
            is_eval_only_turn = sl_debate and (method in ['A', 'B']) and (next_method == method.lower())
            # Variables that must be set after each turn
            sent_choice_idx = None
            sent_choice_prob = None
            value = None
            sc_diffs = None  # Optional to fill in each turns, resets every turn
            if method == 'r':  # Random selection. NB: Make without replacement policy 'R'
                sent_choice_idx = (torch.rand_like(num_sents.float()) * (num_sents.float())).trunc().long().unsqueeze(1)
                sent_choice_prob = torch.ones(bsz) / (num_sents.float())
                value = -1 * torch.ones(bsz)
            elif method == 'g':  # Ground truth, answer-containing selection
                assert not mc, 'RACE does not provide "Ground Truth" answer-supporting sentences. bAbI tasks may, but the ground truth policy is not yet implemented.'
                sent_choice_idx = sent_answer_idx
                sent_choice_prob = torch.ones(bsz)
                value = -1 * torch.ones(bsz)
            elif (method in ['A', 'B']) and self._id_to_oracle_is_complete:  # A/B oracle selection (loaded)
                oracle_infos = [self._id_to_oracle[md['id']] for md in batch['metadata']]
                sc_diffs = [oracle_info['sc_diff'] for oracle_info in oracle_infos]
                sent_choice_idx = torch.LongTensor([oracle_info['sent_choice_idx'] for oracle_info in oracle_infos]).unsqueeze(1)
                sent_choice_prob = torch.ones(bsz)
                value = torch.LongTensor([oracle_info['value'] for oracle_info in oracle_infos]).unsqueeze(1)
            elif (method in ['A', 'B']) and (not self._id_to_oracle_is_complete):  # A/B oracle selection (computed)  # TODO: Fix for BERT
                oracle_func = max if method == 'A' else min  # NOTE: Modify if adding another oracle method
                oracle_eval_method = 'ssp'
                # oracle_eval_method = 'ssp' if mc else 'f1'  # NOTE: Only other option is 'em'
                # if (debater is not None) and ((sl_debate and debater.reward_method == 'sl-ssp') or ((not sl_debate) and debater.reward_method == 'ssp')):
                #     oracle_eval_method = 'ssp'
                # NOTE: Set below to None to make oracle selection simultaneous with other selections
                past_sent_choice_idxs = torch.cat(sent_choice_idxs, 1) if len(sent_choice_idxs) > 0 else None
                opt_sent_idxs = []
                sc_diffs = []  # NOTE: Only stores sc_diffs for most recent oracle run
                oracle_values = []
                judge_was_training = judge.training
                judge.eval()
                for sample_no in range(bsz):
                    # Batch together all possible next outcomes for a sample
                    # RACE: Removes Oracle choices selecting answer sentences (which'll always be shown)
                    num_sent_options = num_sents[sample_no]
                    oracle_batch = self._create_batch_from_idx(batch, sample_no, num_sent_options)
                    oracle_batch['store_metrics'] = False  # Do not update judge metrics
                    oracle_sent_choice_idxs = torch.arange(num_sent_options).unsqueeze(1)
                    if past_sent_choice_idxs is not None:
                        past_idxs_repeat = past_sent_choice_idxs[sample_no].repeat(num_sent_options, 1)
                        oracle_sent_choice_idxs = torch.cat([past_idxs_repeat, oracle_sent_choice_idxs], 1)

                    # Mask Judge's input
                    oracle_sent_choice_input_masks = [sent_input_idxs[sample_no].unsqueeze(0).expand(num_sent_options, -1) == oracle_sent_choice_idxs[:,i].unsqueeze(1) for i in range(turn + 1)]
                    oracle_all_sent_choice_input_mask = torch.stack(oracle_sent_choice_input_masks).sum(0)
                    oracle_all_sent_choice_input_mask = oracle_all_sent_choice_input_mask / (oracle_all_sent_choice_input_mask.clamp(min=1))  # Differentiable clamp to max=1
                    oracle_batch = self._modify_input_passage(oracle_batch, oracle_all_sent_choice_input_mask, mask_tok_val, mod_type)

                    # Get judge results (May require multiple batches)
                    # TODO: Check judge_output_mask is as expected. TODO: Check this gets sliced appropriately and included in oracle_batch
                    if not self._mc_dataset_reader:
                        oracle_batch['valid_output_mask'] = judge_output_mask[sample_no].unsqueeze(0).expand(num_sent_options, -1)
                    # NB: Slice batch based on batch_size. Do several separate forward passes.
                    num_oracle_batch_slices = math.ceil(num_sent_options.item() / float(bsz))
                    oracle_output_dict = None
                    for oracle_batch_slice_num in range(num_oracle_batch_slices):
                        feed_slice = slice(oracle_batch_slice_num * bsz, (oracle_batch_slice_num + 1) * bsz)
                        oracle_batch_slice = self._slice_batch(oracle_batch, feed_slice)
                        oracle_batch_slice_output_dict = self._forward([oracle_batch_slice], judge)
                        # Add results to overall results dictionary
                        if oracle_output_dict is None:
                            oracle_output_dict = oracle_batch_slice_output_dict
                        else:
                            for k, v in oracle_batch_slice_output_dict.items():
                                if isinstance(v, torch.Tensor) and v.dim() > 0:
                                    oracle_output_dict[k] = torch.cat([oracle_output_dict[k], oracle_batch_slice_output_dict[k]], dim=0)
                                else:
                                    if k in oracle_output_dict.keys():
                                        oracle_output_dict.pop(k)
                    if not self._mc_dataset_reader:
                        oracle_batch.pop('valid_output_mask')

                    if oracle_eval_method == 'ssp':
                        if self._mc_dataset_reader:
                            oracle_metrics = [oracle_output_dict['option_probs'][i, oracle_batch['answer_index'][i]].item() for i in range(oracle_output_dict['option_probs'].size(0))]
                        else:
                            oracle_metrics = [oracle_output_dict['span_start_probs'][i, oracle_batch['span_start'][i]].item() for i in range(oracle_output_dict['span_start_probs'].size(0))]
                    else:
                        oracle_metrics = oracle_output_dict[oracle_eval_method].tolist()
                    opt_sc = float(oracle_func(oracle_metrics))
                    oracle_values.append(opt_sc)
                    opt_sent_idxs.append(oracle_metrics.index(opt_sc))
                    baseline_sc = sum(oracle_metrics) / len(oracle_metrics)
                    # NB: Hard-coding different baseline score based on debate_mode
                    if debate_mode == 'gb':  # No sentence choice is baseline
                        baseline_sc = oracle_metrics[past_sent_choice_idxs[sample_no, 0]]
                    # TODO: Add saved sc_diffs for acc/em/f1
                    sc_diffs.append(baseline_sc - opt_sc)
                if judge_was_training:
                    judge.train()

                sent_choice_idx = torch.LongTensor(opt_sent_idxs).unsqueeze(1)
                sent_choice_prob = torch.ones(bsz)
                value = torch.FloatTensor(oracle_values)
            elif method in ['a', 'b']:  # A/B trained selection
                assert debater is not None, 'Cannot use debate method ' + method + ' without debate agents!'
                # Add some debater-specific batch info. NB: 'metadata' usually optional but will now cause error if missing
                for batch_idx in range(bsz):
                    batch['metadata'][batch_idx]['a_turn'] = a_turn[turn]
                if self._mc_dataset_reader:
                    raise NotImplementedError
                batch['valid_output_mask'] = debate_choice_output_mask

                # Debate forward pass
                debater_output_dict = self._forward([batch], debater)
                debater.get_metrics(reset=True)  # A/B metrics currently meaningless, so clear

                # Remove debater-specific batch info.
                for batch_idx in range(bsz):
                    batch['metadata'][batch_idx].pop('a_turn')
                batch.pop('valid_output_mask')

                # Sample from policy's sentence-level distribution
                eos_choice_distribution = debater_output_dict['span_start_probs']  # TODO: Make BERT MC models capable of outputting distributions over EOS tokens
                word_choice_idx = torch.multinomial(eos_choice_distribution, 1) if for_training else torch.argmax(eos_choice_distribution, dim=1, keepdim=True)
                sent_choice_idx = sent_output_idxs.gather(1, word_choice_idx.to(sent_output_idxs.device))

                if sl_debate:  # SL: No sampling for prediction probs. Forcibly choose Oracle's prediction
                    sl_sampling_acc = (sent_choice_idx == oracle_sent_choice_idx).float()
                    self._update_trainer_metrics('sl_sampling_acc' + turn_str[turn], sl_sampling_acc.mean())
                    sc_diffs = torch.Tensor(sc_diffs)
                    for i in range(-1, 10):
                        thres_start = i / 10.
                        thres_end = (i + 1) / 10.
                        thres_start_mask = (sc_diffs.abs() > thres_start).float()
                        thres_end_mask = (thres_end >= sc_diffs.abs()).float()
                        oracle_sc_diff_in_thres_idxs = (thres_start_mask * thres_end_mask).nonzero()
                        self._update_trainer_metrics('sl_num_per_batch_where_' + str(thres_end) + '>=MaxScoreDrop>' + str(thres_start) + turn_str[turn], torch.tensor(float(len(oracle_sc_diff_in_thres_idxs))))
                        for idx in oracle_sc_diff_in_thres_idxs:
                            self._update_trainer_metrics('sl_sampling_acc_where_' + str(thres_end) + '>=MaxScoreDrop>' + str(thres_start) + turn_str[turn], sl_sampling_acc[idx])
                    self._update_trainer_metrics('sl_non_default_preds_per_batch' + turn_str[turn], (sent_choice_idx != 0).float().sum())
                    sent_choice_output_mask = (sent_output_idxs == oracle_sent_choice_idx)  # Force model to choose oracle's choice. Necessary to get the right probability for the NLL loss.
                else:  # RL: Use prob of sampled sentence to calculate loss
                    sent_choice_output_mask = (sent_output_idxs == sent_choice_idx)

                sent_choice_prob = (eos_choice_distribution.to(sent_choice_output_mask.device) * sent_choice_output_mask.to(eos_choice_distribution.dtype)).sum(1)
                value = debater_output_dict['value']
            else:
                raise NotImplementedError('Unimplemented answer selection debate method', method)

            # Apply masks / use probs only after eval-only turns are finished
            if is_eval_only_turn and sl_debate:
                oracle_sent_choice_idx = sent_choice_idx
                if not self._id_to_oracle_is_complete:
                    for sample_no in range(bsz):
                        self._id_to_oracle[batch['metadata'][sample_no]['id']] = {
                            'sent_choice_idx': oracle_sent_choice_idx[sample_no].item(),
                            'value': value[sample_no].item(),
                            'sc_diff': sc_diffs[sample_no],
                        }
                continue

            assert (sent_choice_idx is not None) and (sent_choice_prob is not None) and (value is not None), \
                'Error: Did not fill all necessary variables for turn selection.'
            sent_choice_idxs.append(sent_choice_idx)
            sent_choice_probs.append(sent_choice_prob)
            values.append(value.cpu())

            if not mc:
                answer_sent_chosen = (sent_choice_idx == sent_answer_idx).float()  # NOTE: Assumes answer does not cross ./?/! boundary
                self._update_trainer_metrics('answer_sent_chosen' + turn_str[turn], answer_sent_chosen.mean())

        # Mask Judge's input
        sent_choice_input_masks = [(sent_input_idxs == sent_choice_idx) for sent_choice_idx in sent_choice_idxs]
        all_sent_choice_input_mask = torch.stack(sent_choice_input_masks).sum(0)
        all_sent_choice_input_mask = all_sent_choice_input_mask / (all_sent_choice_input_mask.clamp(min=1))  # Differentiable clamp to max=1.  TODO: Swap these for all_sent_choice_input_mask.clamp(max=1)
        batch = self._modify_input_passage(batch, all_sent_choice_input_mask, mask_tok_val, mod_type)

        # Judge forward pass
        if not self._mc_dataset_reader:
            batch['valid_output_mask'] = judge_output_mask  # TODO: Check judge_output_mask is as expected
        output_dict = self._forward([batch], judge)
        if not self._mc_dataset_reader:
            batch.pop('valid_output_mask')

        # Judge metrics
        j_scores = {
            'em': output_dict['em'].to(sent_choice_probs[0]) if 'em' in output_dict else None,
            'f1': output_dict['f1'].to(sent_choice_probs[0]) if 'f1' in output_dict else None,
        }
        if self._mc_dataset_reader:
            j_scores['ssp'] = torch.tensor([output_dict['option_probs'][i, batch['answer_index'][i]] for i in range(bsz)])
        else:
            j_scores['ssp'] = torch.tensor([output_dict['span_start_probs'][i, batch['span_start'][i]] for i in range(bsz)])

        print_every = 1 if mc else 20
        if self._eval_mode and ((self._batch_num_total % print_every) == 0):
            sent_choice_output_masks = [(sent_output_idxs == sent_choice_idx) for sent_choice_idx in sent_choice_idxs]
            self._print_debate(batch, num_sents, debate_mode, sent_choice_output_masks, sent_choice_idxs, output_dict,
                               j_scores, sc_diffs)

        # Debate losses
        if debater is not None:
            self._add_debate_metrics(output_dict, sent_output_idxs, sent_choice_idxs, num_turns, turn_str)
            j_score = j_scores[debater.reward_method].detach()  # Judge shouldn't get gradients through j_score, used to reward A/B
            # Initialize loss (including J's supervised loss if necessary)
            output_dict = output_dict if self.model.update_judge else {'loss': torch.Tensor([0])}
            output_dict['loss'] = output_dict['loss'].to(j_score)
            # Calculate and set A/B loss
            for turn, method in enumerate(debate_mode[0]):
                if method in ['a', 'b']:
                    if sl_debate:
                        sl_loss = (-torch.log(sent_choice_probs[turn])).mean()  # Upweight prob. of Oracle choice
                        output_dict['loss'] += sl_loss
                        self._update_trainer_metrics('sl_loss' + turn_str[turn], sl_loss)
                    else:
                        reward = j_score if a_turn[turn] else (1. - j_score)
                        j_score_pred = values[turn].to(j_score)
                        baseline = j_score_pred if a_turn[turn] else (1. - j_score_pred)
                        policy_loss = -(torch.log(sent_choice_probs[turn]) * (reward - baseline.detach())).mean()
                        output_dict['loss'] += policy_loss
                        value_loss = 0.5 * ((j_score.detach() - baseline) ** 2).mean()  # Value loss
                        output_dict['loss'] += value_loss
                        self._update_trainer_metrics('policy_loss' + turn_str[turn], policy_loss)
                        self._update_trainer_metrics('value' + turn_str[turn], baseline.mean())
                        self._update_trainer_metrics('value_loss' + turn_str[turn], value_loss)  # Upper bound ~= .125
            if len(values) == 2:
                self._update_trainer_metrics('abs_diff_in_turn_value', (values[1] - values[0]).abs().mean())

        return output_dict

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()
        if (not self.model.update_judge) and (self.model.judge is not None):
            self.model.judge.eval()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data,
                                            num_epochs=1,
                                            shuffle=self.shuffle)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data)/num_gpus)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        cumulative_batch_size = 0
        for batch_group in train_generator_tqdm:
            batches_this_epoch += 1  # NOTE: Divide by self._accumulation_steps?
            self._batch_num_total += 1  # NOTE: Divide by self._accumulation_steps?
            batch_num_total = self._batch_num_total

            if (batch_num_total - 1) % self._accumulation_steps == 0:
                self.optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss = loss / self._accumulation_steps
            loss.backward()

            train_loss += loss.item()

            if batch_num_total % self._accumulation_steps != 0:
                continue  # Don't step with optimizer: accumulate gradients

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters()}
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7))
            else:
                self.optimizer.step()

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, self._trainer_metrics)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:  # NOTE: Could be inaccurate due to gradient accumulation, but not used now
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size/batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
                )
        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, self._trainer_metrics, reset=True)
        metrics['cpu_memory_MB'] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory
        return metrics

    def _validation_loss(self, debate_mode: List[str] = None) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")
        if debate_mode is None:
            debate_mode = self._debate_mode

        self.model.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)

        raw_val_generator = val_iterator(self._validation_data,
                                         num_epochs=1,
                                         shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(val_iterator.get_num_batches(self._validation_data)/num_gpus)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:

            loss = self.batch_loss(batch_group, for_training=False, debate_mode=debate_mode)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch, self._trainer_metrics)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        for epoch in range(epoch_counter, self._num_epochs + self._eval_mode):
            if not self._eval_mode:
                epoch_start_time = time.time()
                train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                                    train_metrics['cpu_memory_MB'])
            for key, value in train_metrics.items():
                if key.startswith('gpu_'):
                    metrics["peak_"+key] = max(metrics.get("peak_"+key, 0), value)

            if self._validation_data is not None:
                with torch.no_grad():
                    # pretrain_task_val_loss, pretrain_task_num_batches = self._validation_loss(debate_mode=['rr'])
                    # pretrain_task_val_metrics = training_util.get_metrics(pretrain_task_val_loss, pretrain_task_num_batches, self._trainer_metrics, reset=True)
                    # NOTE: Can add a "pretrain_task" metric (instead of train or valid). However, this would slow training.

                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, self._trainer_metrics, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            self._tensorboard.log_metrics(train_metrics, val_metrics=val_metrics, log_to_console=True)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

            if self._eval_mode:
                return metrics

            if self._serialization_dir:
                # Save id_to_oracle mapping if it's newly computed
                if (not self._id_to_oracle_is_complete) and (
                        ((self.model.reward_method is not None) and self.model.reward_method.startswith('sl')) or
                        ('A' in self._debate_mode) or
                        ('B' in self._debate_mode)):
                    self._id_to_oracle_is_complete = True
                    dump_metrics(os.path.join(self._serialization_dir, f'id_to_oracle.json'), self._id_to_oracle, log=False)
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            if self._learning_rate_scheduler:
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # These are the training states we need to persist.
        training_states = {
                "metric_tracker": self._metric_tracker.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "batch_num_total": self._batch_num_total
        }

        # If we have a learning rate scheduler, we should persist that too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = (
                    self._learning_rate_scheduler.lr_scheduler.state_dict()
            )

        self._checkpointer.save_checkpoint(
                model_state=self.model.state_dict(),
                epoch=epoch,
                training_states=training_states,
                is_best_so_far=self._metric_tracker.is_best_so_far())

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.lr_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metrics_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metrics_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    model: Model,
                    serialization_dir: str,
                    debate_mode: List[str],
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None,
                    eval_mode: bool = False,
                    breakpoint_level: int = 0,
                    id_to_oracle_filename: str = None,
                    accumulation_steps: int = 1) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   debate_mode,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=scheduler,
                   num_serialized_models_to_keep=num_serialized_models_to_keep,
                   keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period,
                   eval_mode=eval_mode,
                   breakpoint_level=breakpoint_level,
                   id_to_oracle_filename=id_to_oracle_filename,
                   accumulation_steps=accumulation_steps)


class TrainerPieces(NamedTuple):
    """
    We would like to avoid having complex instantiation logic taking place
    in `Trainer.from_params`. This helper class has a `from_params` that
    instantiates a model, loads train (and possibly validation and test) datasets,
    constructs a Vocabulary, creates data iterators, and handles a little bit
    of bookkeeping. If you're creating your own alternative training regime
    you might be able to use this.
    """
    model: Model
    iterator: DataIterator
    train_dataset: Iterable[Instance]
    validation_dataset: Iterable[Instance]
    test_dataset: Iterable[Instance]
    validation_iterator: DataIterator
    params: Params

    @staticmethod
    def from_params(params: Params, serialization_dir: str, cuda_device: int, recover: bool = False,
                    judge_filename: str = None,
                    update_judge: bool = False,
                    eval_mode: bool = False,
                    reward_method: str = None,
                    detach_value_head: bool = False) -> 'TrainerPieces':
        all_datasets = training_util.datasets_from_params(params)
        if eval_mode:  # NB: --eval_mode does not expand vocab based on test data
            params["datasets_for_vocab_creation"] = ['train', 'validation']
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

        logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                    ", ".join(datasets_for_vocab_creation))

        if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
        else:
            vocab = Vocabulary.from_params(
                    params.pop("vocabulary", {}),
                    (instance for key, dataset in all_datasets.items()
                     for instance in dataset
                     if key in datasets_for_vocab_creation)
            )

        # Debate: Load judge model from archive (if applicable)
        judge = None
        update_judge = update_judge and (judge_filename is not None)
        if judge_filename is not None:
            judge_file_ending = judge_filename.split('.')[-1]
            if judge_file_ending == 'gz':
                archive = load_archive(judge_filename, cuda_device=cuda_device)
                config = archive.config
                prepare_environment(config)
                judge = archive.model
            elif judge_file_ending == 'json' or judge_file_ending == 'jsonnet':
                # NB: No overrides for judge. Also, only 'model' field is used.
                judge_params = Params.from_file(judge_filename, params_overrides='')
                judge = Model.from_params(vocab=vocab, params=judge_params.get('model'),
                                          judge=None, update_judge=False, reward_method=None,  # No judge inside this model
                                          detach_value_head=False)
                if not update_judge:
                    warnings.warn('Provided Judge file was a training config file. '
                                  'Training from scratch even though -u was not specified.', UserWarning)
                    update_judge = True

            # Whether to use judge only for black-box reward (no gradient signal)
            if not update_judge:
                judge.eval()
            for parameter in judge.parameters():
                parameter.requires_grad_(update_judge)

        model = Model.from_params(vocab=vocab, params=params.pop('model'),
                                  judge=judge, update_judge=update_judge, reward_method=reward_method,
                                  detach_value_head=detach_value_head)

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(model.vocab)
        validation_iterator_params = params.pop("validation_iterator", None)
        if validation_iterator_params:
            validation_iterator = DataIterator.from_params(validation_iterator_params)
            validation_iterator.index_with(model.vocab)
        else:
            validation_iterator = None

        train_data = all_datasets['train']
        validation_data = all_datasets.get('validation')
        test_data = all_datasets.get('test')
        if eval_mode and (test_data is not None):
            validation_data = test_data

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        return TrainerPieces(model, iterator,
                             train_data, validation_data, test_data,
                             validation_iterator, trainer_params)
