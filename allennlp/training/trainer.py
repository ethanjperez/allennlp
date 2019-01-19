import copy
import logging
import math
import os
import time
import re
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, NamedTuple

import torch
import torch.optim.lr_scheduler
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import (dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary
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
                 breakpoint_level: int = 0) -> None:
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

    def _data_parallel(self, batch, model):
        """
        Do the forward pass using multiple GPUs.  This is a simplification
        of torch.nn.parallel.data_parallel to support the allennlp model
        interface.
        """
        inputs, module_kwargs = scatter_kwargs((), batch, self._cuda_devices, 0)
        used_device_ids = self._cuda_devices[:len(inputs)]
        replicas = replicate(model, used_device_ids)
        outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)

        # Only the 'loss' is needed.
        # a (num_gpu, ) tensor with loss on each GPU
        losses = gather([output['loss'].unsqueeze(0) for output in outputs], used_device_ids[0], 0)
        return {'loss': losses.mean()}

    def _forward(self, batch_group, model):
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

    def _create_batch_from_sample(self, batch, sample_no, bsz):
        """
        Slices and copies an existing batch into a smaller batch. Repeats
        """
        sliced_batch = {}
        idxs = sample_no  # Can replace with slice
        for k, v in batch.items():
            if isinstance(v, dict):
                sliced_batch[k] = {}
                for inner_k, inner_v in v.items():
                    sliced_batch[k][inner_k] = inner_v[idxs].repeat(bsz, *[1 for _ in range(inner_v[idxs].dim())])
            elif isinstance(v, torch.Tensor):
                sliced_batch[k] = v[idxs].repeat(bsz, 1)
            elif isinstance(v, list):
                sliced_batch[k] = [copy.deepcopy(v[idxs]) for _ in range(bsz)]
            else:
                raise NotImplementedError('Unimplemented slice for key, value:', k, v)
        return sliced_batch

    def _update_trainer_metrics(self, metric_name, new_value):
        """
        Updates the trainer's metrics for metric_name with new_value
        """
        self._trainer_metrics[metric_name] = self._trainer_metrics.get(metric_name, Average())
        self._trainer_metrics[metric_name](new_value)

    def _modify_passage(self, batch, sent_choice_masks, pad_masks, mask_tok_val, mod_type):
        """
        Modifies a passage according to sentence selections made elsewhere.
        Supports e.g. masking and deleting portions of the passage.
        Used before passing the batch to the judge.
        """
        mod_type = mod_type.lower()
        if mod_type == 'delete':
            # NB: SQuAD: Can also make deletion-based. Must modify span_end, span_start, and metadata then.
            # NB: RACE: Better to modify metadata here
            post_delete_toks = torch.zeros_like(batch['passage']['tokens'])
            post_delete_tok_chars = torch.zeros_like(batch['passage']['token_characters'])
            for idx in range(batch['passage']['tokens'].size(0)):
                toks = batch['passage']['tokens'][idx]
                tok_chars = batch['passage']['token_characters'][idx]
                reveal_idxs = (toks * (1. - sent_choice_masks[idx])).nonzero().squeeze()
                post_delete_toks[idx][:toks[reveal_idxs].size(0)] = toks[reveal_idxs]
                post_delete_tok_chars[idx][:toks[reveal_idxs].size(0)] = tok_chars[reveal_idxs]
            # Detaching just in case to prevent gradient flow back to agents modifying passage
            batch['passage']['tokens'] = post_delete_toks.detach()
            batch['passage']['token_characters'] = post_delete_tok_chars.detach()
        elif mod_type == 'mask':
            batch['passage']['tokens'] = ((batch['passage']['tokens'] * sent_choice_masks) + (
                    (1 - sent_choice_masks) * mask_tok_val)
            ) * pad_masks
            batch['passage']['token_characters'] = (
               (batch['passage']['token_characters'] * sent_choice_masks.unsqueeze(-1)) + ((
                1 - sent_choice_masks.unsqueeze(-1)) * mask_tok_val)
            ) * pad_masks.unsqueeze(-1)
        else:
            raise NotImplementedError('Modifying passages via mod_type ' + mod_type + ' not supported.')
        return batch

    def _add_debate_metrics(self, output_dict, sent_idxs, sent_choice_idxs, num_turns, turn_str):
        """
        Add various metrics related to the batch's debate (excluding losses)
        """
        # Add stats on if J chosen a sentence from A or B
        # NB: Not useful for RACE (except to check if there's a bug), may be incorrect for SQuAD deletion setting
        j_span_start_sent = sent_idxs.gather(1, output_dict['best_span'][:, :1].to(sent_idxs.device))
        j_span_end_sent = sent_idxs.gather(1, output_dict['best_span'][:, 1:].to(sent_idxs.device))
        j_num_ab_sents_chosen = torch.zeros_like(j_span_start_sent).float()
        for turn in range(num_turns):
            j_sent_chosen = ((j_span_start_sent <= sent_choice_idxs[turn]) * (sent_choice_idxs[turn] <= j_span_end_sent)).float()
            self._update_trainer_metrics('j_sent_chosen' + turn_str[turn], j_sent_chosen.mean())
            j_num_ab_sents_chosen += j_sent_chosen
        j_chose_no_ab_sents = (j_num_ab_sents_chosen == 0).float()
        self._update_trainer_metrics('j_sent_chosen_not_a_or_b', j_chose_no_ab_sents.mean())

    @staticmethod
    def _print_debate(batch, num_sents, debate_mode, sent_choice_masks, sent_choice_idxs, j_em, j_f1, output_dict):
        """
        Neatly prints all debates from a batch.
        """
        bsz = batch['question']['tokens'].size(0)
        for sample_no in range(bsz):
            if bool(num_sents[sample_no] >= 3):
                print('\n***Passage***\n', ' '.join(batch['metadata'][sample_no]['passage_tokens']))
                print('\n***Question***\n', ' '.join(batch['metadata'][sample_no]['question_tokens']))
                print('\n***Answers***\n', [answer if isinstance(answer, str) else ' '.join(answer) for answer in batch['metadata'][sample_no]['answer_texts']])
                toks = batch['metadata'][sample_no]['passage_tokens']
                for turn, method in enumerate(debate_mode[0]):
                    turn_sent_idxs = sent_choice_masks[turn][sample_no].nonzero().squeeze()
                    turn_sent_start_idx = turn_sent_idxs.min()
                    turn_sent_end_idx = turn_sent_idxs.max() + 1
                    print('\n---', method.upper(), '--- Sentence', int(sent_choice_idxs[turn][sample_no]), '\n', ' '.join(toks[turn_sent_start_idx:turn_sent_end_idx]))
                print('\n--- J --- EM / F1 ', float(j_em[sample_no]), '/', float(j_f1[sample_no]), '!\n', ' '.join(toks[output_dict['best_span'][sample_no][0]:output_dict['best_span'][sample_no][1] + 1]))
        return

    # TODO: batch Tensor -> batch_group List[TensorDict]
    def _batch_loss(self, batch_group: List[TensorDict], for_training: bool, debate_mode: List[str] = None) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        # If overriding default passage choosing method
        if debate_mode is None:
            debate_mode = self._debate_mode

        # Set output_dict['loss'] to do gradient descent on.
        if debate_mode[0] == "f":  # Full passage training: Normal SL training
            self._forward(batch_group, self.model)
        else:  # Training on subset of sentence (judge or debate training)
            # Set a few useful variables/aliases
            debater = None if self.model.is_judge else self.model
            judge = self.model if self.model.is_judge else self.model.judge
            race_data = ('SELECT_' in batch_group['metadata'][0]['answer_texts'][0])  # NB: Fix this to be cleaner!
            mod_type = 'delete' if race_data else 'mask'
            num_rounds = len(debate_mode)
            assert num_rounds <= 1, 'No implementation yet for # rounds =' + str(num_rounds)
            num_turns = len(debate_mode[0])
            bsz = batch_group['question']['tokens'].size(0)
            mask_tok_val = self.model.vocab.get_token_index('.')
            a_turn = {turn: debate_mode[0][turn] == 'a' for turn in range(len(debate_mode[0]))}
            turn_str = {turn: "_turn_" + str(turn) + "_agent_" + debate_mode[0][turn] for turn in range(num_turns)}
            sl_debate = (debater is not None) and (debater.reward_method.startswith('sl'))
            debate_mode_with_eval_only_turns = debate_mode[0] if not sl_debate else debate_mode[0].replace('b', 'Bb').replace('a', 'Aa')

            # Precomputation. NB: Move from CPU to GPU if slow
            tok_mask = {'.': (batch_group['passage']['tokens'] == self.model.vocab.get_token_index('.')).long()}
            eos_tok_mask = tok_mask['.']  # TODO: Add '?' and '!'
            # If last non-padding token isn't a period, make it also an eos token in the mask
            if not race_data:  # NB: Remove 'if' later. SQuAD models trained without this clause (slight inaccuracy possibly)
                for i in range(bsz):
                    eos_tok_mask[i, batch_group['passage']['tokens'][i].nonzero()[-1]] = 1

            if race_data:  # Each answer choice counts as 1 sentence no matter what
                ans_toks = ['select_a', 'select_b', 'select_c', 'select_d']
                for tok_str in ans_toks:
                    tok_val = self.model.vocab.get_token_index(tok_str)
                    tok_mask[tok_str] = (batch_group['passage']['tokens'] == tok_val).long()

                # Replace original sent_idxs vector for that portion with sliced vector
                not_ans_mask = tok_mask['select_d'].cumsum(1) - tok_mask['select_d']
                eos_tok_mask *= not_ans_mask
                for ans_tok in ans_toks:
                    eos_tok_mask += tok_mask[ans_tok]
            sent_idxs = eos_tok_mask.cumsum(1) - eos_tok_mask  # NOTE: Padding regions have sent_idxs == num_sents
            pad_masks = (batch_group['passage']['tokens'] != 0).long()
            num_sents = (sent_idxs * pad_masks).max(1)[0] + 1
            sent_answer_idx = sent_idxs.gather(1, batch_group['span_start'].to(sent_idxs.device))

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
                num_first_sents_excluded = len(ans_toks) if race_data else 0
                # Variables that must be set after each turn
                sent_choice_idx = None
                sent_choice_prob = None
                value = None
                if method == 'r':  # Random selection
                    sent_choice_idx = (torch.rand_like(num_sents.float()) * (num_sents.float() - num_first_sents_excluded)).trunc().long().unsqueeze(1) + num_first_sents_excluded
                    sent_choice_prob = torch.ones(bsz) / (num_sents.float() - num_first_sents_excluded)
                    value = -1 * torch.ones(bsz)
                elif method == 'g':  # Ground truth, answer-containing selection
                    assert not race_data, 'RACE does not provide "Ground Truth" answer-supporting sentences'
                    sent_choice_idx = sent_answer_idx
                    sent_choice_prob = torch.ones(bsz)
                    value = -1 * torch.ones(bsz)
                elif method in ['A', 'B']:  # A/B oracle selection
                    oracle_func = max if method == 'A' else min  # NOTE: Modify if adding another oracle method
                    # NB: RACE oracle should preferably use span_start_probs
                    oracle_eval_method = 'em' if race_data else 'f1'  # NOTE: Only other option is 'em'
                    if sl_debate and debater.reward_method == 'sl-ssp':
                        oracle_eval_method = 'ssp'
                    # NOTE: Set below to None to make oracle selection simultaneous with other selections
                    past_sent_choice_idxs = torch.cat(sent_choice_idxs, 1) if len(sent_choice_idxs) > 0 else None
                    opt_sent_idxs = []
                    if sl_debate:
                        sc_diffs = []
                    oracle_values = []
                    judge_was_training = judge.training
                    judge.eval()
                    for sample_no in range(bsz):
                        # Batch together all possible next outcomes for a sample
                        # RACE: Removes Oracle choices selecting answer sentences (which'll always be shown)
                        num_sent_options = num_sents[sample_no] - num_first_sents_excluded
                        oracle_batch = self._create_batch_from_sample(batch_group, sample_no, num_sent_options)
                        oracle_batch['store_metrics'] = False  # Do not update judge metrics
                        oracle_sent_choice_idxs = torch.arange(num_sent_options).unsqueeze(1) + num_first_sents_excluded
                        if past_sent_choice_idxs is not None:
                            past_idxs_repeat = past_sent_choice_idxs[sample_no].repeat(num_sent_options, 1)
                            oracle_sent_choice_idxs = torch.cat([past_idxs_repeat, oracle_sent_choice_idxs], 1)

                        # Modify passage
                        oracle_pad_masks = pad_masks[sample_no].unsqueeze(0)
                        oracle_sent_choice_masks = torch.stack([
                            sent_idxs[sample_no].unsqueeze(0).expand(num_sent_options, -1) ==
                            oracle_sent_choice_idxs[:,i].unsqueeze(1) for i in range(turn + 1)]).sum(0)
                        oracle_sent_choice_masks = oracle_sent_choice_masks / (oracle_sent_choice_masks.clamp(min=1))  # Differentiable clamp to max=1
                        oracle_batch = self._modify_passage(
                            oracle_batch, oracle_sent_choice_masks, oracle_pad_masks, mask_tok_val, mod_type)

                        # Get judge results
                        oracle_output_dict, oracle_metrics = self._forward(oracle_batch, judge)
                        oracle_metrics = oracle_metrics.get_metric(reset=True, per_sample=True)[1 if oracle_eval_method == 'f1' else 0]
                        if oracle_eval_method == 'ssp':
                            oracle_metrics = [oracle_output_dict['span_start_probs'][i, oracle_batch['span_start'][i]].item() for i in range(oracle_output_dict['span_start_probs'].size(0))]
                        opt_sc = float(oracle_func(oracle_metrics))
                        oracle_values.append(opt_sc)
                        opt_sent_idxs.append(oracle_metrics.index(opt_sc) + num_first_sents_excluded)
                        if sl_debate:
                            baseline_sc = sum(oracle_metrics) / len(oracle_metrics)
                            # NB: Hard-coding different baseline score based on debate_mode
                            if debate_mode == 'gb':  # No sentence choice is baseline
                                baseline_sc = oracle_metrics[past_sent_choice_idxs[sample_no, 0]]
                            sc_diffs.append(baseline_sc - opt_sc)
                    if judge_was_training:
                        judge.train()

                    sent_choice_idx = torch.LongTensor(opt_sent_idxs).unsqueeze(1)
                    sent_choice_prob = torch.ones(bsz)
                    value = torch.FloatTensor(oracle_values)
                elif method in ['a', 'b']:  # A/B trained selection
                    assert debater is not None, 'Cannot use debate method ' + method + ' without debate agents!'
                    for batch_idx in range(bsz):  # NB: 'metadata' usually optional but will now cause error if missing
                        batch_group['metadata'][batch_idx]['a_turn'] = a_turn[turn]
                    ab_output_dict = self._forward(batch_group, debater)
                    debater.get_metrics(reset=True)  # A/B metrics currently meaningless, so clear

                    # Sample from policy's sentence-level distribution
                    word_choice_dist = ab_output_dict['span_start_probs']
                    # TODO: Do argmax on sentence-level distribution!
                    word_choice_idx = torch.multinomial(word_choice_dist, 1) if for_training else torch.argmax(word_choice_dist, dim=1, keepdim=True)
                    sent_choice_idx = sent_idxs.gather(1, word_choice_idx.to(sent_idxs.device))

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
                        self._update_trainer_metrics('sl_non_default_preds_per_batch' + turn_str[turn], (sent_choice_idx != num_first_sents_excluded).float().sum())
                        oracle_sent_choice_mask = (sent_idxs == oracle_sent_choice_idx)
                        sent_choice_prob = (word_choice_dist.to(oracle_sent_choice_mask.device) * oracle_sent_choice_mask.to(word_choice_dist.dtype)).sum(1)
                    else:  # RL: Use prob of sampled sentence to calculate loss
                        ab_sent_choice_mask = (sent_idxs == sent_choice_idx)
                        sent_choice_prob = (word_choice_dist.to(ab_sent_choice_mask.device) * ab_sent_choice_mask.to(word_choice_dist.dtype)).sum(1)

                    value = ab_output_dict['value']
                else:
                    raise NotImplementedError('Unimplemented answer selection debate method', method)

                # Map invalid sentence choices to no-ops / no sentence selection
                sent_choice_idx[sent_choice_idx < num_first_sents_excluded] = -1

                # Apply masks / use probs only after eval-only turns are finished
                if is_eval_only_turn and sl_debate:
                    oracle_sent_choice_idx = sent_choice_idx
                    continue

                assert (sent_choice_idx is not None) and (sent_choice_prob is not None) and (value is not None), \
                    'Error: Did not fill all necessary variables for turn selection.'
                sent_choice_idxs.append(sent_choice_idx)
                sent_choice_probs.append(sent_choice_prob)
                values.append(value.cpu())

                answer_sent_chosen = (sent_choice_idx == sent_answer_idx).float()  # NOTE: Assumes answer does not cross period boundary
                self._update_trainer_metrics('answer_sent_chosen' + turn_str[turn], answer_sent_chosen.mean())

            # Remove metadata added for A/B forward pass
            for batch_idx in range(bsz):
                if 'a_turn' in batch_group['metadata'][batch_idx]:
                    batch_group['metadata'][batch_idx].pop('a_turn')

            # Mask passage and pass to Judge
            sent_choice_masks = [(sent_idxs == sent_choice_idx) for sent_choice_idx in sent_choice_idxs]
            all_sent_choice_mask = torch.stack(sent_choice_masks).sum(0)
            all_sent_choice_mask = all_sent_choice_mask / (all_sent_choice_mask.clamp(min=1))  # Differentiable clamp to max=1
            batch_group = self._modify_passage(batch_group, all_sent_choice_mask, pad_masks, mask_tok_val, mod_type)
            output_dict = self._forward(batch_group, judge)

            # Debate metrics and losses
            if debater is not None:
                # Debate metrics
                j_metrics = judge.get_metrics(per_sample=True)
                j_em = torch.tensor(j_metrics['em'], dtype=sent_choice_probs[0].dtype, device=sent_choice_probs[0].device)
                j_f1 = torch.tensor(j_metrics['f1'], dtype=sent_choice_probs[0].dtype, device=sent_choice_probs[0].device)
                if debater.reward_method == 'f1':
                    j_score = j_f1
                elif debater.reward_method == 'ssp':
                    j_score = torch.Tensor([output_dict['span_start_probs'][i, batch_group['span_start'][i]] for i in range(bsz)])
                else:  # EM or SL (where EM is a dummy value)
                    j_score = j_em

                self._add_debate_metrics(output_dict, sent_idxs, sent_choice_idxs, num_turns, turn_str)
                if self._eval_mode and (((self._batch_num_total % 20) == 0) or ((self._batch_num_total % 20) == 1)):
                    print_debate(self, batch_group, num_sents, debate_mode, sent_choice_masks, sent_choice_idxs, j_em, j_f1, output_dict)

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
                            # Use 1-R_a
                            grad_dir = -1 if a_turn[turn] else 1
                            baseline = values[turn].to(j_score)
                            policy_loss = grad_dir * (torch.log(sent_choice_probs[turn]) * (j_score - baseline.detach())).mean()
                            output_dict['loss'] += policy_loss
                            value_loss = 0.5 * ((j_score - baseline) ** 2).mean()  # Value loss
                            output_dict['loss'] += value_loss
                            self._update_trainer_metrics('policy_loss' + turn_str[turn], policy_loss)
                            self._update_trainer_metrics('value' + turn_str[turn], baseline.mean())
                            self._update_trainer_metrics('value_loss' + turn_str[turn], value_loss)  # Upper bound ~= .125
                if len(values) == 2:
                    self._update_trainer_metrics('abs_diff_in_turn_value', (values[1] - values[0]).abs().mean())

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

    # TODO: Move to appropriate util file
    def _get_metrics(self, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``num_batches`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        judge = self.model if self.model.is_judge else self.model.judge
        metrics = judge.get_metrics(reset=reset)
        metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
        trainer_metrics = {name: metric.get_metric(reset).item() for name, metric in self._trainer_metrics.items()}
        metrics.update(trainer_metrics)
        return metrics

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
        self._model.train()
        if (not self._model.update_judge) and (self._model.judge is not None):
            self._model.judge.eval()

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
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()

            train_loss += loss.item()

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
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch)
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

            if self._log_batch_size_period:
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
        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
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
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
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
                    # pretrain_task_val_metrics = self._get_metrics(pretrain_task_val_loss, pretrain_task_num_batches, reset=True)
                    # NOTE: Can add a "pretrain_task" metric (instead of train or valid). However, this would slow training.

                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

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
                    breakpoint_level: int = 0) -> 'Trainer':
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
                   breakpoint_level=breakpoint_level)


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
    def from_params(params: Params, serialization_dir: str, recover: bool = False) -> 'TrainerPieces':
        all_datasets = training_util.datasets_from_params(params)
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

        model = Model.from_params(vocab=vocab, params=params.pop('model'))

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
