"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp train --help

   usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-f] [-o OVERRIDES]
                         [--file-friendly-logging]
                         [--include-package INCLUDE_PACKAGE]
                         param_path

   Train the specified model on the specified dataset.

   positional arguments:
     param_path            path to parameter file describing the model to be
                           trained

   optional arguments:
     -h, --help            show this help message and exit
     -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the model and its logs
     -r, --recover         recover training from the state in serialization_dir
     -f, --force           overwrite the output directory if it exists
     -o OVERRIDES, --overrides OVERRIDES
                           a JSON structure used to override the experiment
                           configuration
     --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
     --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import List
import argparse
import logging
import os
import warnings

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, dump_metrics
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer, TrainerPieces
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.util import create_serialization_dir, evaluate

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help='Train a model')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the model and its logs')

        subparser.add_argument('-r', '--recover',
                               action='store_true',
                               default=False,
                               help='recover training from the state in serialization_dir')

        subparser.add_argument('-f', '--force',
                               action='store_true',
                               required=False,
                               help='overwrite the output directory if it exists')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        # Debate options below
        subparser.add_argument('-d', '--debate_mode',
                               default=['f'],
                               nargs='+',
                               help='how to select sentences shown to judge')

        subparser.add_argument('-j', '--judge_filename',
                               type=str,
                               default=None,
                               help='path to parameter file describing the judge to be trained or'
                                    'path to an archived, trained judge.'
                                    'Do not use this option if training judge only.'
                                    'If updating pre-trained judge, pass in archived *.tar.gz file.'
                                    'If training judge from scratch, pass in *.jsonnet config file.')

        subparser.add_argument('-u', '--update_judge',
                               action='store_true',
                               default=False,
                               help='update judge while training debate agents')

        # NB: --evaluate does not expand vocab based on test data
        subparser.add_argument('-e', '--eval_mode',
                               action='store_true',
                               default=False,
                               help='run in evaluation-only mode on test_data_path (validation if no test given)')

        subparser.add_argument('-m', '--reward_method',
                               type=str,
                               choices=['em', 'f1', 'prob',  # Exact Match, F1, (Span Start) Probability
                                        'sl'],  # Supervised Learning (Oracle Prob)
                               default='f1',
                               help='how to reward debate agents')

        subparser.add_argument('-v', '--detach_value_head',
                               action='store_true',
                               default=False,
                               help='Detach value head prediction network from main policy network,'
                                    'to prevent gradients to value function from overpowering gradients to policy')

        subparser.add_argument('-b', '--breakpoint_level',
                               type=int,
                               default=0,
                               help='Debugging option: Increase to run with more breakpoints. 0 for no breakpoints.')

        subparser.add_argument('-p', '--oracle_outputs_path',
                               type=str,
                               default=None,
                               help='Name file containing oracle predictions to do Supervised Learning on.')

        subparser.add_argument('-a', '--accumulation_steps',
                               type=int,
                               default=1,
                               help='Number of steps to accumulate gradient for before taking an optimizer step.')

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path,
                          args.serialization_dir,
                          args.overrides,
                          args.file_friendly_logging,
                          args.recover,
                          args.force,
                          args.debate_mode,
                          args.judge_filename,
                          args.update_judge,
                          args.eval_mode,
                          args.reward_method,
                          args.detach_value_head,
                          args.breakpoint_level,
                          args.oracle_outputs_path,
                          args.accumulation_steps)


def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False,
                          force: bool = False,
                          debate_mode: List[str] = ('f'),
                          judge_filename: str = None,
                          update_judge: bool = False,
                          eval_mode: bool = False,
                          reward_method: str = None,
                          detach_value_head: bool = False,
                          breakpoint_level: int = 0,
                          oracle_outputs_path: str = None,
                          accumulation_steps: int = 1) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    parameter_filename : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, file_friendly_logging, recover, force, debate_mode, judge_filename,
                       update_judge, eval_mode, reward_method, detach_value_head, breakpoint_level,
                       oracle_outputs_path, accumulation_steps)


def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False,
                force: bool = False,
                debate_mode: List[str] = ('f'),
                judge_filename: str = None,
                update_judge: bool = False,
                eval_mode: bool = False,
                reward_method: str = None,
                detach_value_head: bool = False,
                breakpoint_level: int = 0,
                oracle_outputs_path: str = None,
                accumulation_steps: int = 1) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    num_trained_debater_turns = sum([(('a' in debate_turn) or ('b' in debate_turn)) for debate_turn in debate_mode])
    if ((judge_filename is not None) and (num_trained_debater_turns == 0)):
        warnings.warn('Unnecessary to have debaters in debate mode ' + str(debate_mode) +
                      '. If this was unintentional, please remove the -j flag.', UserWarning)

    prepare_environment(params)
    create_serialization_dir(params, serialization_dir, recover, force)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    trainer_type = params.get("trainer", {}).get("type", "default")

    if trainer_type == "default":
        # Special logic to instantiate backward-compatible trainer.
        pieces = TrainerPieces.from_params(params, serialization_dir, cuda_device, recover,
                                           judge_filename=judge_filename,
                                           update_judge=update_judge,
                                           eval_mode=eval_mode,
                                           reward_method=reward_method,
                                           detach_value_head=detach_value_head)  # pylint: disable=no-member
        trainer = Trainer.from_params(
                model=pieces.model,
                serialization_dir=serialization_dir,
                debate_mode=debate_mode,
                iterator=pieces.iterator,
                train_data=pieces.train_dataset,
                validation_data=pieces.validation_dataset,
                params=pieces.params,
                validation_iterator=pieces.validation_iterator,
                eval_mode=eval_mode,
                breakpoint_level=breakpoint_level,
                oracle_outputs_path=oracle_outputs_path,
                accumulation_steps=accumulation_steps)
        evaluation_iterator = pieces.validation_iterator or pieces.iterator
        evaluation_dataset = pieces.test_dataset
        # TODO: Check you're not modifying variables important for later on, in TrainerPieces
    else:
        assert (len(debate_mode) == 1) and (debate_mode[0] == 'f'), 'TrainerBase untested for debate training.'
        trainer = TrainerBase.from_params(params, serialization_dir, recover)
        # TODO(joelgrus): handle evaluation in the general case
        evaluation_iterator = evaluation_dataset = None

    params.assert_empty('base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)) and not eval_mode:
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Evaluate
    if evaluation_dataset and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(trainer.model, evaluation_dataset, evaluation_iterator,
                                cuda_device=trainer._cuda_devices[0], # pylint: disable=protected-access,
                                # TODO(brendanr): Pass in an arg following Joel's trainer refactor.
                                batch_weight_key="")

        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif evaluation_dataset:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    # Now tar up results
    if not eval_mode:
        archive_model(serialization_dir, files_to_archive=params.files_to_archive)
    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    # We count on the trainer to have the model with best weights
    return trainer.model
