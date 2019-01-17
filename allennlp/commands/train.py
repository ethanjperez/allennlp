"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp train --help
   usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-o OVERRIDES]
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
   -o OVERRIDES, --overrides OVERRIDES
                           a JSON structure used to override the experiment
                           configuration
   --include-package INCLUDE_PACKAGE
                           additional packages to include
   --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
"""
from typing import Dict, Iterable, List
import argparse
import logging
import os
import re
import shutil
import warnings

import torch

from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, \
                                 get_frozen_and_tunable_parameter_names, dump_metrics
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model, CONFIG_NAME, load_archive
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer

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

        # Debate: Debate mode: Use A/B or random sentences (and in what order). Used for training and evaluation.
        subparser.add_argument('-d', '--debate_mode',
                               required=True,
                               nargs='+',
                               help='how to select sentences shown to judge')

        subparser.add_argument('-r', '--recover',
                               action='store_true',
                               default=False,
                               help='recover training from the state in serialization_dir')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        # Debate options below
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
        subparser.add_argument('-e', '--evaluate',
                               action='store_true',
                               default=False,
                               help='run in evaluation-only mode on test_data_path (validation if no test given)')

        subparser.add_argument('-m', '--reward_method',
                               type=str,
                               choices=['em', 'f1', 'sl', 'ssp'],  # Exact Match, F1, Superv. Learning, Span Start Prob.
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

        subparser.set_defaults(func=train_model_from_args)

        return subparser

def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path,
                          args.serialization_dir,
                          args.debate_mode,
                          args.overrides,
                          args.file_friendly_logging,
                          args.recover,
                          args.judge_filename,
                          args.update_judge,
                          args.evaluate,
                          args.reward_method,
                          args.detach_value_head,
                          args.breakpoint_level)


def train_model_from_file(parameter_filename: str,
                          serialization_dir: str,
                          debate_mode: List[str],
                          overrides: str = "",
                          file_friendly_logging: bool = False,
                          recover: bool = False,
                          judge_filename: str = None,
                          update_judge: bool = False,
                          evaluate: bool = False,
                          reward_method: str = None,
                          detach_value_head: bool = False,
                          breakpoint_level: int = 0) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path : ``str``
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
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, debate_mode, file_friendly_logging, recover,
                       judge_filename, update_judge, evaluate, reward_method, detach_value_head, breakpoint_level)


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets


def create_serialization_dir(params: Params, serialization_dir: str, recover: bool) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.

    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    """
    if os.path.exists(serialization_dir) and serialization_dir.endswith('debug'):
        shutil.rmtree(serialization_dir)  # Overwrite "debug" directory automatically (if nec.)

    if os.path.exists(serialization_dir) and os.listdir(serialization_dir) and len(os.listdir(serialization_dir)) > 0:
        if not recover:
            raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            # Check whether any of the training configuration differs from the configuration we are
            # resuming.  If so, warn the user that training may fail.
            fail = False
            flat_params = params.as_flat_dict()
            flat_loaded = loaded_params.as_flat_dict()
            no_error_check_keys = ['test_data_path']
            for key in flat_params.keys() - flat_loaded.keys():
                if key in no_error_check_keys:
                    continue
                logger.error(f"Key '{key}' found in training configuration but not in the serialization "
                             f"directory we're recovering from.")
                fail = True
            for key in flat_loaded.keys() - flat_params.keys():
                if key in no_error_check_keys:
                    continue
                logger.error(f"Key '{key}' found in the serialization directory we're recovering from "
                             f"but not in the training config.")
                fail = True
            for key in flat_params.keys():
                if key in no_error_check_keys:
                    continue
                if flat_params.get(key, None) != flat_loaded.get(key, None):
                    logger.error(f"Value for '{key}' in training configuration does not match that the value in "
                                 f"the serialization directory we're recovering from: "
                                 f"{flat_params[key]} != {flat_loaded[key]}")
                    fail = True
            # if fail:
            #     raise ConfigurationError("Training configuration does not match the configuration we're "
            #                              "recovering from.")
    else:
        # if recover:
        #     raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
        #                              "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)


def train_model(params: Params,
                serialization_dir: str,
                debate_mode: List[str],
                file_friendly_logging: bool = False,
                recover: bool = False,
                judge_filename: str = None,
                update_judge: bool = False,
                evaluate: bool = False,
                reward_method: str = None,
                detach_value_head: bool = False,
                breakpoint_level: int = 0) -> Model:
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

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    num_trained_debater_turns = sum(['a' in debate_turn or 'b' in debate_turn for debate_turn in debate_mode])
    if not ((judge_filename is not None) and (num_trained_debater_turns == 0)):
        warnings.warn('Unnecessary to have debaters in debate mode ' + str(debate_mode) +
                      '. If this was unintentional, please remove the -j flag.', UserWarning)

    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, recover)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    all_datasets = datasets_from_params(params)
    if evaluate:  # NB: --evaluate does not expand vocab based on test data
        params["datasets_for_vocab_creation"] = ['train', 'validation']
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation))
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
            # Load from archive (Modified from evaluate.py)
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
    iterator.index_with(vocab)
    validation_iterator_params = params.pop("validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')
    if evaluate and (test_data is not None):
        validation_data = test_data

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
        get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    trainer_choice = trainer_params.pop_choice("type",
                                               Trainer.list_available(),
                                               default_to_first_choice=True)
    trainer = Trainer.by_name(trainer_choice).from_params(model=model,
                                                          serialization_dir=serialization_dir,
                                                          debate_mode=debate_mode,
                                                          iterator=iterator,
                                                          train_data=train_data,
                                                          validation_data=validation_data,
                                                          params=trainer_params,
                                                          validation_iterator=validation_iterator,
                                                          evaluate=evaluate,
                                                          breakpoint_level=breakpoint_level)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    params.assert_empty('base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)) and not evaluate:
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Now tar up results
    if not evaluate:
        archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)

    # NB: Evaluate command deprecated for debate
    # if test_data and evaluate_on_test:
    #     logger.info("The model will be evaluated using the best epoch weights.")
    #     test_metrics = evaluate(
    #             best_model, test_data, validation_iterator or iterator,
    #             cuda_device=trainer._cuda_devices[0]  # pylint: disable=protected-access
    #     )
    #     for key, value in test_metrics.items():
    #         metrics["test_" + key] = value
    #
    # elif test_data:
    #     logger.info("To evaluate on the test set after training, pass the "
    #                 "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    return best_model
