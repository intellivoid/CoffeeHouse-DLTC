#!/usr/bin/env python
from __future__ import unicode_literals
from coffeehouse_dltc.chmodel.configuration import Configuration
from coffeehouse_dltc.main import DLTC
import sys
import os


def _real_main(argv=None):
    """
    The main command-line processor

    :param argv:
    :return:
    """
    if argv[1] == '--help':
        _help_menu(argv)
    if argv[1] == '--model-info':
        _model_info(argv)
    if argv[1] == '--train-model':
        _train_model(argv)
    if argv[1] == '--test-model':
        _test_model(argv)


def _help_menu(argv=None):
    """
    Displays the help menu and commandline usage

    :param argv:
    :return:
    """
    print(
        "CoffeeHouse DLTC CLI\n\n"
        "   --model-info <directory_structure_input>\n"
        "   --train-model <directory_structure_input>\n"
        "   --test-model <model_directory>\n"
    )
    sys.exit()


def _test_model(argv=None):
    """
    Tests the model's prediction by allowing user input and displaying the
    prediction output

    :param argv:
    :return:
    """
    directory_model_input = os.path.join(os.getcwd(), argv[2])

    if not os.path.exists(directory_model_input):
        print("\nERROR: The directory '{0}' does not exist".format(directory_model_input))
        sys.exit()

    print("Loading model")
    dltc = DLTC()
    dltc.load_model_cluster(directory_model_input)
    print("Ready\n")

    while True:
        input_text = input("> ")
        print(dltc.predict_from_text(input_text))


def _train_model(argv=None):
    """
    Trains the model from the source directory

    :param argv:
    :return:
    """
    directory_structure_input = os.path.join(os.getcwd(), argv[2])

    if not os.path.exists(directory_structure_input):
        print("\nERROR: The directory '{0}' does not exist".format(directory_structure_input))
        sys.exit()

    configuration = Configuration(directory_structure_input)
    _model_info(argv)

    print("\n\n----- Model Training Started -----\n")
    configuration.train_model()


def _model_info(argv=None):
    """
    Displays information about the model and the training configurations

    :param argv:
    :return:
    """
    directory_structure_input = os.path.join(os.getcwd(), argv[2])

    if not os.path.exists(directory_structure_input):
        print("\nERROR: The directory '{0}' does not exist".format(directory_structure_input))
        sys.exit()

    configuration = Configuration(directory_structure_input)
    print(
        "\n--- Model Configuration Information ---\n\n"
        "   Name            : {0}\n"
        "   Author          : {1}\n"
        "   Version         : {2}\n"
        "   Description     : {3}\n"
        "---------------------------------------\n"
        "   EPOCH           : {4}\n"
        "   VEC_DIM         : {5}\n"
        "   TEST_RATIO      : {6}\n"
        "   ARCHITECTURE    : {7}\n"
        "   BATCH_SIZE      : {8}\n"
        "\n".format(
            configuration.__name__,
            configuration.__author__,
            configuration.__version__,
            configuration.__description__,
            configuration.configuration['training_properties']['epoch'],
            configuration.configuration['training_properties']['vec_dim'],
            configuration.configuration['training_properties']['test_ratio'],
            configuration.configuration['training_properties']['architecture'],
            configuration.configuration['training_properties']['batch_size']
        )
    )


if __name__ == '__main__':
    try:
        _real_main(sys.argv)
    except KeyboardInterrupt:
        print('\nERROR: Interrupted by user')
