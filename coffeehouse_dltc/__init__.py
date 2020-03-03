from coffeehouse_dltc.chmodel.configuration import Configuration
from .main import DLTC
import sys
import os


def _real_main(argv=None):
    if argv[1] == '--help':
        _help_menu(argv)
    if argv[1] == '--model-info':
        _model_info(argv)


def _help_menu(argv=None):
    print(
        "CoffeeHouse DLTC CLI\n\n"
        "   --model-info -i <directory_structure_input>"
        "   --train-model -i <directory_structure_input> -o <output_directory>"
        "   --test-model -i <model_directory>"
    )
    sys.exit()


def _model_info(argv=None):
    directory_structure_input = None

    for opt, arg in argv:
        if opt in "-i":
            directory_structure_input = arg

    if os.path.exists(directory_structure_input):
        print("\nERROR: The directory '{0}' does not exist".format(directory_structure_input))
        sys.exit()

    configuration = Configuration(directory_structure_input)
    print(
        "\n--- Model Configuration Information ---\n\n"
        "   Name            : {0}\n"
        "   Author          : {1}\n"
        "   Version         : {2}\n"
        "   Description     : {3}\n"
        "--------------------------------------------"
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


def main(argv=None):
    try:
        _real_main(argv)
    except KeyboardInterrupt:
        print('\nERROR: Interrupted by user')
