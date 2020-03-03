from .main import DLTC

def _real_main(argv=None):
    print(argv)
    print("Running from main")

def main(argv=None):
    try:
        _real_main(argv)
    except KeyboardInterrupt:
        print('\nERROR: Interrupted by user')