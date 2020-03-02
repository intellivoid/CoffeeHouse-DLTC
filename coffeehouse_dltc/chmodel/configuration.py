import json
from os import path


class Configuration(object):

    def __init__(self, src_directory):
        self.src = src_directory
        if not path.exists(src_directory):
            raise FileNotFoundError("The source directory '{0}' was not found".format(src_directory))

        self.configuration_file = path.join(self.src, "model.json")
        if not path.exists(self.configuration_file):
            raise FileNotFoundError("The file 'model.json' was not found in the source directory")

        with open(self.configuration_file, 'r') as f:
            self.configuration = json.load(f)

        self.__name__ = self.configuration['model']['name']
        self.__author__ = self.configuration['model']['author']
        self.__version__ = self.configuration['model']['version']
        self.__description__ = self.configuration['model']['description']
        self.epochs = self.configuration['training_properties']['epoch']

        self.classifications = {}
        for classification_method in self.configuration['classification']:
            self.classifications[classification_method['l']] = path.join(self.src, classification_method['f'])

    def classifier_range(self, classification_name):
        if classification_name in self.classifications:
            with open(self.classifications[classification_name]) as f:
                for i, l in enumerate(f):
                    pass
            return i + 1
        else:
            raise ValueError("The classification '{0}' is not defined in the configuration".format(classification_name))