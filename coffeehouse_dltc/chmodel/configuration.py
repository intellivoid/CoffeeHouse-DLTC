import os
import json
import shutil
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
            with open(self.classifications[classification_name], 'r', encoding="utf8") as f:
                for i, l in enumerate(f):
                    pass
            return i + 1
        else:
            raise ValueError("The classification label '{0}' is not defined in the configuration".format(classification_name))

    def classifier_contents(self, classification_name):
        if classification_name in self.classifications:
            with open(self.classifications[classification_name], 'r', encoding="utf8") as f:
                return f.read().splitlines()
        else:
            raise ValueError("The classification label '{0}' is not defined in the configuration".format(classification_name))

    def create_structure(self):
        temporary_path = "{0}_tmp".format(self.src)
        if path.exists(temporary_path):
            shutil.rmtree(temporary_path)

        data_path = path.join(temporary_path, "_data")
        os.mkdir(temporary_path)
        os.mkdir(data_path)

        classifier_labels = []
        for classifier_name, classifier_data_file in self.classifications:
            classifier_labels.append(classifier_name)
            contents = self.classifier_contents(classifier_name)

            current_value = 0
            for value in contents:
                content_file_path = "{0}_{1}.txt".format(classifier_name, current_value)
                label_file_path = "{0}_{1}.lab".format(classifier_name, current_value)
                with open(path.join(data_path, content_file_path), "w+") as content_file:
                    content_file.write(value)
                    content_file.close()
                with open(path.join(data_path, label_file_path), "w+") as label_file:
                    label_file.write(classifier_name)
                    label_file.close()
                current_value += 1