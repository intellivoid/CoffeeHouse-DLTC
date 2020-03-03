import os
import json
import shutil
from os import path

from coffeehouse_dltc import DLTC


class Configuration(object):

    def __init__(self, src_directory):
        """
        Public Constructor

        :param src_directory:
        """
        self.src = src_directory
        if not path.exists(src_directory):
            raise FileNotFoundError("The source directory '{0}' was not found".
                                    format(src_directory))

        self.configuration_file = path.join(self.src, "model.json")
        if not path.exists(self.configuration_file):
            raise FileNotFoundError("The file 'model.json' was not found in the source directory")

        with open(self.configuration_file, 'r') as f:
            self.configuration = json.load(f)

        self.__name__ = self.configuration['model']['name']
        self.__author__ = self.configuration['model']['author']
        self.__version__ = self.configuration['model']['version']
        self.__description__ = self.configuration['model']['description']

        self.classifications = {}
        for classification_method in self.configuration['classification']:
            self.classifications[classification_method['l']] = path.join(
                self.src, classification_method['f']
            )

    def classifier_range(self, classification_name):
        """
        Determines the range of the classifier

        :param classification_name:
        :return: Integer of the amount of data the classifier contains
        """
        if classification_name in self.classifications:
            with open(self.classifications[classification_name], 'r', encoding="utf8") as f:
                for i, l in enumerate(f):
                    pass
            return i + 1
        else:
            raise ValueError(
                "The classification label '{0}' is not defined in the configuration".format(
                    classification_name))

    def classifier_contents(self, classification_name):
        """
        Returns the contents of the classifier

        :param classification_name:
        :return: Contents of the classifier split into a list type
        """
        if classification_name in self.classifications:
            with open(self.classifications[classification_name], 'r', encoding="utf8") as f:
                return f.read().splitlines()
        else:
            raise ValueError(
                "The classification label '{0}' is not defined in the configuration".format(
                    classification_name))

    def classifier_labels(self):
        """
         Returns list of labels that this model is configured to use based on the classifier data

        :return: List of labels
        """
        classifier_labels = []
        for classifier_name, classifier_data_file in self.classifications.items():
            classifier_labels.append(classifier_name)
        return classifier_labels

    def create_structure(self):
        """
        Creates the model structure which allows training to be simplified

        :return: the path of the directory containing the model structure
        """
        print("Preparing structure directory")
        temporary_path = "{0}_data".format(self.src)
        if path.exists(temporary_path):
            shutil.rmtree(temporary_path)

        data_path = path.join(temporary_path, "model_data")
        os.mkdir(temporary_path)
        print("Created directory '{0}'".format(temporary_path))
        os.mkdir(data_path)
        print("Created directory '{0}'".format(data_path))

        labels_file_path = path.join(temporary_path, "model_data.labels")

        with open(labels_file_path, 'w+', encoding='utf8') as f:
            for item in self.classifier_labels():
                f.write("%s\n" % item)
            f.close()

        print("Processing classifiers")
        for classifier_name, classifier_data_file in self.classifications.items():
            contents = self.classifier_contents(classifier_name)
            print("Processing label '{0}'".format(classifier_name))

            current_value = 0
            for value in contents:
                content_file_path = "{0}_{1}.txt".format(classifier_name, current_value)
                label_file_path = "{0}_{1}.lab".format(classifier_name, current_value)
                with open(path.join(data_path, content_file_path), "w+", encoding="utf8") as content_file:
                    content_file.write(value)
                    content_file.close()
                with open(path.join(data_path, label_file_path), "w+", encoding="utf8") as label_file:
                    label_file.write(classifier_name)
                    label_file.close()
                current_value += 1
            print("Processed label '{0}'".format(classifier_name))

        print("Structure created at '{0}'".format(temporary_path))
        return temporary_path

    def train_model(self):
        """
        Starts the process of training the model by creating a model structure
        and creating the necessary models for classification

        :return: None
        """
        directory_structure = self.create_structure()

        print("Preparing output directory")
        output_path = "{0}_build".format(self.src)

        embeddings_path = path.join(output_path, "{0}.che".format(self.configuration['model']['model_name']))
        scaler_path = path.join(output_path, "{0}.chs".format(self.configuration['model']['model_name']))
        model_file_path = path.join(output_path, "{0}.chm".format(self.configuration['model']['model_name']))
        labels_file_path = path.join(output_path, "{0}.chl".format(self.configuration['model']['model_name']))

        if path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

        print("Initializing CoffeeHouse DLTC Server")
        # noinspection SpellCheckingInspection
        dltc = DLTC()

        print("Creating word to vectors model")
        dltc.train_word2vec(
            path.join(directory_structure, 'model_data'),
            vec_dim=self.configuration['training_properties']['vec_dim']
        )

        print("Fitting Scalers")
        dltc.fit_scaler(path.join(directory_structure, 'model_data'))

        print("Training model")
        dltc.train(
            path.join(directory_structure, 'model_data'),
            self.classifier_labels(),
            nn_model=self.configuration['training_properties']['architecture'],
            batch_size=self.configuration['training_properties']['batch_size'],
            epochs=self.configuration['training_properties']['epoch'],
            test_ratio=self.configuration['training_properties']['test_ratio'],
            verbose=2
        )

        print("Saving data to disk")
        dltc.save_word2vec_model(embeddings_path)
        print("Created file '{0}'".format(embeddings_path))
        dltc.save_scaler(scaler_path)
        print("Created file '{0}'".format(scaler_path))
        dltc.save_model(model_file_path)
        print("Created file '{0}'".format(model_file_path))
        with open(labels_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.classifier_labels(), f, ensure_ascii=False, indent=4)
        print("Created file '{0}'".format(labels_file_path))

        print("Cleaning up")
        if path.exists(directory_structure):
            shutil.rmtree(directory_structure)
        print("Model created at '{0}".format(output_path))
