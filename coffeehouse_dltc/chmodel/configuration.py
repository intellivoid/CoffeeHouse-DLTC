import os
import json
import shutil
from os import path

from coffeehouse_dltc import DLTC


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
            raise ValueError(
                "The classification label '{0}' is not defined in the configuration".format(classification_name))

    def classifier_contents(self, classification_name):
        if classification_name in self.classifications:
            with open(self.classifications[classification_name], 'r', encoding="utf8") as f:
                return f.read().splitlines()
        else:
            raise ValueError(
                "The classification label '{0}' is not defined in the configuration".format(classification_name))

    def classifier_labels(self):
        classifier_labels = []
        for classifier_name, classifier_data_file in self.classifications.items():
            classifier_labels.append(classifier_name)
        return classifier_labels

    def create_structure(self):
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
        directory_structure = self.create_structure()

        print("Preparing output directory")
        output_path = "{0}_output".format(self.src)

        embeddings_path = path.join(output_path, "embeddings")
        scaler_path = path.join(output_path, "scaler")
        model_file_path = path.join(output_path, "model.chm")
        labels_file_path = path.join(output_path, "labels.json")

        if path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

        print("Training model")
        # noinspection SpellCheckingInspection
        dltc = DLTC()
        dltc.init_word_vectors(
            path.join(directory_structure, 'model_data'),
            vec_dim=self.configuration['training_properties']['vec_dim']
        )
        dltc.train(
            directory_structure,
            self.classifier_labels(), epochs=self.configuration['training_properties']['epoch']
        )

        print("Saving data to disk")
        dltc.save_word2vec_model(embeddings_path)
        dltc.save_scaler(scaler_path)
        dltc.save_model(model_file_path)
        with open(labels_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.classifier_labels(), f, ensure_ascii=False, indent=4)
        print("Done")
