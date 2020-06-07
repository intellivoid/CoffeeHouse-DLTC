# CoffeeHouse DLTC

CoffeeHouse Deep Learning Classification Engine is a method for creating K2 Models on large data
to predict labels from them. For example, you can train the model on a bunch of "Sports" articles
and "Political" articles (with the appropriate labels assigned to each article) and train the
model. You can give the model a new article that's either "Sports" or "Political" related and 
the model will be able to predict the likely-hood of the article being Political or Sports related.

This was forked from [magpie](https://github.com/inspirehep/magpie) but rewritten to handle data
and the training process more quickly and efficiently than the original project. 

# Installation

```shell script
python3 setup.py install
```

# Usage

Create a directory for your model, your directory must contain a model.json file
formatted like this

```json
{
    "model": {
        "name": "Spam Ham",
        "model_name": "spam_ham",
        "author": "Zi Xing",
        "version": "1.0.0.0",
        "description": "Model for predicting messages which contains spam or ham"
    },
    "training_properties":{
        "epoch": 35,
        "vec_dim": 100,
        "test_ratio": 0.2,
        "architecture": "cnn",
        "batch_size": 64
    },
    "classification": [
        {"l": "spam", "f": "spam.dat"},
        {"l": "ham", "f": "ham.dat"}
    ]
}

 ```

### Model

| Property Name | Description                                                |
|---------------|------------------------------------------------------------|
| name          | The name of the model                                      |
| model_nme     | The safe name of the model which is used for IO operations |
| author        | The author which constructed the data for the model        |
| version       | The version of the model                                   |
| description   | The description of the model, what it does, etc.           |


### Training Properties

| Property Name | Description                                                                                                                                                                                                           |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| epoch         | The amount of training sessions the model must run through                                                                                                                                                            |
| vec_dim       | The amount of word vector recreations it goes through                                                                                                                                                                 |
| test_ratio    | splits data into train & test datasets and evaluates itself after every epoch displaying it's current loss and accuracy. The default value of  `test_ratio` is 0 meaning that all the data will be used for training. |
| architecture  | The type of model to train on, the possible values are `cnn` and `rnn`                                                                                                                                                |
| batch_size    | The size of the batch for training purposes                                                                                                                                                                           |

### Classification

| Property Name | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| l             | The label for the data, eg; `spam`, `ham`...                                |
| f             | The name of the .dat file which consists of the data split into line breaks |


## Training the model

To train the model, the model must be clustered into a structured directory which will create a
bunch of files for the data and labels which would be easier to manage and train the data from
those files. In which after the temporary directory will be deleted

```python
from coffeehouse_dltc.chmodel.configuration import Configuration

# Model directory must contain model.json and the required .dat files
configuration = Configuration('<Model Directory>')
configuration.train_model()
```

Once this process is done, a output directory will be created with all the generated models

| File Extension | Description                                  |
|----------------|----------------------------------------------|
| `.che`         | This file contains the word vectors          |
| `.chs`         | File format responsible for the scarler data |
| `.chm`         | Main classification model                    |
| `.chl`         | JSON File format which contains the labels   |

All these files are important in order for the model data to be loaded correctly into memory


## Classifying data

Assuming the model files has been created, you can load the model cluster and
predict from text or file input

```python
from coffeehouse_dltc.main import DLTC

dltc = DLTC()
dltc.load_model_cluster('<Model Directory Output>')

dltc.predict_from_text("Hello World")
# [('ham', 0.9650128), ('spam', 0.040875915)]


dltc.predict_from_file("text.txt")
# [('spam', 0.61647576), ('ham', 0.42338383)]
```


## From the CLI

You can access CoffeeHouse-DLTC's features from the command-line interface.

```shell script
python3 -m coffeehouse_dltc --model-info <source directory>
python3 -m coffeehouse_dltc --train-model <source directory>
python3 -m coffeehouse_dltc --test-model <built model directory>
```