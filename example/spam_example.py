from coffeehouse_dltc.main import DLTC

dtlc = DLTC(
    keras_model='/save/my/model/here.h5',
    word2vec_model='/save/my/embeddings/here',
    scaler='/save/my/scaler/here',
    labels=['cat', 'dog', 'cow']
)