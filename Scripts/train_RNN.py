# author - Richard Liao
# Dec 26 2016
import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec

from Scripts.data_helper import get_clean_data
from Scripts.utils import split_train_val, eval

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
char_filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n，。！？“‘、（）……【】：《》'

def preprocess(filename):
    # load data
    # labels type: np.array
    x_text, labels = get_clean_data(filename)
    y = to_categorical(labels, num_classes=len(np.unique(labels)))

    # build vocabulary
    vocab_size = 5000
    max_document_length = 500
    tokenizer = Tokenizer(filters=char_filters, num_words=vocab_size, char_level=False)
    tokenizer.fit_on_texts(x_text)
    x_seqs = tokenizer.texts_to_sequences(x_text)
    x = pad_sequences(x_seqs, maxlen=max_document_length, truncating='post', padding='post')
    print("Vocabulary Size: {:d}".format(len(tokenizer.word_index)))
    print('Shape of data tensor:', x.shape)
    print('Shape of label tensor:', y.shape)

    x_train, y_train, x_val, y_val = split_train_val(x, y, split_rate=0.8)
    print('amount of train samples:', y_train.shape[0])
    print('amount of val samples:', y_val.shape[0])
    print('number of positive and negative reviews:')
    print('train:', y_train.sum(axis=0))
    print('val:', y_val.sum(axis=0))

    return x_train, y_train, x_val, y_val, tokenizer


data_dir = os.path.join(os.path.abspath('..'), 'data')
tech_path = os.path.join(data_dir, 'tech_data', 'tech_train.csv')
x_train, y_train, x_val, y_val, tokenizer = preprocess(tech_path)

vocab_size = len(tokenizer.word_index)
embedding_matrix = np.random.random((vocab_size + 1, EMBEDDING_DIM))

embedding_layer = Embedding(vocab_size + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=500,
                            trainable=True)

sequence_input = Input(shape=(500,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
preds = Dense(2, activation='softmax')(l_lstm)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=18, batch_size=128, verbose=1)

y_train_pred = np.argmax(model.predict(x_train), axis=1)
y_train_ = np.argmax(y_train, axis=1)
eval(y_train_, y_train_pred)
y_val_pred = np.argmax(model.predict(x_val), axis=1)
y_val_ = np.argmax(y_val, axis=1)
eval(y_val_, y_val_pred)

tech_test_path = os.path.join(data_dir, 'tech_data', 'tech_test.csv')
clean_content_test, labels_test = get_clean_data(tech_test_path, train=True)
x_test_seqs = tokenizer.texts_to_sequences(clean_content_test)
x_test = pad_sequences(x_test_seqs, maxlen=500, truncating='post', padding='post')
y_test = to_categorical(labels_test, num_classes=len(np.unique(labels_test)))
y_test_pred = np.argmax(model.predict(x_test), axis=1)
y_test_ = np.argmax(y_test, axis=1)
eval(y_test_, y_test_pred)