# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import tensorflow.contrib.keras as kr
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score
from Scripts.data_helper import get_clean_data
from Scripts.utils import data_to_bow, data_to_tfidf, split_train_val, batch_iter
from Scripts.models.textCNN import TextCNN


def preprocess(filename):
    # load data
    # labels type: np.array
    x_text, labels = get_clean_data(filename)
    y = kr.utils.to_categorical(labels, num_classes=len(np.unique(labels)))

    # build vocabulary
    vocab_size = 5000
    max_document_length = 500
    tokenizer = Tokenizer(num_words=vocab_size, char_level=False)
    tokenizer.fit_on_texts(x_text)
    x_seqs = tokenizer.texts_to_sequences(x_text)
    x = pad_sequences(x_seqs, maxlen=max_document_length, truncating='post', padding='post')
    x_train, y_train, x_val, y_val = split_train_val(x, y, split_rate=0.8)
    print("Vocabulary Size: {:d}".format(len(tokenizer.word_index)))
    print("Train: {:d}\nVal: {:d}".format(len(y_train), len(y_val)))
    return x_train, y_train, x_val, y_val, tokenizer


def train(x_train, y_train, x_val, y_val, vocab_size, num_epochs=30):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = TextCNN(sequnce_length=x_train.shape[1],
                          num_classes=y_train.shape[1],
                          vocab_size=vocab_size,
                          embedding_size=300,
                          filter_sizes=[3, 4, 5],
                          num_filters=256,
                          l2_reg_lambda=0.0)

            # define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # keep track of gradient values and sparsity(optional)
            # ......

            # Initializer all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {cnn.input_x: x_batch,
                             cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: 0.5}

                _, step, loss, acc = sess.run([train_op,
                                                             global_step,
                                                             cnn.loss,
                                                             cnn.accuracy], feed_dict)
                print("Step: {}\ttrain loss: {:.4f}\ttrain acc: {:.2%}".format(step, loss, acc))

            def evaluate(x_, y_, batch_size=128):
                total_loss = 0.0
                data_len = len(x_)
                num_batch = int((data_len - 1) / batch_size) + 1

                y_true = np.argmax(y_, 1)
                y_pred = np.zeros(shape=len(x_), dtype=np.int32)  # 保存预测结果
                for i in range(num_batch):  # 逐批次处理
                    start_id = i * batch_size
                    end_id = min((i + 1) * batch_size, data_len)
                    feed_dict = {cnn.input_x: x_[start_id:   end_id],
                                 cnn.input_y: y_[start_id: end_id],
                                 cnn.dropout_keep_prob: 1.0}

                    loss, y_pred[start_id:end_id] = sess.run([cnn.loss, cnn.predictions], feed_dict=feed_dict)
                    total_loss += batch_size * loss

                print("val loss: {:.4f}\tprecision: {:.2%}\trecall: {:.2%}\tf1_score: {:.2%}"
                      .format(total_loss / data_len,
                              precision_score(y_true, y_pred),
                              recall_score(y_true, y_pred),
                              f1_score(y_true, y_pred)))

            # num_epochs = 30
            for epoch in range(num_epochs):
                print("Epoch--{}:".format(epoch + 1))
                for x_batch, y_batch in batch_iter(x_train, y_train, batch_size=64):
                    train_step(x_batch, y_batch)
                evaluate(x_val, y_val)
            print("train finished.")

def main():
    data_dir = os.path.join(os.path.abspath('..'), 'data')
    tech_path = os.path.join(data_dir, 'tech_data', 'tech_train.csv')
    x_train, y_train, x_val, y_val, tokenizer = preprocess(tech_path)
    train(x_train, y_train, x_val, y_val, len(tokenizer.word_index))

# if __name__ == '__main__':
#     main()