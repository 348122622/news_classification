# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def data_to_bow(clean_content, analyzer="word", ngram_range=(1, 1), max_text_len=5000):
    vectorizer = CountVectorizer(analyzer=analyzer,
                                 ngram_range=ngram_range,
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_text_len)
    data_features = vectorizer.fit_transform(clean_content)
    print("feature extracted.")
    # if tfidf is True:
    #     transformer = TfidfTransformer()
    #     data_features = transformer.fit_transform(data_features)
    return data_features.toarray(), vectorizer


def data_to_tfidf(clean_content, analyzer="word", ngram_range=(1, 1), max_text_len=5000):
    vectorizer = TfidfVectorizer(analyzer=analyzer,
                                 ngram_range=ngram_range,
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=max_text_len)
    data_features = vectorizer.fit_transform(clean_content)
    print("feature extracted.")
    return data_features.toarray(), vectorizer


def split_train_val(data_features, labels, split_rate=0.8):
    ss = StratifiedShuffleSplit(n_splits=1, test_size=1 - split_rate, random_state=0)
    print("validate set splited.")
    for train_index, val_index in ss.split(data_features, labels):
        return data_features[train_index], labels[train_index], \
               data_features[val_index], labels[val_index]


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]


def eval(y, y_pred):
    print("precision: {:.2%}\nrecall: {:.2%}\nf1_score: {:.2%}".format(precision_score(y, y_pred),
                                                                       recall_score(y, y_pred),
                                                                       f1_score(y, y_pred)))
    print("train confusion matrix:")
    print(confusion_matrix(y, y_pred))


