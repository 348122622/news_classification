# coding=utf-8
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit


def data_to_bow(clean_content, max_text_len=5000):
    vectorizer = CountVectorizer(analyzer="word",
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


def data_to_tfidf(clean_content, max_text_len=5000):
    vectorizer = TfidfVectorizer(analyzer="word",
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


# def get_train_val(filepath, split_rate=0.8, tfidf=False):
#     clean_reviews, labels = get_clean_data(filepath, train=True)
#     data_features, vectorizer = data_to_bow(clean_reviews, tfidf=True)
#     return split_train_val(data_features, labels, split_rate=0.8)

