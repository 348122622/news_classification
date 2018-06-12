# coding=utf-8
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def data_to_bow(clean_content, tfidf=False):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)
    data_features = vectorizer.fit_transform(clean_content)
    print("feature extracted.")
    if tfidf is True:
        transformer = TfidfTransformer()
        data_features = transformer.fit_transform(data_features)
    return data_features.toarray(), vectorizer


def split_train_val(data_features, labels, split_rate=0.8):
    total_size = data_features.shape[0]
    shuffle = np.random.permutation(total_size)
    data_features = data_features[shuffle]
    labels = np.array(labels[shuffle])
    train_size = int(total_size * split_rate)
    print("validate set splited.")
    return data_features[:train_size], labels[:train_size], \
           data_features[train_size:], labels[train_size:]


# def get_train_val(filepath, split_rate=0.8, tfidf=False):
#     clean_reviews, labels = get_clean_data(filepath, train=True)
#     data_features, vectorizer = data_to_bow(clean_reviews, tfidf=Ture)
#     return split_train_val(data_features, labels, split_rate=0.8)

