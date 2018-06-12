# coding=utf-8
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from Scripts.data_helper import get_clean_data
from Scripts.utils import data_to_bow, data_to_tfidf, split_train_val


def train(data):
    x_train, y_train, x_val, y_val = data
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    print("start training ...")
    rf.fit(x_train, y_train)
    print("training finished.")
    y_train_p = rf.predict(x_train)
    y_val_p = rf.predict(x_val)
    print("train performance:")
    eval(y_train, y_train_p)
    print("val performance:")
    eval(y_val, y_val_p)
    return rf


def eval(y, y_pred):
    print("precision: {}\nrecall: {}\nf1_score: {}".format(precision_score(y, y_pred),
                                                           recall_score(y, y_pred),
                                                           f1_score(y, y_pred)))
    print("train confusion matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == '__main__':
    data_dir = os.path.join(os.path.abspath('..'), 'data')
    tech_path = os.path.join(data_dir, 'tech_data', 'tech_train.csv')
    clean_content, labels = get_clean_data(tech_path, train=True)
    # data_features, vectorizer = data_to_bow(clean_content)
    data_features, vectorizer = data_to_tfidf(clean_content)
    x_train, y_train, x_val, y_val = split_train_val(data_features, labels, split_rate=0.8)
    rf = train((x_train, y_train, x_val, y_val))

    tech_test_path = os.path.join(data_dir, 'tech_data', 'tech_test.csv')
    clean_content_test, labels_test = get_clean_data(tech_test_path, train=True)
    data_features_test = vectorizer.transform(clean_content_test).toarray()
    labels_pred = rf.predict(data_features_test)
    eval(labels_test, labels_pred)
