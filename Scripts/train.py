# coding=utf-8
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from utils import get_train_val


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
    print("acc: {}, f1_score: {}".format(accuracy_score(y, y_pred), f1_score(y, y_pred)))
    print("train confusion matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == '__main__':
    base_dir = '../dataSet/IMDB'
    ftrain = 'labeledTrainData.tsv'
    data = get_train_val(os.path.join(base_dir, ftrain), tfidf=False)
    cls = train(data)