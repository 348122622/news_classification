# coding=utf-8
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from Scripts.data_helper import get_clean_data
from Scripts.utils import data_to_bow, data_to_tfidf, split_train_val


def train(data):
    x_train, y_train, x_val, y_val = data
    cls = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # cls = GradientBoostingClassifier(n_estimators=100)
    print("start training ...")
    cls.fit(x_train, y_train)
    print("training finished.")
    y_train_p = cls.predict(x_train)
    y_val_p = cls.predict(x_val)
    print("train performance:")
    eval(y_train, y_train_p)
    print("val performance:")
    eval(y_val, y_val_p)
    return cls


def train_cv(x, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=0)
    cls = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    cv_res = cross_validate(cls, x, y,
                            scoring=['precision', 'recall', 'f1'],
                            cv=cv,
                            n_jobs=-1,
                            return_train_score=False)
    print("precision mean: {:.2%}(+/-{:.2%})".format(cv_res['test_precision'].mean(), cv_res['test_precision'].std()))
    print("recall mean: {:.2%}(+/-{:.2%})".format(cv_res['test_recall'].mean(), cv_res['test_recall'].std()))
    print("f1 mean: {:.2%}(+/-{:.2%})".format(cv_res['test_f1'].mean(), cv_res['test_f1'].std()))
    return cv_res


def eval(y, y_pred):
    print("precision: {:.2%}\nrecall: {:.2%}\nf1_score: {:.2%}".format(precision_score(y, y_pred),
                                                                       recall_score(y, y_pred),
                                                                       f1_score(y, y_pred)))
    print("train confusion matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == '__main__':
    data_dir = os.path.join(os.path.abspath('..'), 'data')
    tech_path = os.path.join(data_dir, 'tech_data', 'tech_train.csv')
    clean_content, labels = get_clean_data(tech_path, train=True)
    # data_features, vectorizer = data_to_bow(clean_content)
    data_features, vectorizer = data_to_tfidf(clean_content, ngram_range=(1, 3))
    x_train, y_train, x_val, y_val = split_train_val(data_features, labels, split_rate=0.8)
    # cls = train((x_train, y_train, x_val, y_val))

    tech_test_path = os.path.join(data_dir, 'tech_data', 'tech_test.csv')
    clean_content_test, labels_test = get_clean_data(tech_test_path, train=True)
    data_features_test = vectorizer.transform(clean_content_test).toarray()

    cls = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    cls.fit(data_features, labels)
    labels_pred = cls.predict(data_features_test)
    eval(labels_test, labels_pred)


