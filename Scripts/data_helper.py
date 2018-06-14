# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import jieba
import jieba.posseg as pseg

from Scripts.config import *


def open_file(filename):
    labels = []
    cmid = []
    title = []
    content = []
    with open(filename, encoding='utf8') as file:
        for line in file.readlines():
            line_list = line.rstrip().split('\t')
            labels.append(int(line_list[0]))
            cmid.append(line_list[1])
            title.append(line_list[2])
            content.append(line_list[3])
    print("labels count: {}".format(len(labels)))
    print("cmid count: {}".format(len(cmid)))
    print("title count: {}".format(len(title)))
    print("content count: {}".format(len(content)))
    return labels, cmid, title, content


def to_csv(csv_path, labels, cmid, title, content):
    data = {'类别': labels, 'cmid': cmid, '标题': title, '内容': content}
    data_pd = pd.DataFrame(data, columns=['类别', 'cmid', '标题', '内容'])
    data_pd.to_csv(csv_path, columns=['类别', 'cmid', '标题', '内容'],
                   encoding='utf8', index=False)
    print('gen csv done.')


def split_data(csv_path, split_rate=0.7):
    data = pd.read_csv(csv_path, header=0, encoding='utf8')
    labels = data['类别']
    cmid = data['cmid']
    title = data['标题']
    content = data['内容']
    x = np.arange(data.shape[0])
    y = labels.values
    ss = StratifiedShuffleSplit(n_splits=1, test_size=1 - split_rate, random_state=0)
    for train_index, test_index in ss.split(x, y):
        return train_index, test_index, labels, cmid, title, content


def get_data(csv_path, train=True):
    data = pd.read_csv(csv_path, header=0, encoding='utf8')
    if train is False:
        return data['标题'].values, data['内容'].values
    return data['类别'].values, data['标题'].values, data['内容'].values


def content_to_words(raw_content, use_pseg=False):
    # 去符号
    if use_pseg is True:
        seg_line = pseg.cut(raw_content, HMM=True)
        words = ''
        for word, flag in seg_line:
            if flag != 'x':
                words += word
                words += ' '
        return words.strip()
    # 不去符号
    seg_line = jieba.cut(raw_content, HMM=True)
    return ' '.join(seg_line)


def get_clean_data(filepath, train=True):
    data = get_data(filepath, train)
    title = data[1]
    content = data[2]
    num_samples = len(title)
    clean_content = []
    for i in range(num_samples):
        clean_content.append(content_to_words(str(title[i]) + ' ' + str(content[i])))
    print("data cleaned.")
    if train is False:
        # return just content
        return clean_content
    # return labels
    return clean_content, data[0]


def max_len(content_list):
    max_length = 0
    for content in content_list:
        max_length = max(max_length, len(content))
    return max_length


if __name__ == '__main__':
    # labels, cmid, title, content = open_file(world_test_path)
    # to_csv(os.path.join(data_dir, 'world_data', 'world_test.csv'), labels, cmid, title, content)
    # train_index, test_index, labels, cmid, title, content = split_data(os.path.join(data_dir, 'finance_data', 'finance.csv'))
    # labels_train = labels[train_index]
    # cmid_train = cmid[train_index]
    # title_train = title[train_index]
    # content_train = content[train_index]
    # to_csv(os.path.join(data_dir, 'finance_data', 'finance_train.csv'), labels_train, cmid_train, title_train, content_train)
    #
    # labels_test = labels[test_index]
    # cmid_test = cmid[test_index]
    # title_test = title[test_index]
    # content_test = content[test_index]
    # to_csv(os.path.join(data_dir, 'finance_data', 'finance_test.csv'), labels_test, cmid_test, title_test, content_test)
    content, labels = get_clean_data(FINACE_TRAIN_PATH)