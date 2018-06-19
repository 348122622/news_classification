# coding=utf-8
import requests
import json
from bs4 import BeautifulSoup as bs
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


def get_content(cmid):
    url = 'http://newsarticle.cms.webdev.com/newsarticle/news/getOMArticleById?articleId=' + cmid[:-2]
    try:
        res = json.loads(requests.get(url).content)
        content_str = res['response']['data']['content']
    except:
        print("爬取异常")
        return ''
    else:
        soup = bs(content_str, 'html.parser')
        content_str = soup.get_text().strip()
        return content_str


def jiedu_data_prepare(input_csv, output_csv):
    data = pd.read_csv(input_csv, header=0, encoding='utf-8')
    data['cmid'] = data['url'].apply(lambda x: x.split('/')[-1].strip())
    data['内容'] = data['cmid'].apply(lambda x: get_content(x))
    data['类别'] = data['解读评论']
    data[['类别', 'cmid', '标题', '内容']].to_csv(output_csv, index=False, encoding='utf8')
    print("data prepared.")
    return data

if __name__ == '__main__':
    # file to csv
    # labels, cmid, title, content = open_file(world_test_path)
    # to_csv(os.path.join(data_dir, 'world_data', 'world_test.csv'), labels, cmid, title, content)

    # split train and test set
    all_path = NEW_JIEDU_PATH
    train_path = NEW_JIEDU_TRAIN_PATH
    test_path = NEW_JIEDU_TEST_PATH

    train_index, test_index, labels, cmid, title, content = split_data(all_path)
    labels_train = labels[train_index]
    cmid_train = cmid[train_index]
    title_train = title[train_index]
    content_train = content[train_index]
    to_csv(train_path, labels_train, cmid_train, title_train, content_train)
    #
    labels_test = labels[test_index]
    cmid_test = cmid[test_index]
    title_test = title[test_index]
    content_test = content[test_index]
    to_csv(test_path, labels_test, cmid_test, title_test, content_test)


    # content, labels = get_clean_data(FINACE_TRAIN_PATH)
    # JIEDU_PATH = os.path.join(DATA_DIR, 'jiedu', 'new_jiedu_data.csv')
    # output = os.path.join(DATA_DIR, 'jiedu', 'new_jiedu.csv')
    # df = jiedu_data_prepare(JIEDU_PATH, output)