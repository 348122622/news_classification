# coding=utf-8
import sys
import os

sys.path.append(os.path.abspath('..'))

# base dir
DATA_DIR = os.path.join(os.path.abspath('..'), 'data')

# original data file
FINANCE_PATH = os.path.join(DATA_DIR, 'finance_data', 'train_data')
TECH_PATH = os.path.join(DATA_DIR, 'tech_data', 'train_data')
JIEDU_PATH = os.path.join(DATA_DIR, 'jiedu', 'jiedu.csv')
NEW_JIEDU_PATH = os.path.join(DATA_DIR, 'jiedu', 'new_jiedu.csv')

# train and test csv
# finance
FINACE_TRAIN_PATH = os.path.join(DATA_DIR, 'finance_data', 'finance_train.csv')
FINACE_TEST_PATH = os.path.join(DATA_DIR, 'finance_data', 'finance_test.csv')

# tech
TECH_TRAIN_PATH = os.path.join(DATA_DIR, 'tech_data', 'tech_train.csv')
TECH_TEST_PATH = os.path.join(DATA_DIR, 'tech_data', 'tech_test.csv')

# world
WORLD_TRAIN_PATH = os.path.join(DATA_DIR, 'world_data', 'world_train.csv')
WORLD_TEST_PATH = os.path.join(DATA_DIR, 'world_data', 'world_test.csv')

# jiedu
JIEDU_TRAIN_PATH = os.path.join(DATA_DIR, 'jiedu', 'jiedu_train.csv')
JIEDU_TEST_PATH = os.path.join(DATA_DIR, 'jiedu', 'jiedu_test.csv')
NEW_JIEDU_TRAIN_PATH = os.path.join(DATA_DIR, 'jiedu', 'new_jiedu_train.csv')
NEW_JIEDU_TEST_PATH = os.path.join(DATA_DIR, 'jiedu', 'new_jiedu_test.csv')


