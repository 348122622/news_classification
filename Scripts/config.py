# coding=utf-8
import sys
import os

sys.path.append(os.path.abspath('..'))

# base dir
DATA_DIR = os.path.join(os.path.abspath('..'), 'data')

# original data file
FINANCE_PATH = os.path.join(DATA_DIR, 'finance_data', 'train_data')
TECH_PATH = os.path.join(DATA_DIR, 'tech_data', 'train_data')

# train and test csv
FINACE_TRAIN_PATH = os.path.join(DATA_DIR, 'finance_data', 'finance_train.csv')
FINACE_TEST_PATH = os.path.join(DATA_DIR, 'finance_data', 'finance_test.csv')
TECH_TRAIN_PATH = os.path.join(DATA_DIR, 'tech_data', 'tech_train.csv')
TECH_TEST_PATH = os.path.join(DATA_DIR, 'tech_data', 'tech_test.csv')
WORLD_TRAIN_PATH = os.path.join(DATA_DIR, 'world_data', 'world_train.csv')
WORLD_TEST_PATH = os.path.join(DATA_DIR, 'world_data', 'world_test.csv')


