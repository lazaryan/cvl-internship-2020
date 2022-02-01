"""
File for constants in project
"""

import os

NAME_PROJECT = 'cvl-internship-2020-lazaryan'

# DIR
DIR_DATA = '../data/'
DIR_TMP = 'tmp/'
DIR_TRAIN = 'train/'

# PATH
ROOT = os.getcwd()

PATH_TO_DATASET = os.path.join(ROOT, f'{DIR_DATA}{NAME_PROJECT}/')
PATH_TO_TMP = os.path.join(ROOT, DIR_TMP)
PATH_TO_TRAIN = os.path.join(ROOT, DIR_TRAIN)

# VIEW
COLOR_CONTOUR = "212 59 59"
SIZE_CONTOUR = 5
