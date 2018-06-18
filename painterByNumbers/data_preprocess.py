import zipfile
import sys
import os
import io
import json
import time
import numpy as np
import pandas as pd
import urllib3
from urllib3.util import Retry
import multiprocessing

from PIL import Image
from tqdm import tqdm
import cv2
# follow the following solution if you can't import matplotlib.pyplot
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
import matplotlib.pyplot as plt  # import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.image as mpimg
from ast import literal_eval
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

data_dir = "/Users/fan_heng/.kaggle/competitions/painter-by-numbers"

train_datasets = ['train_{}.zip'.format(n) for n in range(1,10)]


class KaggleDataDownloader():
    def __init__(self, credentials=None, verbose=True, logger=None):
        self.crendentials = credentials
        self.verbose = verbose
        self.logger = logger

    def download_dataset(self, competition=None, file_name=None):
        try:
            os.system('kaggle competitions files -c {}'.format(competition))
        except:
            print("Use a valid competition name!")

        self.kaggle_data_dir = "/Users/fan_heng/.kaggle/competitions/{}".format(competition)

        # download dataset
        if file_name is None:
            # download all files
            os.system('kaggle competitions download -c {}'.format(competition))
        else:
            # pass if file alreay exists
            if os.path.isfile(os.path.join(self.kaggle_data_dir, file_name)):
                print("File {} already exists".format(file_name))
            elif os.path.exists(os.path.join(self.kaggle_data_dir, os.path.splitext(file_name)[0])):
                print("Unziped File of {} already exists".format(file_name))
            else:
                os.system('kaggle competitions download -c {} -f {}'.format(competition, file_name))

    def unzip_dataset(self, file_name, keep=False):
        base_f = os.path.splitext(file_name)[0]
        if (file_name.endswith('.zip')) & (not os.path.exists(os.path.join(data_dir, base_f))):
            print(os.path.join(data_dir, base_f))
            print(os.path.exists(os.path.join(data_dir, base_f)))
            zip_ref = zipfile.ZipFile(os.path.join(self.kaggle_data_dir, file_name), 'r')
            zip_ref.extractall(data_dir)
            zip_ref.close()
        #remove zipfile if don't want to keep
        if not keep:
            os.remove(os.path.join(self.kaggle_data_dir, file_name))

    def downsizing_image_file(self, image_dir, image_file=None, shape=None, proportion=0.5, new_folder_name='default_data_small'):
        '''
        Make image file smaller to save storage space
        shape: None, (height, width)
        '''
        # new_folder = os.path.basename(image_dir) + '_small'
        new_folder = new_folder_name
        out_dir = os.path.join(os.path.dirname(image_dir), new_folder)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        def resize_image(file_name):
            if os.path.isfile(os.path.join(out_dir, file_name)):
                return
            img = cv2.imread(os.path.join(image_dir, file_name))
            if img is None:
                return
            if shape is None:
                img = cv2.resize(img, (img.shape[0]*proportion, img.shape[1]*proportion), interpolation = cv2.INTER_AREA)
            else:
                img = cv2.resize(img, (min(shape[0], img.shape[0]), min(shape[1], img.shape[1])), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, file_name), img)

        if image_file is None:
            for i, file_name in enumerate(os.listdir(image_dir)):
                resize_image(file_name)
                if i % 100.0 == 0:
                    print("Processed {} images".format(i))
        else:
            resize_image(image_file)



def unzip_datafile(data_dir):
    for f in os.listdir(data_dir):
        base_f = os.path.splitext(f)[0]
        if (f.endswith('.zip')) & (not os.path.isfile(os.path.join(data_dir, base_f))):
            print(os.path.join(data_dir, base_f))
            zip_ref = zipfile.ZipFile(os.path.join(data_dir, f), 'r')
            zip_ref.extractall(data_dir)
            zip_ref.close()



def reshape_image():
    pass

def download_image():
    reshape_image()
    pass


if __name__ == '__main__':
    downloader = KaggleDataDownloader()
    for train_zip in train_datasets[1:]:
        downloader.download_dataset(competition = 'painter-by-numbers', file_name = train_zip)
        if not os.path.exists(os.path.join(downloader.kaggle_data_dir, os.path.splitext(train_zip)[0])):
            downloader.unzip_dataset(file_name=train_zip)
        img_dir = os.path.join(downloader.kaggle_data_dir, os.path.splitext(train_zip)[0])
        try:
            downloader.downsizing_image_file(img_dir,shape=(256,256), new_folder_name='train_small')
        except:
            continue

    start_path = '/Users/fan_heng/.kaggle/competitions/painter-by-numbers/train_small'  # To get size of current directory
    total_size = 0
    for path, dirs, files in os.walk(start_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    print("Directory size: " + str(total_size/1024./1024) + ' MB')
