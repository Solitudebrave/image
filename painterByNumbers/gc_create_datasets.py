# gc: genre classification model
# simple image classification implementation
import os
import numpy as np
import pandas as pd
import urllib3
from urllib3.util import Retry
import multiprocessing

from PIL import Image
from tqdm import tqdm
import cv2
import random

'''
folder structure: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
'''

data_dir = "/Users/fan_heng/.kaggle/competitions/painter-by-numbers"

data_folders = ['train_{}_small'.format(n) for n in range(1,10)]

def build_data_directory():
    ind_df = pd.read_csv('train_info.csv')
    gg = ind_df.groupby('genre')
    for name, group in gg:
        if not os.path.exists(os.path.join('gc_data/train', name.replace(' ', '_').replace('-', '_'))):
            os.makedirs(os.path.join('gc_data/train', name.replace(' ', '_').replace('-', '_')))
        if not os.path.exists(os.path.join('gc_data/test', name.replace(' ', '_').replace('-', '_'))):
            os.makedirs(os.path.join('gc_data/test', name.replace(' ', '_').replace('-', '_')))
        for image_file in group['filename']:
            if os.path.exists('/Users/fan_heng/.kaggle/competitions/painter-by-numbers/train_small/{}'.format(image_file)):
                # 10% as test and 90% as training data
                if random.uniform(0, 1) <= 0.1:
                    os.system('cp /Users/fan_heng/.kaggle/competitions/painter-by-numbers/train_small/{} gc_data/test/{}'.format(image_file, name))
                else:
                    os.system('cp /Users/fan_heng/.kaggle/competitions/painter-by-numbers/train_small/{} gc_data/train/{}'.format(image_file, name))




if __name__ == '__main__':
    build_data_directory()
