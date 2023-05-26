import os
import os.path as osp
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def label_dic():
    label_l = os.listdir('./caltech20/')
    label_l.sort()
    label_dic = {key: i for i, key in enumerate(label_l)}
    return label_dic


def make_img_list(rootdir=".", suffix=""):
    return [
        osp.join(path, filename)
        for path, dirs, filenames in os.walk(rootdir)
        # ./caltech20/ant/, [], [imgname.jpg,...]
        for filename in filenames
        if filename.endswith(suffix)
    ]


def load_data(path='./caltech20/'):
    files_path = make_img_list(rootdir=path, suffix='.jpg')
    global label_dictionary
    label_dictionary = label_dic()

    data_l = []
    for img_path in files_path:
        label = img_path.split('/')[-2]
        label = label_dictionary[label]
        label_nd = np.array([label])
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)  # type: 'numpy.ndarray'
        data = {'img': img, 'label': label_nd, 'path': img_path}
        data_l.append(data)

    # split into train/test
    train_set, test_set = train_test_split(data_l, test_size=0.2, shuffle=True, random_state=0)
    return train_set, test_set


if __name__ == '__main__':
    label_dictionary = label_dic()
    print(label_dictionary)
