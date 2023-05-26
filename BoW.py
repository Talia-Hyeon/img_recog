import random
import warnings

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from caltech20 import *

warnings.filterwarnings("ignore")


def dense_SIFT(data_set, step_size=10, sampling=True, sample_portion=0.4):
    for data in data_set:
        img = data['img']
        sift = cv2.SIFT_create()
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
              for x in range(0, img.shape[1], step_size)]

        if sampling == True:
            sample_num = int(len(kp) * sample_portion)
            kp = random.sample(kp, k=sample_num)

        dense_kp, desc = sift.compute(img, kp)
        data['kp'] = dense_kp
        data['desc'] = desc

    return data_set


def get_all_item(data, key):
    item_l = []
    for i in data:
        value = i[key]
        item_l.append(value)
    all_item = np.concatenate(item_l, axis=0)
    return all_item


def scaling_pca(all_desc, num_com=0.95):  # 95% 분산 유지하도록 수정 (60-70개)
    # scaling
    scaler = StandardScaler()
    f_scaled = scaler.fit_transform(all_desc)
    # pca
    pca = PCA(n_components=num_com)
    f_pca = pca.fit_transform(f_scaled)
    return f_pca, pca


def encode_img(desc, pca, km):
    desc_pca = pca.transform(desc)
    labels = km.predict(desc_pca)
    img_vector = np.zeros(num_cluster)  # 1500
    for cluster in labels:
        img_vector[cluster] += 1
    return img_vector


if __name__ == '__main__':
    path = './caltech20/'
    train_set, test_set = load_data(path)  # image's shape: (300,204,3)

    # extract feature for sample
    sample_set = dense_SIFT(train_set, sampling=True)  # desc'shape: (252,128)
    print("# of training img: {}".format(len(sample_set)))
    # visualization
    img = train_set[0]['img']
    kp = train_set[0]['kp']
    draw_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(draw_img)
    # plt.show()

    # pca
    sample_desc = get_all_item(data=sample_set, key='desc')
    feature_pca, pca = scaling_pca(sample_desc)
    print("reduced demension: {}".format(feature_pca[0].shape))

    # k-means clusturing
    global num_cluster
    num_cluster = 1500
    km = KMeans(n_clusters=num_cluster, random_state=21)
    km.fit(feature_pca)

    # learn visual dictionary
    # for train set
    train_set = dense_SIFT(train_set, sampling=False)  # extract feature
    img_vectors_train = []
    for train_img in train_set:
        train_desc = train_img['desc']
        train_img_vector = encode_img(desc=train_desc, pca=pca, km=km)
        img_vectors_train.append(train_img_vector)
    X_train = np.vstack(img_vectors_train)
    np.save('train_img_vector.npy', X_train)
    y_train = get_all_item(data=train_set, key='label')
    np.save('train_label.npy', y_train)

    # for test set
    test_set = dense_SIFT(test_set, sampling=False)
    img_vectors_test = []
    for test_img in test_set:
        test_desc = test_img['desc']
        test_img_vector = encode_img(desc=test_desc, pca=pca, km=km)
        img_vectors_test.append(test_img_vector)
    X_test = np.concatenate(img_vectors_test, axis=0)
    np.save('test_img_vector.npy', X_test)
    y_test = get_all_item(data=test_set, key='label')
    np.save('test_label.npy', y_test)
