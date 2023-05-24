import random
import cv2
from caltech20 import *


def dense_SIFT(data_set, step_size=10, train=True, sample_portion=0.4):
    for data in data_set:
        img = data['img']
        sift = cv2.SIFT_create()
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
              for x in range(0, img.shape[1], step_size)]

        if train == True:
            sample_num = int(len(kp) * sample_portion)
            kp = random.sample(kp, k=sample_num)

        dense_kp, desc = sift.compute(img, kp)
        data['kp'] = dense_kp
        data['desc'] = desc

    return data_set


if __name__ == '__main__':
    path = './caltech20/'
    train_set, test_set = load_data(path)
    #  image's shape: (300,204,3)
    train_set = dense_SIFT(train_set, train=True)

    # img=train_set[0]['img']
    # kp=train_set[0]['kp']
    # draw_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(draw_img)
    # plt.show()
