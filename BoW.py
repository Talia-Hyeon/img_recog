import cv2
from caltech20 import *


def dense_SIFT(train_set, step_size=5):
    for data in train_set:
        img = data['img']
        sift = cv2.SIFT_create()
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
              for x in range(0, img.shape[1], step_size)]

        dense_kp, desc = sift.compute(img, kp)
        data['kp'] = dense_kp
        data['desc'] = desc

        # draw_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(draw_img)
        # plt.show()

    return train_set


if __name__ == '__main__':
    path = './caltech20/'
    train_set, test_set = load_data(path)
    #  image's shape: (300,204,3)
    train_set = dense_SIFT(train_set)
    print(train_set[0])
