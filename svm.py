import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from caltech20 import label_dic


def svm_pip(x_train, y_train, x_test, y_test, label):
    # scale
    scaler = MinMaxScaler()
    x_train_min = scaler.fit_transform(x_train)

    svc = SVC(random_state=42)
    # adjust hyper-parameter
    param_grid = [
        {'C': [0.5, 1.0, 5, 10, 15, 20]},
        {'kernel': ['linear', 'rbf', 'poly', 'sigmoid']},
        {'degree': [2, 3]}, {'gamma': ['scale', 'auto']}
    ]
    grid_search = GridSearchCV(svc, param_grid, cv=5)
    grid_search.fit(x_train_min, y_train)
    print("best hyper-parameter\n{}".format(grid_search.best_params_))
    svc_min = grid_search.best_estimator_

    # svc_min = Pipeline([
    #     ("scaler", MinMaxScaler()),
    #     ("svc_clf", SVC(C=10, random_state=42))
    # ])
    svc_min.fit(x_train, y_train)

    # test
    y_pred = svc_min.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy of SVC_minmax=', accuracy)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('./figure/heatmap_1700.png')
    return svc_min


if __name__ == '__main__':
    # load data: .npy
    x_train = np.load('train_img_vector_1700.npy')
    y_train = np.load('train_label.npy')
    x_test = np.load('test_img_vector_1700.npy')
    y_test = np.load('test_label.npy')
    print("train) x's shape: {}, y's shape:{}".format(x_train.shape, y_train.shape))
    print("test) x's shape: {}, y's shape:{}".format(x_test.shape, y_test.shape))

    # get label list
    label_dictionary = label_dic()
    label_l = list(label_dictionary.keys())
    label_l.sort()

    scv = svm_pip(x_train, y_train, x_test, y_test, label=label_l)
