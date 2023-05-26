from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def svm_pip(x_train, y_train, x_test, y_test):
    svc = SVC(random_state=42)
    # adjust hyper-parameter
    param_grid = [
        {'C': [0.5, 1.0, 5, 10, 15, 20]},
        {'kernel': ['linear', 'rbf', 'poly', 'sigmod']},
        {'degree': [2, 3]}, {'gamma': ['scale', 'auto']}
    ]
    grid_search = GridSearchCV(svc, param_grid, cv=5)

    # MinMaxScaler()
    ##    print('find best prameters of SGDClassifier_MinMax')
    ##    grid_search.fit(x_train_min, y_train)
    ##    print(grid_search.best_params_)
    # accuracy
    svc_min = Pipeline([
        ("scaler", MinMaxScaler()),
        ("svc_clf", SVC(C=10, random_state=42))
    ])
    svc_min.fit(x_train, y_train)
    y_pred = svc_min.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy of SVC_minmax=', accuracy)