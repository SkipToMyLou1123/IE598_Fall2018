import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Part 1
# Splitting data into 90% training and 10% test data

in_sample_acc = []
out_sample_acc = []
for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=i)

    tree.fit(X_train, y_train)
    curr_in_sample_acc = tree.score(X_train, y_train)
    curr_out_sample_acc = tree.score(X_test, y_test)
    in_sample_acc.append(curr_in_sample_acc)
    out_sample_acc.append(curr_out_sample_acc)
    # print("In sample accuracy is: ", curr_in_sample_acc)
    # print("Out sample accuracy is: ", curr_out_sample_acc)

in_sample_mean = np.mean(in_sample_acc)
out_sample_mean = np.mean(out_sample_acc)
in_sample_std = np.std(in_sample_acc)
out_sample_std = np.std(out_sample_acc)
# print(in_sample_mean, out_sample_mean, in_sample_std, out_sample_std)

tree = DecisionTreeClassifier()
cvs = cross_val_score(tree, X, y, scoring="accuracy", cv=10)
cvs_mean = np.mean(cvs)
cvs_std = np.std(cvs)
# print(cvs_mean, cvs_std)

in_sample_acc = []
out_sample_acc = []
for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=i, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=i)
    clf = GridSearchCV(tree, param_grid={}, cv=10)
    clf.fit(X_train, y_train)
    curr_in_sample_acc = clf.score(X_train, y_train)
    curr_out_sample_acc = clf.score(X_test, y_test)
    in_sample_acc.append(curr_in_sample_acc)
    out_sample_acc.append(curr_out_sample_acc)
    print(curr_out_sample_acc)

in_sample_mean = np.mean(in_sample_acc)
out_sample_mean = np.mean(out_sample_acc)
in_sample_std = np.std(in_sample_acc)
out_sample_std = np.std(out_sample_acc)
print(in_sample_mean, out_sample_mean, in_sample_std, out_sample_std)

# cvs = cross_val_score(tree, X, y, scoring="accuracy", cv=10)
# print( "Accuracy is: ", metrics.accuracy_score(y, y_test_pred))
# print(tree.score(X_test, y_test))