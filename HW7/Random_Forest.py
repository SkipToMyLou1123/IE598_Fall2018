__author__ = 'Yijun Lou'
__copyright__ = "Copyright 2018, Homework 7 of IE598"
__credits__ = ["Yijun Lou"]
__license__ = "University of Illinois, Urbana Champaign"
__version__ = "1.0.0"
__maintainer__ = "Yijun Lou"
__email__ = "ylou4@illinois.edu"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                test_size=0.1,
                random_state=1,
                stratify=y)

# rfc = RandomForestClassifier(random_state=1)
# param_grid = [{'n_estimators': [100,200,500,1000]}]
# gs = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
for i in [100, 200, 500, 1000]:
    rfc = RandomForestClassifier(random_state=1, n_estimators=i)
    rfc.fit(X_train, y_train)
    print("In sample accuracy with n_estimators = %d is %f" % (i, rfc.score(X_train, y_train)))
# gs.fit(X_train, y_train)

# print("Best score: ", gs.best_score_)
# print("Best parameters: ", gs.best_params_)

feat_labels = dataset.columns[1:]
rfc = RandomForestClassifier(random_state=1,n_estimators=500)
rfc.fit(X_train,y_train)
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(X_train.shape[1]):
    print("%-*s %f" % (30, feat_labels[indices[i]],importances[indices[i]]))

print("My name is Yijun Lou\n"
              "My NetId is: ylou4\n"
              "I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

