# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 22:14:53 2018

@author: Fwh_FrozenFire
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score

df_EC =pd.read_csv('/Users/danaoqueyang/Desktop/MLF_GP2_EconCycle.csv')

#print(df_EC.head())
df_EC.columns = ['Date','T1Y Index','T2Y Index','T3Y Index','T5Y Index','T7Y Index',
               'T10Y Index','CP1M','CP3M','CP6M','CP1M_T1Y','CP3M_T1Y','CP6M_T1Y',
               'USPHCI','PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD']
y1=df_EC['PCT 3MO FWD']
y2=df_EC['PCT 6MO FWD']
y3=df_EC['PCT 9MO FWD']
X1=df_EC.drop(['Date','USPHCI','PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD'],axis = 1)
#Normalize the data to 0-10
X= ((X1-X1.min())/(X1.max()-X1.min()))*10
#Hold out 10% data for out-of-sample test
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1,test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2,test_size=0.2, random_state=42)
X3_train, X3_test, y3_train, y3_test = train_test_split(X, y3,test_size=0.2, random_state=42)
#print(X.info())

######################## GradientBoostingRegressor########################
print('======================GradientBoostingRegressor================')
from sklearn.ensemble import GradientBoostingRegressor

sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=160,                                
            random_state=2)
sgbr.fit(X1_train,y1_train)
print("y1_train score:",sgbr.score(X1_train,y1_train))
print("y1_test score:",sgbr.score(X1_test,y1_test))

importances = pd.Series(data=sgbr.feature_importances_, index= X3_train.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightblue')
plt.title('Features Importances for y1')
plt.savefig('Features Importances for y1.pdf')
plt.show()

sgbr.fit(X2_train,y2_train)
print("y2_train score:",sgbr.score(X2_train,y2_train))
print("y2_test score:",sgbr.score(X2_test,y2_test))

importances = pd.Series(data=sgbr.feature_importances_, index= X3_train.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightblue')
plt.title('Features Importances for y2')
plt.savefig('Features Importances for y2.pdf')
plt.show()

sgbr.fit(X3_train,y3_train)
print("y3_train score:",sgbr.score(X3_train,y3_train))
print("y3_test score:",sgbr.score(X3_test,y3_test))

importances = pd.Series(data=sgbr.feature_importances_, index= X3_train.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightblue')
plt.title('Features Importances for y3')
plt.savefig('Features Importances for y3.pdf')
plt.show()

########################PCA########################
print('======================PCA + GBR===============================')
from sklearn.decomposition import PCA

sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=160,                                
            random_state=2)

pca =  PCA(n_components = 9)
X1_train_pca = pca.fit_transform(X1_train)
X1_test_pca = pca.transform(X1_test)

sgbr.fit(X1_train_pca,y1_train)
print("y1_train score:",sgbr.score(X1_train_pca,y1_train))
print("y1_test score:",sgbr.score(X1_test_pca,y1_test))

pca =  PCA(n_components = 9)
X2_train_pca = pca.fit_transform(X2_train)
X2_test_pca = pca.transform(X2_test)

sgbr.fit(X2_train_pca,y2_train)
print("y2_train score:",sgbr.score(X2_train_pca,y2_train))
print("y2_test score:",sgbr.score(X2_test_pca,y2_test))

pca =  PCA(n_components = 9)
X3_train_pca = pca.fit_transform(X3_train)
X3_test_pca = pca.transform(X3_test)

sgbr.fit(X3_train_pca,y3_train)
print("y3_train score:",sgbr.score(X3_train_pca,y3_train))
print("y3_test score:",sgbr.score(X3_test_pca,y3_test))
print("Notice that n_components = 9 generates the best score, which means 3 of the features are dropped")

########################kPCA########################
print('======================kPCA + GBR===============================')
from sklearn.decomposition import KernelPCA

sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=160,                                
            random_state=2)

kpca =  KernelPCA(n_components = 2, kernel='poly',gamma=0.5)
X1_train_kpca = kpca.fit_transform(X1_train,y1_train)
X1_test_kpca = kpca.transform(X1_test)

sgbr.fit(X1_train_kpca,y1_train)
print("y1_train score:",sgbr.score(X1_train_kpca,y1_train))
print("y1_test score:",sgbr.score(X1_test_kpca,y1_test))

kpca =  KernelPCA(n_components = 2, kernel='poly',gamma=0.5)
X2_train_kpca = kpca.fit_transform(X2_train,y2_train)
X2_test_kpca = kpca.transform(X2_test)

sgbr.fit(X2_train_kpca,y2_train)
print("y2_train score:",sgbr.score(X2_train_kpca,y2_train))
print("y2_test score:",sgbr.score(X2_test_kpca,y2_test))

kpca =  KernelPCA(n_components = 2, kernel='poly',gamma=0.5)
X3_train_kpca = kpca.fit_transform(X3_train,y3_train)
X3_test_kpca = kpca.transform(X3_test)

sgbr.fit(X3_train_kpca,y3_train)
print("y3_train score:",sgbr.score(X3_train_kpca,y3_train))
print("y3_test score:",sgbr.score(X3_test_kpca,y3_test))
print('======================PCA is better============================')
########################GridSearch########################
print('======================GridSearch on PCA + GBR==================')
from sklearn.model_selection import GridSearchCV
####GridSearch here, too slow to run, so commented it
#sgbr = GradientBoostingRegressor(random_state=2)
#params_sgbr = {
#    'n_estimators' :[100,160,200],
#    'min_samples_leaf':[1,2,5],
#    'max_depth':[None,4,5,6],
#    'subsample':[0.95,0.85,0.8,0.9],
#    'max_features':["auto",0.75,0.8,0.7]
#}
#grid_sgbr = GridSearchCV(estimator=sgbr,
#                       param_grid=params_sgbr,
#                       scoring='r2',
#                       cv=5,
#                       verbose=1,
#                       n_jobs=1)
#grid_sgbr.fit(X1_train_pca, y1_train)
#best_model = grid_sgbr.best_estimator_
#print(best_model)

#Results
sgbr1 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=6, max_features=0.9,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=5,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=160, presort='auto', random_state=2,
             subsample=0.8, verbose=0, warm_start=False)

sgbr2 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=4, max_features=0.9,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=4,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=2,
             subsample=0.95, verbose=0, warm_start=False)

sgbr3 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=None, max_features=0.75,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=2,
             subsample=0.85, verbose=0, warm_start=False)

sgbr1.fit(X1_train_pca,y1_train)
print("y1_train score:",sgbr1.score(X1_train_pca,y1_train))
print("y1_test score:",sgbr1.score(X1_test_pca,y1_test))

sgbr2.fit(X2_train_pca,y2_train)
print("y2_train score:",sgbr2.score(X2_train_pca,y2_train))
print("y2_test score:",sgbr2.score(X2_test_pca,y2_test))

sgbr3.fit(X3_train_pca,y3_train)
print("y3_train score:",sgbr3.score(X3_train_pca,y3_train))
print("y3_test score:",sgbr3.score(X2_test_pca,y3_test))
