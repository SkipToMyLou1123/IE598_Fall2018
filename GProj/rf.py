from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor


import math
import pandas as pd

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits



def plot_residuals(name, test_pred, train_pred):
    plt.scatter(train_pred, train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
    plt.scatter(test_pred, test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    plt.xlabel(name + ' Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=0, xmax=100, color='black', lw=2)
    #plt.xlim([20, 100])
    plt.show()

    
def print_r2_mse_score(name, model, model_test_pred, model_train_pred):
    print(name + " score on test: ", model.score(X_test_std, y_test))
    print('Slope: %.3f' % model.coef_[0])
    print("Model Coefficients", model.coef_)
    print('Intercept: %.3f' % model.intercept_)
    print('MSE train: %.3f, test: %.3f' % ( MSE(y_train, model_train_pred), MSE(y_test, model_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, model_train_pred), r2_score(y_test, model_test_pred)))



df_econ_cycle = pd.read_csv('/Users/danaoqueyang/Desktop/MLF_GP2_EconCycle.csv')


#Train Test Split
X, y = df_econ_cycle.iloc[:, 1:12].values, df_econ_cycle.iloc[:, 15].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Scale the input data 
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

pca = PCA(n_components=9)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


print("------------------Random forest(PCA)------------------")
rf = RandomForestRegressor(min_samples_leaf=1, n_estimators=100, max_depth=10, n_jobs=-1)
#print("Cross_Val_Score")
#print(np.mean(cross_val_score(rf, X_test_pca, y_test, cv=5)))


rf.fit(X_train_pca, y_train)
y_test_pred = rf.predict(X_test_pca)
y_train_pred = rf.predict(X_train_pca)
print("rf training score(PCA):", rf.score(X_train_pca, y_train))
print("rf test score(PCA):", rf.score(X_test_pca, y_test))



#print("------------------Adaboost regressor------------------")
##Tried using adaboost, performance varies greatly
#ad = AdaBoostRegressor(base_estimator=rf, n_estimators=10, loss='square')
#ad.fit(X_train_pca, y_train)
#print("adaboost training score:", ad.score(X_train_pca, y_train))
#print("adaboost test score:", ad.score(X_test_pca, y_test))

#
#logreg = LogisticRegression()
#svm = SVC()
#
#gammas = [0.2]
#for gam in gammas:
#    kpca = KernelPCA(n_components=10, kernel='linear', gamma=gam) #Test different gamma values
#    X_train_kpca = kpca.fit_transform(X_train_pca)
#    X_test_kpca = kpca.transform(X_test_pca)
#    
#    print("RF")
#    rf = RandomForestRegressor(min_samples_leaf=1, n_estimators=100, max_depth=10, n_jobs=-1)
#    rf.fit(X_train_kpca, y_train)
#    y_test_pred = rf.predict(X_test_kpca)
#    y_train_pred = rf.predict(X_train_kpca)
#    print(rf.score(X_train_kpca, y_train))
#    print(rf.score(X_test_kpca, y_test))
#    

  
#Hyperparameter Tuning
print("------------------hyperparameter tuning------------------")
params_rf = {'min_samples_leaf': [1], 
             'n_estimators': [100, 50, 75, 90], 
             'max_depth':[10]}
grid = GridSearchCV(rf, params_rf, n_jobs=-1, cv=3)
grid.fit(X_train_std, y_train)
print("best test score:",grid.best_score_, "\nbest hyperparameters:",grid.best_params_)

print("------------------after hyperparameter tuning------------------")
rf_2 = RandomForestRegressor(min_samples_leaf=1, n_estimators=90, max_depth=10, n_jobs=-1)

rf_2.fit(X_train_pca, y_train)
y_test_pred = rf_2.predict(X_test_pca)
y_train_pred = rf_2.predict(X_train_pca)
print("rf training score:", rf_2.score(X_train_pca, y_train))
