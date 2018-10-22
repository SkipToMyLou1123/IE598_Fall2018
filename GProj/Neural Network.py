if __name__ == '__main__':
    import os
    os.chdir("C:/Users/Shawn/Desktop/Study Files/Working/598_group_project")
    
    import pandas as pd
    import numpy as np
    
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score as rs
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GridSearchCV
    
    import keras
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.wrappers.scikit_learn import KerasRegressor
    
    data = pd.read_csv('MLF_GP2_EconCycle.csv')
    
    dataraw = data.values
    X = dataraw[:,1:13] 
    y3 = dataraw[:,14]
    y6 = dataraw[:,15]
    y9 = dataraw[:,16]
    
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size = 0.2, random_state = 42)
    X_train6, X_test6, y_train6, y_test6 = train_test_split(X, y6, test_size = 0.2, random_state = 42)
    X_train9, X_test9, y_train9, y_test9 = train_test_split(X, y9, test_size = 0.2, random_state = 42)
    
    print("###########by using default regressor##############")
    model = MLPRegressor()
    model.fit(X_train3,y_train3)
    print("score for 3 month in sample: ",model.score(X_train3,y_train3))
    print("score for 3 month out sample: ",model.score(X_test3,y_test3))
    model.fit(X_train6,y_train6)
    print("score for 6 month in sample: ",model.score(X_train3,y_train3))
    print("score for 6 month out sample: ",model.score(X_test6,y_test6))
    model.fit(X_train9,y_train9)
    print("score for 9 month in sample: ",model.score(X_train3,y_train3))
    print("score for 9 month out sample: ",model.score(X_test9,y_test9))
    print("####################################################")
    
    data = np.delete(data.values,0,axis=1)
    data = np.delete(data,6,axis=1)
    data = np.delete(data,6,axis=1)
    data = np.delete(data,6,axis=1)
    
    X = data[:,0:9]  
    y3 = data[:,10]
    y6 = data[:,11]
    y9 = data[:,12]
    
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size = 0.2, random_state = 42)
    X_train6, X_test6, y_train6, y_test6 = train_test_split(X, y6, test_size = 0.2, random_state = 42)
    X_train9, X_test9, y_train9, y_test9 = train_test_split(X, y9, test_size = 0.2, random_state = 42)
    
    print("########################After PCA####################")
    
    model = MLPRegressor()
    model.fit(X_train3,y_train3)
    print("score for 3 month in sample: ",model.score(X_train3,y_train3))
    print("score for 3 month out sample: ",model.score(X_test3,y_test3))
    model.fit(X_train6,y_train6)
    print("score for 6 month in sample: ",model.score(X_train3,y_train3))
    print("score for 6 month out sample: ",model.score(X_test6,y_test6))
    model.fit(X_train9,y_train9)
    print("score for 9 month in sample: ",model.score(X_train3,y_train3))
    print("score for 9 month out sample: ",model.score(X_test9,y_test9))
    print("####################################################")
    
    
#    print("By using neural network model in keras:")
# 
#    def model():
#    	model = Sequential()
#    	model.add(Dense(6, input_dim=3, kernel_initializer='normal', activation='relu'))
#    	model.add(Dense(10, kernel_initializer='normal', activation='relu'))
#    	model.add(Dense(16, kernel_initializer='normal', activation='relu'))
#    	model.add(Dense(12, kernel_initializer='normal', activation='relu'))
#    	model.add(Dense(1, kernel_initializer='normal'))
#    	keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
#    	model.compile(loss='mean_squared_error', optimizer='SGD')
#    	return model
#        
#    estimators = []
#    estimators.append(('standardize', StandardScaler()))
#    estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=100, batch_size=5, verbose=0)))
#    pipeline = Pipeline(estimators)
#    
#    print("###############Following Train Scores##################")
#    
#    pipeline.fit(X_train3,y_train3)
#    print("Model has R^2 score for 3-monthes as ", pipeline.score(X_train3,y_train3))
#    
#    pipeline.fit(X_train6,y_train6)
#    print("Model has R^2 score for 3-monthes as ", pipeline.score(X_train6,y_train6))
#    
#    pipeline.fit(X_train9,y_train9)
#    print("Model has R^2 score for 3-monthes as ", pipeline.score(X_train9,y_train9))
#    
#    print("######################################################")
#    
#    print("###############Following Test Scores##################")
#    
#    pipeline.fit(X_train3,y_train3)
#    print("Model has R^2 score for 3-monthes as ", pipeline.score(X_test3,y_test3))
#    
#    pipeline.fit(X_train6,y_train6)
#    print("Model has R^2 score for 3-monthes as ", pipeline.score(X_test6,y_test6))
#    
#    pipeline.fit(X_train9,y_train9)
#    print("Model has R^2 score for 3-monthes as ", pipeline.score(X_test9,y_test9))
#   
    print("####################After Tuning###########################")
    
    model = MLPRegressor()
    
##########################################################################    
    hidden_layer_sizes=[]
    
    for i in range(6,13):
        for j in range(6,13):
            for k in range(6,13):
                for l in range(6,13):
                    hidden_layer_sizes.append((i,j,k,l,))
                    
    model_parameter = {
            'hidden_layer_sizes': hidden_layer_sizes, 
            'activation' : ['relu'],
            'solver' : ['lbfgs'],
            'learning_rate_init' : [0.001],
            'alpha' : [0.001], 
            'learning_rate' : ['adaptive'],
            'early_stopping': [True],
            'validation_fraction': [0.3]
            }
    
    grid_model = GridSearchCV(estimator = model,
                              param_grid = model_parameter,
                              scoring = 'r2',
                              cv =10,
                              verbose = 1,
                              n_jobs = -1
            )
#    
#    grid_model.fit(X_train3, y_train3)
#    model3 = grid_model.best_estimator_
#    
    model3 = MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(6, 6, 9, 12), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.3,
       verbose=False, warm_start=False)
    
    m3 = []
    m4 = []
    for i in range(1000):
        model3.fit(X_train3, y_train3)
        m3.append(model3.score(X_train3,y_train3))
        m4.append(model3.score(X_test3,y_test3))
    print("score for 3-monthes in sample: ", sum(m3)/len(m3))
    print("score for 3-monthes out sample: ", sum(m4)/len(m4))

##########################################################################   
    
    hidden_layer_sizes=[]
    for i in range(9,12):
        for j in range(9,12):
            for k in range(9,12):
                for l in range(9,12):
                    hidden_layer_sizes.append((i,j,k,l,))
                    
    model_parameter = {
            'hidden_layer_sizes': hidden_layer_sizes, 
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs','adam'],
            'learning_rate_init' : [0.01],
            'alpha' : [0.001], 
            'learning_rate' : ['invscaling', 'adaptive'],
            'early_stopping': [True],
            'validation_fraction': [0.1,0.3,0.5,0.7,0.9]
            }
    
    grid_model = GridSearchCV(estimator = model,
                              param_grid = model_parameter,
                              scoring = 'r2',
                              cv = 7,
                              verbose = 1,
                              n_jobs = -1
            )
    MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(9, 10, 11, 10), learning_rate='adaptive',
       learning_rate_init=0.01, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.9,
       verbose=False, warm_start=False)
    
#    grid_model.fit(X_train6, y_train6)
    model6 = MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(9, 10, 11, 10), learning_rate='adaptive',
       learning_rate_init=0.01, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.9,
       verbose=False, warm_start=False)
    m6 = []
    m7 = []
    for i in range(1000):
        model6.fit(X_train6, y_train6)
        m6.append(model3.score(X_train6,y_train6))
        m7.append(model3.score(X_test6,y_test6))
    print("score for 6-monthes in sample: ", sum(m6)/len(m6))
    print("score for 6-monthes out sample: ", sum(m7)/len(m7))

##########################################################################    
    
    hidden_layer_sizes=[]
    for i in range(9,12):
        for j in range(9,12):
            for k in range(9,12):
                for l in range(9,12):
                    hidden_layer_sizes.append((i,j,k,l,))
               
    model_parameter = {
            'hidden_layer_sizes': hidden_layer_sizes, 
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs','adam'],
            'learning_rate_init' : [0.01],
            'alpha' : [0.001], 
            'learning_rate' : ['invscaling', 'adaptive'],
            'early_stopping': [True],
            'validation_fraction': [0.1,0.3,0.5,0.7,0.9]
            }
    
    grid_model = GridSearchCV(estimator = model,
                              param_grid = model_parameter,
                              scoring = 'r2',
                              cv = 7,
                              verbose = 1,
                              n_jobs = -1
            )
    
    MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(11, 10, 10, 10), learning_rate='adaptive',
       learning_rate_init=0.01, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.7,
       verbose=False, warm_start=False)
    
#    grid_model.fit(X_train9, y_train9)
    model9 = MLPRegressor(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(11, 10, 10, 10), learning_rate='adaptive',
       learning_rate_init=0.01, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.7,
       verbose=False, warm_start=False)
    m6 = []
    m7 = []
    for i in range(1000):
        model9.fit(X_train9, y_train9)
        m6.append(model3.score(X_train9,y_train9))
        m7.append(model3.score(X_test9,y_test9))
    print("score for 9-monthes in sample: ", sum(m6)/len(m6))
    print("score for 9-monthes out sample: ", sum(m7)/len(m7))

    print("######################################################")
    
