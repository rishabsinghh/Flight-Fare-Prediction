from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import mean_squared_error,mean_absolute_error,r2_score
import xgboost as xg
import pandas as pd
import numpy as np
class Model_Finder:
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestRegressor()
        self.xgr = xg.XGBRegressor()

    def get_best_params_for_random_forest(self,train_x,train_y):
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'sqrt'],"min_samples_split":[2,5,10,15,100],"min_samples_leaf":[1,2,5,10]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.min_samples_leaf=self.grid.best_params_['min_samples_leaf']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_split=self.min_samples_split,min_samples_leaf=self.min_samples_leaf,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()
    def get_best_params_for_XG(self, train_x, train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_XG Regressor method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xg = {
                'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
                'max_depth' : [3,4,5,6,8,10,12,15],
                'min_child_weight':[1,3,5,7],
                'gamma':[0.0,0.1,0.2,0.3,0.4],
                'colsample_bytree':[0.3,0.4,0.5,0.7]
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.xgr, self.param_grid_xg, verbose=3,
                                     cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.min_child_weight = self.grid.best_params_['min_child_weight']
            self.gamma = self.grid.best_params_['gamma']
            self.colsample_bytree=self.grid.best_params_['colsample_bytree']

            # creating a new model with the best parameters
            self.xgr = xg.XGBRegressor(objective ='reg:linear',learning_rate=self.learning_rate, max_depth=self.max_depth, min_child_weight=self.min_child_weight,gamma=self.gamma,colsample_bytree=self.colsample_bytree)
            # training the mew model
            self.xgr.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGR best params: ' + str(
                                       self.grid.best_params_) + '. Exited the XGR method of the Model_Finder class')
            return self.xgr
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in XGR method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGR Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def get_best_model(self,train_x,train_y,test_x,test_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for KNN
        try:
            self.xgr=self.get_best_params_for_XG(train_x, train_y) 
            #self.xgr=xg.XGBRegressor()
            #self.xgr.fit(train_x,train_y)
            self.prediction_xg= self.xgr.predict(test_x) # Predictions using the KNN Model
            self.xg_score = r2_score(test_y, self.prediction_xg)
            self.logger_object.log(self.file_object, 'r2_score for xg:' + str(self.xg_score))  # Log AUC

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            #self.random_forest=RandomForestRegressor()
            #self.random_forest.fit(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm
            self.random_forest_score=r2_score(test_y,self.prediction_random_forest)
            self.logger_object.log(self.file_object, 'r2_score for RF:' + str(self.random_forest_score))

            #comparing the two models
            if(self.random_forest_score <  self.xg_score):
                return 'XG',self.xgr
            else:
                return 'RandomForest',self.random_forest

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

