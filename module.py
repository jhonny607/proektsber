import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



class TrainModel:
    def __init__(self,X,Y):
        self.Y = Y
        self.X = X
    def train(self):
        X_train, X_test, Y_train,Y_test = train_test_split(self.X,self.Y,test_size=0.2,random_state=42)
        model = GradientBoostingRegressor(n_estimators = 100,random_state=42)
        model.fit(X_train, Y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        rmse_train= np.sqrt(mean_squared_error(Y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(Y_test,y_pred_test))
        print(f'rmse_train:{ rmse_train}')
        print(f'rmse_test:{ rmse_test}')