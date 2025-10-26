import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

class TrainModel:
    def __init__(self,X,Y):
        self.Y = Y
        self.X = X
    def train(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_gb.fit(X_train, Y_train)
        y_pred_train_gb = model_gb.predict(X_train)
        y_pred_test_gb = model_gb.predict(X_test)
        rmse_train_gb = np.sqrt(mean_squared_error(Y_train, y_pred_train_gb))
        rmse_test_gb = np.sqrt(mean_squared_error(Y_test, y_pred_test_gb))
        print('GradientBoosting')
        print(f'rmse_train: {rmse_train_gb}')
        print(f'rmse_test: {rmse_test_gb}')

        model_lr = LinearRegression()
        model_lr.fit(X_train, Y_train)
        y_pred_train_lr = model_lr.predict(X_train)
        y_pred_test_lr = model_lr.predict(X_test)
        rmse_train_lr = np.sqrt(mean_squared_error(Y_train, y_pred_train_lr))
        rmse_test_lr = np.sqrt(mean_squared_error(Y_test, y_pred_test_lr))
        print('LinearRegression')
        print(f'rmse_train: {rmse_train_lr}')
        print(f'rmse_test: {rmse_test_lr}')

        model_lasso = Lasso(alpha=0.1, random_state=42)
        model_lasso.fit(X_train, Y_train)
        y_pred_train_lasso = model_lasso.predict(X_train)
        y_pred_test_lasso = model_lasso.predict(X_test)
        rmse_train_lasso = np.sqrt(mean_squared_error(Y_train, y_pred_train_lasso))
        rmse_test_lasso = np.sqrt(mean_squared_error(Y_test, y_pred_test_lasso))

        print('Lasso')
        print(f'rmse_train: {rmse_train_lasso}')
        print(f'rmse_test: {rmse_test_lasso}')
