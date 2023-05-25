import os
import sys
import numpy as np
import pandas as pd
import pickle
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    

def compute_metrics(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            grid_search = GridSearchCV(model, 
                                       param, 
                                       cv=3)
            grid_search.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            
            model_test_mae , model_test_rmse, model_test_r2 = compute_metrics(y_test, y_test_pred)

            report[list(models.keys())[i]] = {
                'mae': model_test_mae,
                'rmse': model_test_rmse,
                'r2_score': model_test_r2
            }
        
        return report


    except Exception as e:
        raise CustomException(e, sys)
