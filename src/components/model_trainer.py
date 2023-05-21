import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso' : Lasso(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                'xgboost' : XGBRegressor(),
                'Lightgbm' : LGBMRegressor(),
                'Gradient boosting' : GradientBoostingRegressor(),
                'support vector': SVR(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            param_grid = {
                'Linear Regression': {},
                'Ridge': {'alpha': [0.1, 1.0, 10.0]},
                'Lasso': {'alpha': [0.1, 1.0, 10.0]},
                'K-Neighbors Regressor': {'n_neighbors': [3, 5, 7]},
                'Decision Tree': {'max_depth': [None, 5, 10]},
                'Random Forest Regressor': {'n_estimators': [100, 200, 300]},
                'xgboost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.01]},
                'Lightgbm': {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.01]},
                'Gradient boosting': {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.01]},
                'support vector': {'C': [1.0, 10.0], 'kernel': ['linear', 'rbf']},
                'AdaBoost Regressor': {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1.0]}
            }
   

            model_report: dict = evaluate_models(X_train=X_train, 
                                                y_train=y_train, 
                                                X_test=X_test, 
                                                y_test=y_test, 
                                                models=models,
                                                params=param_grid
                                                )
            logging.info(f"Model Report : {model_report}")

            best_model_name = max(model_report, key=lambda x: model_report[x]['r2_score'])
            best_r2_score = model_report[best_model_name]['r2_score']
            best_model = models[best_model_name]

            if best_r2_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2_score : {best_r2_score}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            model_r2_score = r2_score(y_test, predicted)

            return model_r2_score

        except Exception as e:
            raise CustomException(e, sys)