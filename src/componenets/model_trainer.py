import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def inititate_model_trainer(self, train_array,test_array,preprocessor_path=""):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neigbours Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor()
            }
            param_grids = {
    "Random Forest": {'n_estimators': [100, 200], 'max_depth': [5, 10],'criterion':['squared_error','friedman_mse']},
    
    "Gradient Boosting": {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.01]},
    "Linear Regression": {},  # No hyperparameters to tune for Linear Regression
    "K-Neigbours Regressor": {'n_neighbors': [3, 5, 7]},
    "XGBRegressor": {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.01]},
    "CatBoosting Regressor": {'iterations': [100, 200], 'depth': [4, 6]},
    "AdaBoost Regressor": {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
    "Decision Tree": {'max_depth': [5, 10]}
}


            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,param=param_grids)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best Model Found on both Training and Testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(x_test)
            r2_score_value = r2_score(y_test,predicted)

            return r2_score_value
        
        except Exception as e:
            raise CustomException(e,sys)
