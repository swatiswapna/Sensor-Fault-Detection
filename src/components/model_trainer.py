from src.logger import logging 
from src.exception import CustomException
import os 
import sys 
from dataclasses import dataclass
from utils import save_object
import pandas as pd 
import numpy as np 

from utils import evaluate_model
# Ml models 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
@dataclass
class Modeltrainerconfig:
    training_model_file_path = os.path.join('artifacts','model.pkl')

class Model_trainer:
    def __init__(self):
        self.model_trainer_config = Modeltrainerconfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("model training intitated")
            X_train = train_arr[:,:-1]
            X_test = test_arr[:,:-1]
            y_train =train_arr[:,-1]
            y_test = test_arr[:,-1]
            
            model = {
               'SVC': SVC(),
               'RVC':RandomForestClassifier()
            }
            model_report : dict = evaluate_model(X_train,y_train,X_test,y_test,model)

            logging.info(f'Model report : {model_report}')

            if model_report is not None:
                best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = model[best_model_name]

            logging.info(f'best model : {best_model_name},R2 score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.training_model_file_path,
                obj = best_model
            )
        except Exception as e:
            logging.info("Error occured in prediction")
            raise CustomException(e,sys)
            

    



