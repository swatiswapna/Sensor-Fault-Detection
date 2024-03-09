import os 
import sys 
sys.path.append('/Users/swatea/Downloads/Sensor_detection/src')
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd 
from utils import MainUtils 
from flask import request
import pickle 
from dataclasses import dataclass




@dataclass
class predictionpipelineconfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    model_file_path: str = os.path.join('artifacts', "model.pkl" )
    preprocessor_path: str = os.path.join('artifacts', "preprocesssor.pkl")
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)


class predictionpipeline:
    def __init__(self,request: request):
        self.request = request
        self.prediction_pipeline_config  = predictionpipelineconfig()
        self.utils = MainUtils()

    def save_input_files(self)->str:
        try:
            
            prediction_file_input_dir = 'prediction_artifacts'
            os.makedirs(prediction_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(prediction_file_input_dir, input_csv_file.filename)
            input_csv_file.save(pred_file_path)

            return pred_file_path
        except Exception as e:
            logging.info("Error in prediction_pipeline_main")
            raise CustomException(e,sys)
    
    def predict(self,features):

        try:
           
            model = load_object(file_path = self.prediction_pipeline_config.model_file_path)
            preprocessor = load_object(file_path=self.prediction_pipeline_config.preprocessor_path)

            X_transfrom = preprocessor.transform(features)
            y_pred = model.predict(X_transfrom)

            return y_pred
        
        except Exception as e:
            logging.info("Error occur in prediction pipeline predicts")
            raise CustomException(e,sys)
        
    def get_prediction_dataframe(self, input_dataframe_path:pd.DataFrame):
        try:
            y_col_name : str= "quality"
            input_df :pd.DataFrame = pd.read_csv(input_dataframe_path)

            input_df =  input_df.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_df.columns else input_df

            predictions = self.predict(input_df)
            print(predictions)
            input_df[y_col_name] = [pred for pred in predictions]
            target_column_mapping = {-1 :'bad', 1:'good'}
            input_df[y_col_name] = input_df[y_col_name].map(target_column_mapping)

            os.makedirs( self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
            input_df.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
            logging.info("predictions completed. ")
        except Exception as e:
            logging.info("Error occur in prediction pipeline get prediction dataframe")
            raise CustomException(e,sys)
        
    def run_pipeline(self):
          try:
              input_csv_path = self.save_input_files()
              self.get_prediction_dataframe(input_csv_path)

              return self.prediction_pipeline_config


          except Exception as e:
             raise CustomException(e,sys)  


        

        

        
    
        


    







