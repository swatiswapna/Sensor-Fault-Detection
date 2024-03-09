import sys 
import os 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer  # to combine multiple pipelines 
from sklearn.impute import KNNImputer    # impute nearest mean points 
from sklearn.impute import SimpleImputer  # MEAN IMPUTE 
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTETomek

import pandas as pd
import numpy as np 

from src.utils import save_object


@dataclass
class DataTransformationconfig:
    preprocessor_file_path  = os.path.join('artifacts','preprocesssor.pkl')
    # preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocesssor_obj.pkl')

class DataTransformation:

    def __init__(self):
       self.data_transformation_config = DataTransformationconfig()

    def get_data_tranformation_obj(self):
        try:
            logging.info('pipeline initiated')
            imputer = KNNImputer(n_neighbors=5)
            preprocessor= Pipeline(
                steps = [
                    ('imputer', imputer),('scaler',RobustScaler())
                    
                ]
            )
            logging.info('pipeline completed')
            return preprocessor

        except Exception as e :
            logging.info("Error in data_transformation  object creation")
            raise CustomException(e,sys)
        
    def check_std(self,df :pd.DataFrame):
        drop_col = []
        num_cols = [col for col in df.columns if df[col].dtype != 'O']
        for col in num_cols:
          if df[col].std() == 0:
              drop_col.append(col)
        return drop_col
    
    def check_null(self,df: pd.DataFrame , thresold = .7):
        cols_missing_ratios = df.isna().sum().div(df.shape[0])
        cols_to_drop = list(cols_missing_ratios[cols_missing_ratios > thresold].index)
        return cols_to_drop 
    
    def initiate_data_tranformation(self, train_data_path, test_data_path):
        try:

            train_dataframe = pd.read_csv(train_data_path)
            test_dataframe = pd.read_csv(test_data_path)

            logging.info("data initilised in data tranformation")

            target_col = 'Good/Bad'
            drop_col = 'Unnamed: 0'
            preprocessing_obj = self.get_data_tranformation_obj()

            
            drop_std = self.check_std(train_dataframe)
            drop_null = self.check_null(train_dataframe)

            drop_columns = drop_std
            drop_columns.extend(drop_null)
            drop_columns.append(target_col)
            drop_columns.append(drop_col)

            input_features_train_df  = train_dataframe.drop(columns=drop_columns,axis=1)
            input_features_test_df  =  test_dataframe.drop(columns=drop_columns,axis = 1)

            target_features_train_df = train_dataframe[target_col]
            target_features_test_df = test_dataframe[target_col]

            if isinstance(target_features_test_df, pd.Series):
                target_features_test_df = target_features_test_df.to_frame()
            
            input_features_train_array = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_array = preprocessing_obj.transform(input_features_test_df)

            resampler  = SMOTETomek(sampling_strategy='auto')
            X_resample , y_resample= resampler.fit_resample(input_features_train_array,target_features_train_df)

            
            logging.info("Applying preprocessing in training and testing datasets")

            train_arr = np.c_[X_resample,np.array(y_resample)]
            test_arr = np.c_[input_features_test_array, np.array(target_features_test_df)]


            

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj
            )

            logging.info("processor pickel is created and saved")

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_file_path
            )

        except Exception as e :
            logging.info("Error in data_transformation initiation ")
            raise CustomException(e,sys)


