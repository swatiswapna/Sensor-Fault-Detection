import os 
import sys
sys.path.append('/Users/swatea/Downloads/Sensor_detection/src')

from src.logger import logging 
from src.exception import CustomException
from src.components.data_injection import Datainjection
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import Model_trainer




if __name__ == "__main__":
    logging.info("training pipeline started ")
    obj = Datainjection()

    train_data_path , test_data_path = obj.initiate_data_injection()

    print(train_data_path , test_data_path)
    data_transformation_obj = DataTransformation()

    train_arr,test_arr,_ = data_transformation_obj.initiate_data_tranformation(train_data_path,test_data_path)

    # print(train_arr)

    logging.info("tranformation completed")

    model_trainer = Model_trainer()
    model_trainer.initiate_model_training(train_arr,test_arr)


