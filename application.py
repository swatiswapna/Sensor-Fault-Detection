from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging 
import os,sys
from src.pipelines.prediction_pipeline import predictionpipeline

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to my application"


@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            # Instantiate prediction pipeline
            pipeline = predictionpipeline(request=request)

            # Run prediction pipeline
            prediction_file_detail = pipeline.run_pipeline()

            logging.info("Prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                             download_name=prediction_file_detail.prediction_file_name,
                             as_attachment=True)
        else:
            return render_template('upload.html')  
    
    except Exception as e:
        logging.exception("An error occurred during prediction.")
        raise CustomException(e, sys)

    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)