from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        # Get the news text from the form
        news_text = request.form.get('news_text', '')
        
        # For this form, we'll put the entire text in the 'text' field and leave title empty
        data = CustomData(
            title='',  # Empty title since we only have one text field
            text=news_text
        )

        pred_df = data.get_data_as_data_frame()

        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)
        
        # Format the results for the template
        prediction_value = results[0]
        if prediction_value == 1:
            result = {
                'prediction': 'Real News',
                'status': '✅'
            }
        else:
            result = {
                'prediction': 'Fake News', 
                'status': '❌'
            }

        return render_template('home.html', result=result)


if __name__  == '__main__':
    app.run(host='0.0.0.0', debug=True)