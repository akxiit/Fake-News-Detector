import streamlit as st
import numpy as np
import time
import os
from datetime import datetime
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom minimal CSS
st.markdown("""
<style>
    body { font-family: 'Poppins', sans-serif; }
    .title {
        text-align: center;
        background: linear-gradient(135deg, #007bff, #00bfa6);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .result {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-weight: 600;
    }
    .real {
        background-color: #00b894;
        color: white;
    }
    .fake {
        background-color: #d63031;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title"><h1>AI Fake News Detector</h1><p>Check whether a news article is Real or Fake</p></div>', unsafe_allow_html=True)

# Text area for news input
news_text = st.text_area("Enter News Article Below:", height=200, placeholder="Paste the article text here...")

# Predict button
if st.button("Analyze"):
    if len(news_text.strip()) < 50:
        st.warning("Please enter at least 50 characters for better prediction.")
    else:
        with st.spinner("Analyzing the article..."):
            time.sleep(2)

            try:
                model_files = ['artifacts/model.pkl', 'artifacts/preprocessor.pkl', 'artifacts/vectorizer.pkl']
                if not all(os.path.exists(f) for f in model_files):
                    st.error("Model files missing. Please train the model first.")
                else:
                    data = CustomData(title='', text=news_text)
                    pred_df = data.get_data_as_data_frame()
                    pipeline = PredictPipeline()
                    result = pipeline.predict(pred_df)

                    
                    prediction = "Real News" if result[0] == 1 else "Fake News"

                    color_class = "real" if prediction == "Real News" else "fake"
                    st.markdown(f'<div class="result {color_class}">{prediction} </div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Developed by Students | Machine Learning Based Fake News Detector")
