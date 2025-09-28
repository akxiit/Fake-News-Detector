# 📰 Fake News Detector

A simple machine learning project to detect fake news using AI. This project uses TF-IDF vectorization and Logistic Regression to classify news articles as real or fake.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

### 3. Run the App
```bash
streamlit run app.py
```

## 📁 Project Structure
```
├── app.py              # Streamlit web interface
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── artifacts/         # Data and model files
│   ├── true.csv       # Real news dataset
│   ├── fake.csv       # Fake news dataset
│   ├── model.pkl      # Trained model (created after training)
│   └── vectorizer.pkl # TF-IDF vectorizer (created after training)
└── venv/              # Virtual environment (optional)
```

## 🤖 How It Works

1. **Data Loading**: Combines real and fake news datasets
2. **Preprocessing**: Cleans and prepares text data
3. **Feature Extraction**: Uses TF-IDF to convert text to numerical features
4. **Model Training**: Trains a Logistic Regression classifier
5. **Prediction**: Analyzes new text and predicts if it's real or fake

## 📊 Features

- **Simple Interface**: Easy-to-use web interface
- **Fast Training**: Quick model training process
- **Accurate Predictions**: High accuracy on test data
- **Confidence Scores**: Shows prediction confidence
- **Example Testing**: Built-in examples to test

## 🎯 Usage

1. Start by training the model with `python train.py`
2. Launch the web app with `streamlit run app.py`
3. Enter any news text in the interface
4. Get instant predictions with confidence scores

## 📈 Model Performance

The model typically achieves:
- **Accuracy**: ~99%
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF (10,000 features)
- **Training Data**: Combined real and fake news articles

## 🛠️ Requirements

- Python 3.7+
- streamlit
- pandas
- scikit-learn
- numpy
- nltk

## 📝 Notes

- The model is trained on a specific dataset and may not generalize to all types of news
- Always verify important news from multiple reliable sources
- This is an educational project demonstrating ML concepts

## 🔧 Troubleshooting

**Model not found error?**
- Make sure to run `python train.py` first

**Import errors?**
- Check that all requirements are installed: `pip install -r requirements.txt`

**Training taking too long?**
- The training should complete in a few minutes on most systems
