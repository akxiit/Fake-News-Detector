# 📰 AI-Powered Fake News Detector

An advanced machine learning project that detects fake news using AI algorithms. This project features a modular architecture with multiple ML models, a beautiful web interface, and comprehensive data processing pipeline.

## 🌟 Features

- **🤖 Multiple ML Models**: Traditional ML (TF-IDF + Logistic Regression, SVM, Random Forest, Naive Bayes) and Transformer-based approaches
- **🎨 Beautiful Web Interface**: Modern, responsive Flask web application with glassmorphism design
- **⚡ Real-time Prediction**: Instant fake news detection with user-friendly results
- **🔧 Modular Architecture**: Well-structured codebase with separate components for data processing, model training, and prediction
- **📊 Comprehensive Logging**: Detailed logging system for debugging and monitoring
- **🛡️ Error Handling**: Robust custom exception handling system

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/akxiit/Fake-News-Detector.git
cd Fake-News-Detector
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train.py
```

### 5. Run the Web Application
```bash
python app.py
```

The app will be available at `http://localhost:5000`

## 📁 Project Structure

```
Fake-News-Detector/
├── 📄 app.py                          # Flask web application
├── 🎯 train.py                        # Model training script
├── ⚙️ setup.py                        # Package setup configuration
├── 📝 requirements.txt                # Python dependencies
├── 📖 README.md                       # Project documentation
├── 🗂️ artifacts/                      # Generated model files and data
│   ├── 📊 data.csv                    # Processed dataset
│   ├── 🧪 train.csv                   # Training data
│   ├── 🧪 test.csv                    # Testing data  
│   ├── 🤖 model.pkl                   # Trained ML model
│   ├── 🔧 preprocessor.pkl            # Text preprocessor
│   ├── 📐 vectorizer.pkl              # TF-IDF vectorizer
│   └── 📐 tfidf_vectorizer.pkl        # Alternative vectorizer
├── 🎨 templates/                      # HTML templates
│   ├── 🏠 index.html                  # Landing page
│   └── 🔍 home.html                   # News analysis page
├── 📂 src/                            # Source code
│   ├── 🔧 __init__.py                 # Package initializer
│   ├── 🛠️ utils.py                    # Utility functions
│   ├── 📋 logger.py                   # Logging configuration
│   ├── ⚠️ exception.py                # Custom exception handling
│   ├── 🧩 components/                 # ML pipeline components
│   │   ├── 🔧 __init__.py            
│   │   ├── 📥 data_ingestion.py       # Data loading and splitting
│   │   ├── 🔄 data_transformation.py  # Text preprocessing and vectorization
│   │   └── 🏋️ model_trainer.py        # Model training and evaluation
│   └── 🚀 pipeline/                   # Prediction pipelines
│       ├── 🔧 __init__.py            
│       ├── 🔮 predict_pipeline.py     # Traditional ML prediction
│       └── 🤖 transformer_predict_pipeline.py # Transformer-based prediction
├── 📊 logs/                           # Application logs
│   └── 📅 [date_time].log/           # Timestamped log files
└── 📓 notebook/                       # Jupyter notebooks and data
    └── 📊 data/                       # Raw datasets
        ├── 🚫 fake.csv               # Fake news dataset
        └── ✅ true.csv               # Real news dataset
```

## 🛠️ How It Works

### 1. **Data Ingestion** (`src/components/data_ingestion.py`)
- Loads real and fake news datasets
- Combines and processes the data
- Splits into training and testing sets

### 2. **Data Transformation** (`src/components/data_transformation.py`)
- **Text Preprocessing**: Cleaning, lowercasing, removing special characters
- **Feature Engineering**: Combines title and text content
- **TF-IDF Vectorization**: Converts text to numerical features (5000 features)
- **Data Filtering**: Removes very short articles

### 3. **Model Training** (`src/components/model_trainer.py`)
- **Multiple Algorithms**: Logistic Regression, Random Forest, Naive Bayes, SVM
- **Model Evaluation**: Accuracy scoring and comparison
- **Best Model Selection**: Automatically selects the best performing model
- **Model Persistence**: Saves the trained model and components

### 4. **Prediction Pipeline** (`src/pipeline/`)
- **Traditional ML**: Fast prediction using saved TF-IDF + ML model
- **Transformer-based**: Advanced contextual analysis using pre-trained models
- **Error Handling**: Robust prediction with fallback mechanisms

### 5. **Web Interface** (`app.py` + `templates/`)
- **Modern Design**: Glassmorphism UI with gradient backgrounds
- **Model Selection**: Choose between Traditional ML and AI Transformer
- **Real-time Analysis**: Instant results with visual feedback
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🎯 Usage

### Web Interface
1. **Start the application**: `python app.py`
2. **Open your browser**: Navigate to `http://localhost:5000`
3. **Choose detection method**: Traditional ML or AI Transformer
4. **Enter news text**: Paste the article in the text area
5. **Get results**: Instant prediction with status indicator

### Programmatic Usage
```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create prediction pipeline
pipeline = PredictPipeline()

# Prepare your data
data = CustomData(title="News Title", text="News content here...")
df = data.get_data_as_data_frame()

# Make prediction
result = pipeline.predict(df)
print("Real News" if result[0] == 1 else "Fake News")
```

## 📈 Model Performance

### Traditional ML Models
- **Best Model**: Automatically selected based on accuracy
- **Feature Engineering**: TF-IDF with 5000 features
- **Preprocessing**: Advanced text cleaning and filtering
- **Accuracy**: Typically 85-95% on test data

### Available Algorithms
- **Logistic Regression**: Fast and interpretable
- **Random Forest**: Robust ensemble method
- **Naive Bayes**: Good for text classification
- **SVM**: Support Vector Machine with linear kernel

### Transformer Models (Advanced)
- **BART**: Zero-shot classification for contextual understanding
- **Pre-trained**: Leverages large-scale language model knowledge
- **Contextual**: Better understanding of nuanced language patterns

## 🔧 Configuration

### Model Training Parameters
- **TF-IDF Features**: 5000 (configurable in `data_transformation.py`)
- **Text Preprocessing**: Lowercase, special character removal, length filtering
- **Train/Test Split**: Configurable in data ingestion
- **Model Selection**: Automatic based on accuracy

### Web Application
- **Host**: `0.0.0.0` (accessible from network)
- **Port**: `5000` (default Flask port)
- **Debug Mode**: Enabled for development

## 🚨 Troubleshooting

### Common Issues

**❌ "Model not found" error**
```bash
# Solution: Train the model first
python train.py
```

**❌ "Feature dimension mismatch" error**
- The prediction pipeline automatically handles feature mismatches
- Check if the correct vectorizer file is being loaded

**❌ "Import errors"**
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**❌ "Empty prediction results"**
- Ensure the input text is not too short (minimum 10 characters)
- Check if the preprocessing is filtering out the content

## 📋 Requirements

### Python Version
- **Python 3.7+** (Recommended: Python 3.8 or higher)

### Dependencies
```
flask              # Web framework
pandas             # Data manipulation
scikit-learn       # Machine learning algorithms
numpy              # Numerical computing
streamlit          # Alternative UI framework
transformers       # Transformer models (optional)
torch              # PyTorch for transformers (optional)
```

## 🔮 Future Enhancements

- [ ] **Model Ensemble**: Combine multiple models for better accuracy
- [ ] **Real-time Training**: Continuous learning from user feedback
- [ ] **API Integration**: RESTful API for external applications
- [ ] **Batch Processing**: Handle multiple articles simultaneously
- [ ] **Advanced Analytics**: Confidence scores and prediction explanations
- [ ] **Database Integration**: Store predictions and user interactions
- [ ] **Docker Deployment**: Containerized application deployment

## 📊 Datasets

### Training Data Sources
- **Real News**: Legitimate news articles from reliable sources
- **Fake News**: Verified misinformation and fake articles
- **Preprocessing**: Text cleaning, duplicate removal, balanced sampling

### Data Format
```
Columns: ['title', 'text', 'label']
- title: Article headline
- text: Article body content  
- label: 0 (Fake) or 1 (Real)
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**: Follow the existing code style
4. **Add tests**: Ensure your changes work correctly
5. **Commit changes**: `git commit -am 'Add new feature'`
6. **Push to branch**: `git push origin feature-name`
7. **Submit Pull Request**: Describe your changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/akxiit/Fake-News-Detector/issues)
- **Email**: Support for questions and suggestions
- **Documentation**: Check this README for comprehensive guidance

## ⚠️ Disclaimer

This tool is for educational and research purposes. While it provides good accuracy, always verify important news from multiple reliable sources. The model's predictions should not be the sole basis for determining news authenticity.

---

**Built with ❤️ by [akxiit](https://github.com/akxiit)** | **Powered by Machine Learning & AI**
