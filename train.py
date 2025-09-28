import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and combine the datasets"""
    print("Loading datasets...")
    
    # Load true and fake news datasets
    true_df = pd.read_csv('artifacts/true.csv', low_memory=False)
    fake_df = pd.read_csv('artifacts/fake.csv', low_memory=False)
    
    # Add labels
    true_df['label'] = 1  # True news
    fake_df['label'] = 0  # Fake news
    
    # Combine datasets
    df = pd.concat([true_df, fake_df], ignore_index=True)
    
    print(f"Total samples: {len(df)}")
    print(f"True news: {len(true_df)}")
    print(f"Fake news: {len(fake_df)}")
    
    return df

def preprocess_data(df):
    """Preprocess the text data with better cleaning"""
    print("Preprocessing data...")
    
    # Fill missing values
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # Combine title and text for better features
    df['content'] = df['title'] + ' ' + df['text']
    
    # Remove rows with empty content
    df = df[df['content'].str.strip() != '']
    
    # Basic text cleaning
    df['content'] = df['content'].str.lower()
    df['content'] = df['content'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
    df['content'] = df['content'].str.replace(r'\s+', ' ', regex=True)
    df['content'] = df['content'].str.strip()
    
    # Remove very short articles (likely to be noise)
    df = df[df['content'].str.len() > 50]
    
    print(f"After preprocessing: {len(df)} samples")
    
    return df['content'], df['label']

def evaluate_models(X, y):
    """Evaluate multiple models and find the best one"""
    print("Evaluating multiple models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF vectorizer with reduced features to prevent overfitting
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Reduced from 10000
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Define models with regularization to prevent overfitting
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0,  # Regularization parameter
            solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'SVM': SVC(
            kernel='linear',
            C=1.0,  # Regularization parameter
            random_state=42,
            probability=True
        )
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_tfidf, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
        
        # Test predictions
        y_pred = model.predict(X_test_tfidf)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'train_accuracy': model.score(X_train_tfidf, y_train)
        }
        
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Train accuracy: {results[name]['train_accuracy']:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Check for overfitting
        overfitting = results[name]['train_accuracy'] - test_accuracy
        if overfitting > 0.05:
            print(f"⚠️  Potential overfitting detected (difference: {overfitting:.4f})")
        else:
            print("✅ Good generalization")
    
    # Find best model based on cross-validation score
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best model: {best_model_name}")
    print(f"Cross-validation score: {results[best_model_name]['cv_mean']:.4f}")
    print(f"Test accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    
    # Detailed evaluation of best model
    print(f"\nDetailed evaluation of {best_model_name}:")
    y_pred_best = best_model.predict(X_test_tfidf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Fake', 'Real']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)
    
    return best_model, vectorizer, results

def save_model(model, vectorizer, results):
    """Save the trained model, vectorizer, and results"""
    print("\nSaving model and results...")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs('artifacts', exist_ok=True)
    
    # Save model and vectorizer
    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('artifacts/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save model comparison results
    with open('artifacts/model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Model, vectorizer, and results saved successfully!")

def main():
    """Main training function"""
    try:
        # Load data
        df = load_data()
        
        # Preprocess data
        X, y = preprocess_data(df)
        
        # Evaluate multiple models
        best_model, vectorizer, results = evaluate_models(X, y)
        
        # Save model and results
        save_model(best_model, vectorizer, results)
        
        print("\n" + "="*60)
        print("🎉 Training completed successfully!")
        print("="*60)
        print("\n📊 Model Comparison Summary:")
        
        for name, result in results.items():
            overfitting = result['train_accuracy'] - result['test_accuracy']
            status = "⚠️ " if overfitting > 0.05 else "✅"
            print(f"{status} {name}:")
            print(f"   CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
            print(f"   Test Acc: {result['test_accuracy']:.4f}")
            print(f"   Overfitting: {overfitting:.4f}")
            print()
        
        print("Files saved in 'artifacts/' folder:")
        print("- model.pkl (best model)")
        print("- vectorizer.pkl (text preprocessor)")
        print("- model_results.pkl (all model results)")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()
