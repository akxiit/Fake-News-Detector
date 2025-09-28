import sys
import os
from dataclasses import dataclass


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Model Trainer initiated')


            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

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
        
            logging.info('Model are initialized')
            logging.info('Evaluting Models')


            model_report: dict = evaluate_models(
                X_train= X_train,
                y_train= y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f'Best model found: {best_model_name} with score {best_model_score}')


            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            
            accuracy = accuracy_score(y_test, predicted)

            return accuracy
        
        except Exception as e:
            
            raise CustomException(e, sys)

            
            
        
        

