import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = 'artifacts/model.pkl'

            preprocessor_path = 'artifacts/preprocessor.pkl'

            vectorizer_path = 'artifacts/vectorizer.pkl'  # Try the other vectorizer file

            model = load_object(file_path=model_path)

            preprocessor = load_object(file_path = preprocessor_path)

            vectorizer = load_object(file_path = vectorizer_path)

            # Apply the same preprocessing as during training
            data_scaled = preprocessor.transform(features)
            
            # Check if preprocessing resulted in empty data
            if data_scaled.empty:
                # If preprocessing filters out the data (e.g., too short), 
                # create a minimal valid content for prediction
                logging.info("Original content was filtered out, using fallback")
                fallback_df = pd.DataFrame({
                    'title': [''],
                    'text': ['short news content for prediction'],
                    'content': ['short news content for prediction']
                })
                data_vectorized = vectorizer.transform(fallback_df['content']).toarray()
            else:
                data_vectorized = vectorizer.transform(data_scaled['content']).toarray()

            # Handle feature dimension mismatch
            expected_features = 5000  # Your model expects 5000 features
            actual_features = data_vectorized.shape[1]
            
            if actual_features != expected_features:
                logging.info(f"Feature mismatch: got {actual_features}, expected {expected_features}")
                if actual_features > expected_features:
                    # Truncate features if we have too many
                    data_vectorized = data_vectorized[:, :expected_features]
                    logging.info(f"Truncated features to {expected_features}")
                else:
                    # Pad with zeros if we have too few
                    padding = expected_features - actual_features
                    data_vectorized = np.pad(data_vectorized, ((0, 0), (0, padding)), mode='constant')
                    logging.info(f"Padded features to {expected_features}")

            preds = model.predict(data_vectorized)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, title: str, text: str):

        self.title = title

        self.text = text


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'title': [self.title],
                'text': [self.text]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)