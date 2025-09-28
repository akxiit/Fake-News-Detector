import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import save_obj

from sklearn.feature_extraction.text import TfidfVectorizer

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    tfidf_vectorizer_file_path: str = os.path.join('artifacts', 'vectorizer.pkl')  # Changed to match prediction pipeline


class TextPreprocessor:

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, df):

        logging.info('Starting text preprocessing')

        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')


        #combine title and text for better features
        df['content'] = df['title'] + ' ' + df['text']


        #remove empty rows
        df = df[df['content'].str.strip() != '']


        #Basic text cleaning
        df['content'] = df['content'].str.lower()
        df['content'] = df['content'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
        df['content'] = df['content'].str.replace(r'\s+', ' ', regex=True)
        df['content'] = df['content'].str.strip()

        #Remove very short articles
        df = df[df['content'].str.len() > 50]

        logging.info('Text preprocessing completed')

        return df

class DataTransformation:
    """
        This function is responsible for transforming the data
        It will handle the numerical and categorical columns
    
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.textpreprocessor_obj = TextPreprocessor()
        self.vectorizer_obj = TfidfVectorizer(
                                                max_features=5000,  # Changed back to 5000 to match the trained model
                                                stop_words='english',
                                                lowercase=True,
                                                ngram_range=(1, 2),
                                                min_df=2,  # Ignore terms that appear in less than 2 documents
                                                max_df=0.95  # Ignore terms that appear in more than 95% of documents
                                            )


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            
            target_column_name = 'label'

            logging.info('Split input and target feature')

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Create content column before preprocessing
            input_feature_train_df['content'] = input_feature_train_df['title'].fillna('') + " " + input_feature_train_df['text'].fillna('')
            input_feature_test_df['content'] = input_feature_test_df['title'].fillna('') + " " + input_feature_test_df['text'].fillna('')


            logging.info('Apply preprocessing object on training and testing data')

            ##Now apply text preprocessing

            input_feature_train_processed = self.textpreprocessor_obj.transform(input_feature_train_df)
            input_feature_test_processed = self.textpreprocessor_obj.transform(input_feature_test_df)

            logging.info('Tfidf vectorization')
            train_features = self.vectorizer_obj.fit_transform(input_feature_train_processed['content']).toarray()
            test_features = self.vectorizer_obj.transform(input_feature_test_processed['content']).toarray()

            # Get the indices after preprocessing to align with features
            train_indices = input_feature_train_processed.index
            test_indices = input_feature_test_processed.index
            
            # Get corresponding target labels for the processed data
            target_train_aligned = target_feature_train_df.loc[train_indices].values
            target_test_aligned = target_feature_test_df.loc[test_indices].values

            # Combine features and target labels
            train_arr = np.c_[train_features, target_train_aligned]
            test_arr = np.c_[test_features, target_test_aligned]

            logging.info('Saved preprocessing object')

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = self.textpreprocessor_obj,
            )

            save_obj(
                file_path = self.data_transformation_config.tfidf_vectorizer_file_path,
                obj = self.vectorizer_obj,
            )

            logging.info('Preprocessor pickle file saved')


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            

        except Exception as e:
            raise CustomException(e, sys)

