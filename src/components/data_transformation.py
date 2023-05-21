import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.exchange_rate = 0.012  # 19/05/2023

    def get_data_transformer_object(self):
        """
        Function to handle data transformation
        """
        try:
            categorical_columns = [
                'brand',
                'processor_brand',
                'processor_name',
                'processor_gnrtn',
                'ram_gb',
                'ram_type',
                'ssd',
                'hdd',
                'os',
                'os_bit',
                'graphic_card_gb',
                'weight',
                'warranty',
                'Touchscreen',
                'msoffice',
                'rating'
            ]

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_ojb = self.get_data_transformer_object()

            target_column_name = "Price"
            other_columns_to_drop = ['Number of Ratings', 'Number of Reviews'] 

            input_feature_train_df = train_df.drop(columns=[target_column_name] + other_columns_to_drop, axis=1)
            target_feature_train_df = train_df[target_column_name] * self.exchange_rate

            input_feature_test_df = test_df.drop(columns=[target_column_name] + other_columns_to_drop, axis=1)
            target_feature_test_df = test_df[target_column_name] * self.exchange_rate

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessor_ojb.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_ojb.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr.toarray(), np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr.toarray(), np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_ojb
            )

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)


        except Exception as e:
            raise CustomException(e, sys)