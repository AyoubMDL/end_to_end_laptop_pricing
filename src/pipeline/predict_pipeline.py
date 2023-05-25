import sys
import pandas as pd
from src.exception import CustomException
import os

from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor", "preprocessor.pkl") 
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_preprocessed = preprocessor.transform(features)
            prediction = model.predict(data_preprocessed)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 brand: str,
                 processor_brand: str,
                 processor_name: str,
                 processor_gnrtn: str,
                 ram_gb: str,
                 ram_type: str,
                 ssd: str,
                 hdd: str,
                 os: str,
                 os_bit: str,
                 graphic_card_gb: str,
                 weight: str,
                 warranty: str,
                 touchscreen: str,
                 msoffice: str,
                 rating: str):
        
        self.brand = brand
        self.processor_brand = processor_brand
        self.processor_name = processor_name
        self.processor_gnrtn = processor_gnrtn
        self.ram_gb = ram_gb
        self.ram_type = ram_type
        self.ssd = ssd
        self.hdd = hdd
        self.os = os
        self.os_bit = os_bit
        self.graphic_card_gb = graphic_card_gb
        self.weight = weight
        self.warranty = warranty
        self.touchscreen = touchscreen
        self.msoffice = msoffice
        self.rating = rating
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "brand": [self.brand],
                "processor_brand": [self.processor_brand],
                "processor_name": [self.processor_name],
                "processor_gnrtn": [self.processor_gnrtn],
                "ram_gb": [self.ram_gb],
                "ram_type": [self.ram_type],
                "ssd": [self.ssd],
                "hdd": [self.hdd],
                "os": [self.os],
                "os_bit": [self.os_bit],
                "graphic_card_gb": [self.graphic_card_gb],
                "weight": [self.weight],
                "warranty": [self.warranty],
                "Touchscreen": [self.touchscreen],
                "msoffice": [self.msoffice],
                "rating": [self.rating],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
