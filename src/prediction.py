# prediction.py

import joblib
import pandas as pd
import numpy as np


class UserBehaviorPredictor:

    def __init__(
        self,
        model_path="models/user_behavior_model.pkl",
        gender_encoder_path="models/gender_encoder.pkl",
        os_encoder_path="models/os_encoder.pkl",
        device_encoder_path="models/device_encoder.pkl"
    ):

        self.model = joblib.load(model_path)

        self.gender_encoder = joblib.load(
            gender_encoder_path
        )

        self.os_encoder = joblib.load(
            os_encoder_path
        )

        self.device_encoder = joblib.load(
            device_encoder_path
        )

    # -------------------------------------
    # Encode Inputs
    # -------------------------------------

    def encode_inputs(
        self,
        gender,
        operating_system,
        device_model
    ):

        gender_encoded = (
            self.gender_encoder
            .transform([gender])[0]
        )

        os_encoded = (
            self.os_encoder
            .transform([operating_system])[0]
        )

        device_encoded = (
            self.device_encoder
            .transform([device_model])[0]
        )

        return (
            gender_encoded,
            os_encoded,
            device_encoded
        )

    # -------------------------------------
    # Create Feature Vector
    # -------------------------------------

    def prepare_input(
        self,
        device_model,
        operating_system,
        app_usage_time,
        screen_on_time,
        battery_drain,
        apps_installed,
        data_usage,
        age,
        gender
    ):

        (
            gender_encoded,
            os_encoded,
            device_encoded
        ) = self.encode_inputs(
            gender,
            operating_system,
            device_model
        )

        data = pd.DataFrame({

            "Device Model":
                [device_encoded],

            "Operating System":
                [os_encoded],

            "App Usage Time (min/day)":
                [app_usage_time],

            "Screen On Time (hours/day)":
                [screen_on_time],

            "Battery Drain (mAh/day)":
                [battery_drain],

            "Number of Apps Installed":
                [apps_installed],

            "Data Usage (MB/day)":
                [data_usage],

            "Age":
                [age],

            "Gender":
                [gender_encoded]

        })

        return data

    # -------------------------------------
    # Predict Class
    # -------------------------------------

    def predict(
        self,
        device_model,
        operating_system,
        app_usage_time,
        screen_on_time,
        battery_drain,
        apps_installed,
        data_usage,
        age,
        gender
    ):

        input_df = self.prepare_input(
            device_model,
            operating_system,
            app_usage_time,
            screen_on_time,
            battery_drain,
            apps_installed,
            data_usage,
            age,
            gender
        )

        prediction = self.model.predict(
            input_df
        )[0]

        return prediction

    # -------------------------------------
    # Predict Probability
    # -------------------------------------

    def predict_probability(
        self,
        device_model,
        operating_system,
        app_usage_time,
        screen_on_time,
        battery_drain,
        apps_installed,
        data_usage,
        age,
        gender
    ):

        input_df = self.prepare_input(
            device_model,
            operating_system,
            app_usage_time,
            screen_on_time,
            battery_drain,
            apps_installed,
            data_usage,
            age,
            gender
        )

        probabilities = (
            self.model
            .predict_proba(input_df)[0]
        )

        return probabilities

    # -------------------------------------
    # Behavior Meaning
    # -------------------------------------

    def get_behavior_description(
        self,
        behavior_class
    ):

        descriptions = {

            1:
            "Very Low Smartphone Usage",

            2:
            "Low Smartphone Usage",

            3:
            "Moderate Smartphone Usage",

            4:
            "High Smartphone Usage",

            5:
            "Very High Smartphone Usage"

        }

        return descriptions.get(
            behavior_class,
            "Unknown"
        )

    # -------------------------------------
    # Complete Prediction
    # -------------------------------------

    def full_prediction(
        self,
        device_model,
        operating_system,
        app_usage_time,
        screen_on_time,
        battery_drain,
        apps_installed,
        data_usage,
        age,
        gender
    ):

        prediction = self.predict(
            device_model,
            operating_system,
            app_usage_time,
            screen_on_time,
            battery_drain,
            apps_installed,
            data_usage,
            age,
            gender
        )

        probability = self.predict_probability(
            device_model,
            operating_system,
            app_usage_time,
            screen_on_time,
            battery_drain,
            apps_installed,
            data_usage,
            age,
            gender
        )

        description = (
            self.get_behavior_description(
                prediction
            )
        )

        return {

            "predicted_class":
                int(prediction),

            "description":
                description,

            "confidence":
                round(
                    max(probability) * 100,
                    2
                ),

            "probabilities":
                probability.tolist()

        }


# -------------------------------------
# Testing
# -------------------------------------

if __name__ == "__main__":

    predictor = UserBehaviorPredictor()

    result = predictor.full_prediction(

        device_model="Google Pixel 5",

        operating_system="Android",

        app_usage_time=350,

        screen_on_time=6.5,

        battery_drain=1800,

        apps_installed=70,

        data_usage=1200,

        age=25,

        gender="Male"
    )

    print(result)
