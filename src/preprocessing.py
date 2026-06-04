# preprocessing.py

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:

    def __init__(self):
        self.gender_encoder = LabelEncoder()
        self.os_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()

    def load_data(self, file_path):
        """
        Load dataset from CSV file
        """
        df = pd.read_csv(file_path)
        return df

    def clean_data(self, df):
        """
        Handle missing values and duplicates
        """

        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values
        for col in df.columns:

            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])

            else:
                df[col] = df[col].fillna(df[col].median())

        return df

    def encode_features(self, df):
        """
        Encode categorical columns
        """

        df["Gender"] = self.gender_encoder.fit_transform(
            df["Gender"]
        )

        df["Operating System"] = self.os_encoder.fit_transform(
            df["Operating System"]
        )

        df["Device Model"] = self.device_encoder.fit_transform(
            df["Device Model"]
        )

        return df

    def save_encoders(self):
        """
        Save encoders for Streamlit deployment
        """

        joblib.dump(
            self.gender_encoder,
            "models/gender_encoder.pkl"
        )

        joblib.dump(
            self.os_encoder,
            "models/os_encoder.pkl"
        )

        joblib.dump(
            self.device_encoder,
            "models/device_encoder.pkl"
        )

    def prepare_training_data(self, df):
        """
        Create X and y
        """

        X = df.drop(
            columns=[
                "User ID",
                "User Behavior Class"
            ]
        )

        y = df["User Behavior Class"]

        return X, y

    def split_data(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42
    ):
        """
        Train Test Split
        """

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    def full_pipeline(self, file_path):
        """
        Complete preprocessing pipeline
        """

        print("Loading Dataset...")
        df = self.load_data(file_path)

        print("Cleaning Dataset...")
        df = self.clean_data(df)

        print("Encoding Features...")
        df = self.encode_features(df)

        print("Saving Encoders...")
        self.save_encoders()

        print("Preparing Features...")
        X, y = self.prepare_training_data(df)

        print("Splitting Dataset...")
        X_train, X_test, y_train, y_test = self.split_data(
            X,
            y
        )

        print("Preprocessing Complete!")

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )


if __name__ == "__main__":

    preprocessor = DataPreprocessor()

    X_train, X_test, y_train, y_test = (
        preprocessor.full_pipeline(
            "data/user_behavior_dataset.csv"
        )
    )

    print("\nTraining Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)
