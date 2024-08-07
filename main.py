"""
This file creates and trains all the models necessary for the streamlit app to use
"""
import pandas as pd
from joblib import dump
from src.pipeline import pipeline
from src.model_training import create_model



if __name__ == '__main__':

    # Dictionary of target variables and their respective sampling methods
    models = {
        "Machine failure": "undersample",
        "TWF": "undersample",
        "HDF": "undersample",
        "PWF": "oversample",
        "OSF": "oversample",
    }

    initial_data = pd.read_csv("data/ai4i2020.csv")

    # Create and train models for each target (key) in the models dictionary
    for target, sampling_method in models.items():
        X, y = pipeline(initial_data, target, 
                        sample_strategy=sampling_method)
        trained_model, accuracy, precision, recall, f1 = create_model(X, y)
        print(f"Model completed for {target}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print("*************************************\n\n")
        models[target] = trained_model

    # Save the trained models
    with open("models/models.joblib", "wb") as f:
        dump(models, f, protocol=5)