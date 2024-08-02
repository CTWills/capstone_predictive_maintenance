import pandas as pd
from src.pipeline import pipeline

if __name__ == '__main__':
    initial_data = pd.read_csv("data/ai4i2020.csv")
    X, y = pipeline(initial_data)
    print(X.head())
    print(y.head())
    