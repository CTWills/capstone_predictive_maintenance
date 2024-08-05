from joblib import load
import pandas as pd

if __name__ == '__main__':
    with open("models/models.joblib", "rb") as f:
        models = load(f)


    sensor_data = pd.DataFrame({
        "Air temperature [K]": [298.2],
        "Process temperature [K]": [308.4],
        "Rotational speed [rpm]": [1310],
        "Torque [Nm]": [61.0],
        "Tool wear [min]": [189],
    })

    for model_name, model in models.items():
        print(model_name, model.predict(sensor_data))
        print(model_name, model.predict_proba(sensor_data))