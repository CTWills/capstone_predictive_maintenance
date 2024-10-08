from joblib import load
import pandas as pd
import streamlit as st

def display_classification(models: dict, sensor_data: pd.DataFrame):
    """
    Classifies the given sensor data using the given models and displays the type of failure that is occuring

    params:
        models: dict - the dictionary of models to be used for classification
        sensor_data: pd.DataFrame - the sensor data to feed the models
    """
    failure_descriptions = {
        "TWF": "Tool wear failure",
        "HDF": "Heat dissipation failure",
        "PWF": "Power failure",
        "OSF": "Overstrain failure",
    }
    st.write("## Failures occured:")
    for model_name, model in models.items():
        if model_name != "Machine failure":
            if model.predict(sensor_data)[0] == 1:
                st.write(f"### {failure_descriptions[model_name]}")

if __name__ == '__main__':
    with open("models/models.joblib", "rb") as f:
        models = load(f)

    sensor_data = pd.DataFrame({
        "Air temperature [K]": [295.2],
        "Process temperature [K]": [308.4],
        "Rotational speed [rpm]": [1310],
        "Torque [Nm]": [61.0],
        "Tool wear [min]": [189],
        "L": [0],
        "M": [0]
    })

    st.title("Failure Classification")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        tool_quality = st.selectbox("Tool quality", ["Low", "Medium", "High"])
        sensor_data["L"] = [1 if tool_quality == "Low" else 0]
        sensor_data["M"] = [1 if tool_quality == "Medium" else 0]

        air_temp = st.number_input("Air temperature [K]", value=295.2, min_value=0.0)
        sensor_data["Air temperature [K]"] = [air_temp]

        proc_temp = st.number_input(
            "Process temperature [K]", value=308.4, min_value=0.0)
        sensor_data["Process temperature [K]"] = [proc_temp]

        rot_speed = st.number_input(
            "Rotational speed [rpm]", value=1310, min_value=0)
        sensor_data["Rotational speed [rpm]"] = [rot_speed]

        torque = st.number_input("Torque [Nm]", value=61.0, min_value=0.0)
        sensor_data["Torque [Nm]"] = [torque]

        tool_wear = st.number_input(
            "Tool wear [min]", value=189, min_value=0)
        sensor_data["Tool wear [min]"] = [tool_wear]



    with col2:
        display_classification(models, sensor_data)