"""
This module runs initial data cleaning and preparation for model training.
"""
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def undersample(data: pd.DataFrame) -> pd.DataFrame:
    """
    Undersamples the initial data using RandomUnderSampler to fix 
    the imblance of failures and non failures

    Params:
        data: pd.DataFrame - the initial data read in as a 
                            dataframe from the csv file
    Returns:
        pd.DataFrame - the new undersampled dataframe with a balance
                          of failures and non failures
    """
    rus = RandomUnderSampler(random_state=42, replacement=True)
    X = data.drop(
        columns=["Machine failure"])
    y = data["Machine failure"]
    x_rus, y_rus = rus.fit_resample(X, y)

    undersampled_data_df = pd.concat([x_rus, y_rus], axis=1)
    print("Completed undersampling")
    print(f"Original data lenght: {len(data)}")
    print(f"Undersampled data lenght: {len(undersampled_data_df)}")
    return undersampled_data_df

def category_to_binary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the categorical data in the dataframe to binary data 
    for model training using one hot encoding, pandas get_dummies().

    Params:
        data: pd.DataFrame - the dataframe to convert the categorical.
                            The data should be the undersampled data
    Returns:
        pd.DataFrame - the dataframe with the categorical data converted
                        to binary data
    """
    binary_data = pd.get_dummies(
        data['Type'], dtype=int, drop_first=True)
    result = pd.concat([data, binary_data], axis=1)
    print("Completed converting categorical data to binary")
    return result

def pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the initial data and prepares it for model training.

    Params:
        data: pd.DataFrame - the initial data read in as a 
                            dataframe from the csv file
    returns:
        X: - the processed features data ready for model training
        y: - the processed target data ready for model training
    """
    undersampled_data = undersample(data)
    cleaned_data = category_to_binary(undersampled_data)
    cleaned_data.drop(columns=["Type", "Product ID",
                                                  "UDI","TWF", "HDF", 
                                                  "PWF", "OSF", "RNF"], 
                                                  inplace=True)
    X = cleaned_data.drop(columns=["Machine failure"])
    y = cleaned_data["Machine failure"]
    print("Data ready for model training")
    return X, y

if __name__ == '__main__':
    initial_data = pd.read_csv("../data/ai4i2020.csv")

    X, y = pipeline(initial_data)