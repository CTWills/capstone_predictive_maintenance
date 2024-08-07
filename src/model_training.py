"""
This module creates and trains the logistic regression model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             f1_score,
                            )
from imblearn.under_sampling import RandomUnderSampler


def train_model(model: LogisticRegression, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
    """
    Trains the given model on the given data by splitting the given data
    with test size of 0.25

    params:
        model: LogisticRegression - the model to be trained
        X: pd.DataFrame - the features to be trained on
        y: pd.DataFrame | pd.Series - the target variable
    returns:
        model: LogisticRegression - the trained model
        X_train: pd.DataFrame - the training data
        y_train: pd.DataFrame | pd.Series - the training target data
        X_test: pd.DataFrame - the test data
        y_test: pd.DataFrame | pd.Series - the test target data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test

def test_model(model: LogisticRegression, X: pd.DataFrame, y: pd.DataFrame | pd.Series):
    """
    Tests the given model on the given data using KFold cross validation

    params:
        model: LogisticRegression - the model to be tested
        X: pd.DataFrame - the features to be tested on
        y: pd.DataFrame | pd.Series - the target variable
    returns:
        model: LogisticRegression - the trained model
        accuracy_scores: float - the average accuracy score
        precision_scores: float - the average precision score
        recall_scores: float - the average recall score
        f1_scores: float - the average f1 score
    """
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    trained_model, X_train, y_train, X_test, y_test = train_model(model, 
                                                                  X, 
                                                                  y)
    for train_set, val_set in kf.split(X_train):
        X_train_split, y_train_split = X_train.iloc[train_set], y_train.iloc[train_set]
        X_val_split, y_val_split = X_train.iloc[val_set], y_train.iloc[val_set]
        trained_model.fit(X_train_split, y_train_split)
        model_predictions = trained_model.predict(X_val_split)

        accuracy_scores.append(accuracy_score(y_val_split, model_predictions))
        precision_scores.append(precision_score(
            y_val_split, model_predictions, zero_division=0))
        recall_scores.append(recall_score(y_val_split, model_predictions))
        f1_scores.append(f1_score(y_val_split, model_predictions))
    
    trained_model.fit(X_train, y_train)

    # Create and save confusion matrix for the model
    print(f"y_test: {y_test.value_counts()}")
    ConfusionMatrixDisplay(confusion_matrix(
        y_test, trained_model.predict(X_test))).plot()
    plt.title(f"Target: {y.name}")
    plt.savefig(f"./images/confusion_matrix_{y.name}.png")

    trained_model.fit(X, y)

    return (
        trained_model,
            np.mean(accuracy_scores), 
            np.mean(precision_scores), 
            np.mean(recall_scores), 
            np.mean(f1_scores)
            )
    
def create_model(X: pd.DataFrame, y: pd.DataFrame | pd.Series):
    """
    Creates a logistic regression model and tests it on the given data

    params:
        X: pd.DataFrame - the features to classify target variable
        y: pd.DataFrame | pd.Series - the target variable
    returns:
        model: LogisticRegression - the trained model
        accuracy: float - the average accuracy score
        precision: float - the average precision score
        recall: float - the average recall score
        f1: float - the average
    """
    log_reg = LogisticRegression(max_iter=1000)
    model, accuracy, precision, recall, f1 = test_model(log_reg, X, y)
    return model, accuracy, precision, recall, f1


if __name__ == '__main__':
    pass