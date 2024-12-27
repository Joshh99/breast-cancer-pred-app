import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def train_test_model(X, y) -> tuple:
    """Creates a model using the given data by fitting a model to the data.
    Applies some preprocessing steps to the data before fitting the model.
    Args:
        X (pd.DataFrame): The features of the data.
        y (pd.Series): The target variable of the data.
    """
    # Scale the features data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Fit the model to the data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy


def main():
    breast_cancer = load_breast_cancer(as_frame=True)
    X, y = load_breast_cancer(return_X_y= True, as_frame=True)
    model, scaler, accuracy = train_test_model(X, y)
    
    # Save the model and scaler
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        

    
    
    
if __name__ == "__main__":
    main()