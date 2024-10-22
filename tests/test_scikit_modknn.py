# tests/test_scikit_modknn.py

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from scikit_modknn.scikit_modknn import ModKNNClassifier


def run_mod_knn_accuracy():
    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and train the custom k-NN classifier
    custom_knn = ModKNNClassifier(k=3, exponents_array=np.ones(X_train.shape[1]))
    custom_knn.fit(X_train, y_train)
    custom_predictions = custom_knn.predict(X_test)
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    print(f"Custom KNN Accuracy: {custom_accuracy:.2f}")

    # Initialize and train the standard k-NN classifier from scikit-learn
    standard_knn = KNeighborsClassifier(n_neighbors=3)
    standard_knn.fit(X_train, y_train)
    standard_predictions = standard_knn.predict(X_test)
    standard_accuracy = accuracy_score(y_test, standard_predictions)
    print(f"Standard KNN Accuracy: {standard_accuracy:.2f}")


if __name__ == "__main__":
    run_mod_knn_accuracy()
