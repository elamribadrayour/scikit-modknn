import numpy
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


class ModKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k: int, exponents_array: numpy.ndarray ) -> None:
        self.k = k
        self.exponents_array = exponents_array

    @staticmethod
    def modular_distance(
        x1: numpy.ndarray, x2: numpy.ndarray, exponents_array: numpy.ndarray
    ) -> float:
        return numpy.sum(numpy.abs(x1 - x2) ** exponents_array)

    def fit(self, X: numpy.ndarray, y: numpy.ndarray) -> "ModKNNClassifier":
        self.X_train = X
        self.y_train = y

        if self.exponents_array.shape[0] != X.shape[1]:
            raise ValueError(
                "The number of exponents must match the number of features."
            )

        return self

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        return numpy.array([self._predict(x) for x in X])

    def _predict(self, x: numpy.ndarray) -> int:
        distances = [
            self.modular_distance(x, x_train, self.exponents_array)
            for x_train in self.X_train
        ]
        k_indices = numpy.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_breast_cancer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    standard_knn = KNeighborsClassifier(n_neighbors=3)
    standard_knn.fit(X_train, y_train)
    standard_predictions = standard_knn.predict(X_test)
    standard_accuracy = accuracy_score(y_test, standard_predictions)
    print(f"Standard KNN Accuracy: {standard_accuracy:.2f}")

    for i in range(1, 10):
        # Initialize and train the custom k-NN classifier
        custom_knn = ModKNNClassifier(
            k=3, exponents_array=i * numpy.ones(X_train.shape[1])
        )
        custom_knn.fit(X_train, y_train)
        custom_predictions = custom_knn.predict(X_test)
        custom_accuracy = accuracy_score(y_test, custom_predictions)
        print(f"Custom KNN Accuracy for i={i}: {custom_accuracy:.2f}")
