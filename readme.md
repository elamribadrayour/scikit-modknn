# scikit-modknn

[![PyPI version](https://badge.fury.io/py/scikit-modknn.svg)](https://badge.fury.io/py/scikit-modknn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`scikit-modknn` is a Python package that extends Scikit-learn's k-nearest neighbors (k-NN) algorithm with a customizable modular distance metric. It is designed for users who need more flexibility in distance calculations within Scikit-learn pipelines.

## Features

- **Custom Distance Metric**: Implement your own distance metric to suit specific data requirements.
- **Seamless Integration**: Easily integrate the custom k-NN into existing Scikit-learn workflows.
- **Flexible Configuration**: Adjust the distance metric parameters to optimize model performance.

## Installation

Install the package via pip:

```bash
pip install scikit-modknn
```

## Usage

Here's a quick example of how to use `scikit-modknn` in a Scikit-learn pipeline:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scikit_modknn import CustomKNNClassifier

# Load sample data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Define your pipeline
pipeline = Pipeline([
    ('knn', CustomKNNClassifier(k=3, exponents_array=np.array([1, 1, 1, 1])))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
```

## Configuration

The `CustomKNNClassifier` can be customized with various parameters:
- **k**: Number of nearest neighbors to consider.
- **exponents_array**: Array specifying the power to which each feature's absolute difference is raised.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [Badr Ayour EL AMRI](mailto:badrayour.elamri@ankorstore.com).

## Acknowledgements

- [Scikit-learn](https://scikit-learn.org) for the foundational machine learning framework.
- The open-source community for continuous inspiration and support.
