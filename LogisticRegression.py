import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
       
        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Update weights and bias
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            print("dw: ", dw)
            print("db: ", db)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_class = [1 if pred >= 0.5 else 0 for pred in y_pred]
        return y_pred_class

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    
if __name__ == '__main__':
    # Imports
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Load dataset
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Initialize model
    model = LogisticRegression(learning_rate=0.0001, num_iterations=500)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(accuracy)