import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_logistic_regression():
    st.title("🧠 Logistic Regression from Scratch")
    st.write(
        "This demo implements a logistic regression model from scratch to predict heart disease outcomes "
        "using the Framingham dataset. Feel free to check the implementation in the source code."
    )
    st.write("Dataset: [Framingham Heart Study](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/data)")

    # Sidebar for threshold
    st.sidebar.header("Prediction Settings")
    threshold = st.sidebar.slider("Classification Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Load dataset
    heart_disease = pd.read_csv("./data/framingham.csv")
    heart_disease = heart_disease.dropna()
    st.subheader("📊 Dataset Preview")
    st.write("The feature we want to predict is: TenYearCHD")
    st.write(f"Shape of the dataset: {heart_disease.shape}")
    st.dataframe(heart_disease.head(10))

    # Fixed parameters
    lr = 0.1
    epochs = 1000

    # Select features and labels
    X = heart_disease.drop(columns='TenYearCHD', axis=1)
    y = heart_disease['TenYearCHD']

    # Split data and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Logistic Regression implementation
    class LogisticRegressionScratch:
        def __init__(self, lr=0.1, epochs=1000):
            self.lr = lr
            self.epochs = epochs

        def fit(self, X, y):
            m, n = X.shape
            self.theta = np.zeros((n, 1))
            self.bias = 0
            y = y.reshape(m, 1)
            self.losses = []

            for _ in range(self.epochs):
                z = np.dot(X, self.theta) + self.bias
                y_hat = sigmoid(z)
                loss = -(1/m) * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
                self.losses.append(loss)

                dw = (1/m) * np.dot(X.T, (y_hat - y))
                db = (1/m) * np.sum(y_hat - y)
                self.theta -= self.lr * dw
                self.bias -= self.lr * db

        def predict_proba(self, X):
            return sigmoid(np.dot(X, self.theta) + self.bias)

        def predict(self, X, threshold=0.5):
            y_prob = self.predict_proba(X)
            return (y_prob >= threshold).astype(int)

    # Train the model
    model = LogisticRegressionScratch(lr=lr, epochs=epochs)
    model.fit(X_train, y_train.values)

    # Predictions
    y_prob = model.predict_proba(X_test).flatten()
    y_pred = model.predict(X_test, threshold=threshold).flatten()
    accuracy = np.mean(y_pred == y_test) * 100

    # Display accuracy
    st.subheader("Model Performance")
    st.write(f"Model Accuracy (threshold={threshold}): {accuracy:.2f}%")

    # Plot loss
    st.subheader("📉 Loss Over Time")
    fig, ax = plt.subplots()
    ax.plot(model.losses, color='blue')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Function Decrease During Training")
    st.write("This plot shows how the model’s error decreases as it learns during training. A downward trend indicates improvement.")
    st.pyplot(fig)

    # Scatter plot
    st.subheader("🔍 Predicted Probability vs Actual Values (Test Set)")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(range(len(y_test)), y_test.values, label="Actual", color='black', alpha=0.7)
    ax.scatter(range(len(y_test)), y_prob, label="Predicted Probability", color='red', alpha=0.5)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Heart Disease (0=No, 1=Yes)")
    ax.set_title(f"Predicted Probability vs Actual Labels (Threshold = {threshold})")
    ax.legend()
    st.write("This scatter plot shows the **predicted probabilities** (red) compared to the **true labels** (black). Points closer to the actual labels indicate better prediction confidence.")
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    st.write("The confusion matrix shows how many predictions were correct or incorrect for each class (0 = no disease, 1 = disease).")
    st.pyplot(fig)

    st.markdown("---")
    st.info("In this dataset, I saw that it was very difficult for the algorithm to predict positive labels to a reasonable theshold and accuracy. With this, I concluded that the challenge in predicting heart diseases reflects the complexity of human biology more than model limitations")
