import streamlit as st


st.sidebar.title("Machine Learning Playground")
option = st.sidebar.selectbox(
    "Choose a module:",
    ["Logistic Regression (from scratch)", "Softmax Regression", "MNIST Classifier", "Recommender System"]
)

if option == "Logistic Regression (from scratch)":
    from modules.logistic_regression import show_logistic_regression
    show_logistic_regression()

if option == "MNIST Classifier":
    from modules.mnist_classifier import show_mnist_classifier
    show_mnist_classifier()