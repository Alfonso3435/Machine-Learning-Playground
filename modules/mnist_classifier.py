import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def show_mnist_classifier():
    st.title("🔢 MNIST Digit Classifier — Neural Network for Multiclass classification")
    st.write(
        "This model implements a dense neural network using TensorFlow to classify handwritten digits (0-9) from the MNIST dataset."
    )
    st.write("Dataset source: [MNIST Handwritten Digits](https://keras.io/api/datasets/mnist/)")

    # Fixed Hyperparameter
    epochs = 10
    batch_size = 128
    hidden_units = 64
    learning_rate = 0.001

    st.sidebar.header("Fixed Model Parameters")
    st.sidebar.write(f"**Epochs:** {epochs}")
    st.sidebar.write(f"**Batch Size:** {batch_size}")
    st.sidebar.write(f"**Hidden Units:** {hidden_units}")
    st.sidebar.write(f"**Learning Rate:** {learning_rate}")

    # Load and preprocess dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize and flatten
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train_oh = to_categorical(y_train, 10)
    y_test_oh = to_categorical(y_test, 10)

    # --- About the Dataset ---
    st.subheader("📚 About the Dataset")
    st.write(
        "The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9). "
        "Each image is 28x28 pixels. The dataset is widely used to evaluate image classification models. "
        "Below you can see some examples from the training set."
    )

    # Display random training samples
    indices = np.random.choice(len(X_train), 12, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[indices[i]], cmap='gray')
        ax.set_title(f"Label: {y_train[indices[i]]}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    st.write(
        "Each image corresponds to a handwritten digit that the neural network will learn to classify. "
        "After training, the model will predict which number is written in unseen images."
    )

    # Build and train the model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(hidden_units, activation='relu'),
        Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    with st.spinner('Training neural network... please wait ⏳'):
        history = model.fit(X_train, y_train_oh, validation_split=0.1, epochs=epochs,
                            batch_size=batch_size, verbose=0)
    st.success("✅ Model training complete!")

    # --- Evaluate performance ---
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    st.subheader("📊 Model Evaluation")
    st.write(f"**Test Accuracy:** {test_acc * 100:.2f}%")
    st.write(f"**Test Loss:** {test_loss:.4f}")

    # --- Plot loss and accuracy ---
    st.subheader("📈 Training Progress")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title("Loss over Epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title("Accuracy over Epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    st.pyplot(fig)
    st.write("These plots show how the model improves during training and how well it generalizes on validation data.")

    # Predictions on test data
    y_pred = np.argmax(model.predict(X_test[:2000]), axis=1)
    y_true = y_test[:2000]

    st.subheader("🔍 Sample Predictions on Test Set")
    n_samples = 10
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 1.5, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X_test[indices[i]], cmap='gray')
        pred_label = np.argmax(model.predict(X_test[indices[i]].reshape(1, 28, 28)), axis=1)[0]
        color = 'green' if pred_label == y_test[indices[i]] else 'red'
        ax.set_title(f"P: {pred_label}\nT: {y_test[indices[i]]}", color=color, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Each image shows its Predicted (P) and True (T) label — green means correct, red means incorrect.")

    # --- Confusion Matrix ---
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix for MNIST Neural Network")
    st.pyplot(fig)


    # --- Insights ---
    st.markdown("---")
    st.info(
        "- The network uses one hidden layer with 64 ReLU units and a softmax output for 10 classes.\n"
        "- MNIST digits are well-suited for testing simple neural networks.\n"
        "- With 10 epochs, the model achieves about **97–98% accuracy**.\n"
    )
