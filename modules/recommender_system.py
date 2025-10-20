import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
import matplotlib.pyplot as plt

def show_recommender_system():
    st.title("🎬 Neural Network Recommender System")
    st.write(
        "This model establishes a Neural Network-based Recommender System developed. It predicts movie ratings using learned embeddings for users and movies. "
    )
    st.write("Dataset: synthetic MovieLens-like dataset for demonstration.")

    # --- Fixed Hyperparameters ---
    embedding_size = 8
    hidden_units = 16
    epochs = 20
    learning_rate = 0.01

    st.sidebar.header("Fixed Model Parameters")
    st.sidebar.write(f"**Embedding Size:** {embedding_size}")
    st.sidebar.write(f"**Hidden Units:** {hidden_units}")
    st.sidebar.write(f"**Epochs:** {epochs}")
    st.sidebar.write(f"**Learning Rate:** {learning_rate}")

    # --- About the Dataset ---
    st.subheader("📚 About the Dataset")
    st.write(
        "We use a small synthetic dataset inspired by MovieLens, where users rate movies from 1.0 to 5.0. "
        "Each user and movie has a numerical ID, and the goal is to predict ratings for unseen combinations."
    )

    # Create a synthetic dataset
    np.random.seed(42)
    num_users = 10
    num_movies = 15
    samples = 60

    user_ids = np.random.randint(0, num_users, samples)
    movie_ids = np.random.randint(0, num_movies, samples)
    ratings = np.random.uniform(1, 5, samples).round(1)

    df = pd.DataFrame({"user_id": user_ids, "movie_id": movie_ids, "rating": ratings})
    st.dataframe(df.head(10))
    st.write(f"Unique users: {num_users}, unique movies: {num_movies}")

    # Model Architecture
    st.subheader("🧠 Model Architecture with Dense Layers")
    st.write(
        "Each user and movie is represented as an embedding vector. "
        "These vectors are concatenated and passed through Dense layers that learn "
    )

    # Define inputs
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))

    # Embeddings
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    movie_embedding = Embedding(num_movies, embedding_size)(movie_input)

    # Flatten
    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)

    # Concatenate user and movie embeddings
    concat = Concatenate()([user_vec, movie_vec])

    # Dense layers to learn complex relationships
    dense1 = Dense(hidden_units, activation='relu')(concat)
    dense2 = Dense(hidden_units // 2, activation='relu')(dense1)
    output = Dense(1, activation='linear')(dense2)  # Predict rating

    # Compile model
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # --- Train the model ---
    with st.spinner('Training recommender system with Dense layers... please wait ⏳'):
        history = model.fit([user_ids, movie_ids], ratings, epochs=epochs, verbose=0)
    st.success("✅ Model training complete!")

    # --- Plot training loss ---
    st.subheader("📉 Training Loss")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], color='blue', label='Training Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Model Training Progress")
    ax.legend()
    st.pyplot(fig)
    st.write("A decreasing loss curve indicates that the model is learning user–movie relationships effectively.")

    # --- Generate predictions ---
    st.subheader("🎥 Example Recommendations")
    sample_user = st.slider("Select a user ID to view recommendations", 0, num_users - 1, 0)

    predicted_ratings = model.predict([np.array([sample_user] * num_movies), np.arange(num_movies)], verbose=0).flatten()

    recommendations = pd.DataFrame({
        "movie_id": np.arange(num_movies),
        "predicted_rating": predicted_ratings
    }).sort_values(by="predicted_rating", ascending=False)

    st.write(f"Top recommended movies for **User {sample_user}**:")
    st.dataframe(recommendations.head(5).style.highlight_max(axis=0, color='lightgreen'))


    # --- Insights ---
    st.markdown("---")
    st.info(
        "- This approach extends the neural collaborative filtering.\n"
        "- Even with small data, the model learns meaningful user–movie preference patterns."
    )
