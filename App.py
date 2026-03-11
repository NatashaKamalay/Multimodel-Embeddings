import streamlit as st
import pickle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_text_embedding

st.title("Semantic Image Search using Gemini Embeddings")

query = st.text_input("Enter your search query")

if query:

    query_vector = np.array(get_text_embedding(query)).reshape(1, -1)

    with open("image_vectors.pkl", "rb") as f:
        image_vectors = pickle.load(f)

    best_match = None
    best_score = -1

    for img, vec in image_vectors.items():

        vec = np.array(vec).reshape(1, -1)

        similarity = cosine_similarity(query_vector, vec)[0][0]

        if similarity > best_score:
            best_score = similarity
            best_match = img

    st.subheader("Most Relevant Image")

    image_path = os.path.join("images", best_match)

    st.image(image_path, caption=f"{best_match} (similarity: {best_score:.3f})")