## Multimodal Semantic Image Search using Gemini Embeddings
A lightweight multimodal semantic search system that retrieves the most relevant image from a dataset using natural language queries.
The system leverages Google Gemini Embeddings to convert both text and images into high-dimensional vector representations and performs similarity matching using cosine similarity.
This approach enables semantic retrieval, allowing images to be discovered based on conceptual meaning rather than exact keyword matches.

## Architecture
User Query
   │
   ▼
Text Embedding (Gemini)
   │
   ▼
Query Vector
   │
   ▼
Cosine Similarity
   │
   ▼
Image Embedding Store
   │
   ▼
Top Match
   │
   ▼
Streamlit UI

## Features
- Multimodal embeddings (text + images)
- Semantic search instead of keyword matching
- Image retrieval using vector similarity
- Cosine similarity ranking
- Precomputed image embedding storage
- Interactive Streamlit UI
- Lightweight and easy to run locally

## Tech Stack
## Programming Language
- Python

## Libraries / Tools
- Google Gemini API
- NumPy
- Scikit-learn
- Streamlit
- Python Dotenv

## Running the Project
- Generate image embeddings
- python embed_images.py
- Run semantic search from terminal
- python search.py
- Run the Streamlit interface
- streamlit run app.py

## Learning Outcomes
This project demonstrates practical concepts used in modern AI retrieval systems:
- Multimodal embeddings
- Vector similarity search
- Semantic retrieval
- AI-powered search pipelines
- API integration with Gemini
- Interactive ML applications using Streamlit

## Project Structure
MultiModel-Embeddings/
── images/                # Dataset images used for search
── embed_images.py        # Script to generate image embeddings
── search.py              # Text-to-image similarity search
── app.py                 # Streamlit web interface
── requirements.txt
── .env                   # API key 
── README.md
