import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_text_embedding

query = input("Enter search query: ")

query_vector = np.array(get_text_embedding(query)).reshape(1, -1)

with open("image_vectors.pkl", "rb") as f:
    image_vectors = pickle.load(f)

scores = {}

for img, vec in image_vectors.items():

    vec = np.array(vec).reshape(1, -1)

    similarity = cosine_similarity(query_vector, vec)[0][0]

    scores[img] = similarity

results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\nTop Matches:\n")

for img, score in results:

    print(img, ":", score)