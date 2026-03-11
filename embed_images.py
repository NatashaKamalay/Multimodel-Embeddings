import os
import pickle
from utils import get_image_embedding

IMAGE_FOLDER = "images"

embeddings = {}

for img in os.listdir(IMAGE_FOLDER):

    path = os.path.join(IMAGE_FOLDER, img)

    print("Embedding:", img)

    embeddings[img] = get_image_embedding(path)

with open("image_vectors.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Image embeddings saved!")