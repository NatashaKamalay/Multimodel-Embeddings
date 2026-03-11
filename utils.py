import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_image_embedding(image_path):

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            )
        ],
    )

    return result.embeddings[0].values


def get_text_embedding(text):

    result = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=[text],
    )

    return result.embeddings[0].values