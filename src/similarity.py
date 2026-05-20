from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from sklearn.metrics.pairwise import cosine_similarity

model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_embedding(image):
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"])
        embedding = outputs.image_embeds

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()


def load_product_images(folder="data/products"):
    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(Path(folder).glob(ext))

    return image_paths


def find_similar_images(uploaded_image, top_k=5):
    uploaded_embedding = get_image_embedding(uploaded_image)
    product_paths = load_product_images()

    similarities = []

    for path in product_paths:
        try:
            product_image = Image.open(path).convert("RGB")
            product_embedding = get_image_embedding(product_image)

            score = cosine_similarity(uploaded_embedding, product_embedding)[0][0]

            similarities.append((path, score))

        except Exception as e:
            print(f"Error processing {path}: {e}")

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]