import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from ultralytics import YOLO

# ---------- Load Models ----------
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

yolo_model = YOLO("yolov8s.pt")  # small model, fast and good enough


# ---------- CLIP Embeddings ----------
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.squeeze().numpy()


def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2))


# ---------- YOLO Object Detection ----------
def detect_objects(image_path):
    results = yolo_model(image_path)[0]
    objects = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        objects.append(label)

    return objects


# ---------- Compare Images ----------
def compare_images_with_objects(imgA, imgB, threshold=0.85):
    # CLIP similarity
    embA = get_image_embedding(imgA)
    embB = get_image_embedding(imgB)
    sim = cosine_similarity(embA, embB)

    print(f"\nCLIP Similarity: {sim:.4f}")

    # YOLO object detection
    objectsA = detect_objects(imgA)
    objectsB = detect_objects(imgB)

    setA = set(objectsA)
    setB = set(objectsB)

    print(f"\nObjects in Image A: {setA}")
    print(f"Objects in Image B: {setB}")

    # Check for new or missing objects
    new_objects = setB - setA
    removed_objects = setA - setB

    if sim >= threshold:
        print("\n✅ Images are semantically similar by CLIP.")
    else:
        print("\n❌ Images are semantically different by CLIP.")

    if new_objects:
        print(f"🆕 New objects in Image B: {new_objects}")

    if removed_objects:
        print(f"❌ Objects missing in Image B: {removed_objects}")

    if not new_objects and not removed_objects:
        print("\nNo significant object change detected by YOLO.")


# ---------- RUN ----------
if __name__ == "__main__":
    imgA = "image1.jpg"
    imgB = "image2.jpg"
    compare_images_with_objects(imgA, imgB)
