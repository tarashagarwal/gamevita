import torch
import sys
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from ultralytics import YOLO

# ---------- Load Models ----------
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

yolo_model = YOLO("yolov8s.pt")  # You can switch to yolov8n.pt for faster


# ---------- CLIP Embedding ----------
def get_embedding(image_path):
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

    return set(objects)  # return unique set of object names


# ---------- Compare 3 Images ----------
def compare_three_images(imgA, imgB, imgC, threshold=0.80):
    print("\nProcessing images...\n")

    # ---- CLIP Similarity ----
    embA = get_embedding(imgA)
    embB = get_embedding(imgB)
    embC = get_embedding(imgC)

    sim_AC = cosine_similarity(embA, embC)
    sim_BC = cosine_similarity(embB, embC)

    # ---- YOLO Objects ----
    objA = detect_objects(imgA)
    objB = detect_objects(imgB)
    objC = detect_objects(imgC)

    # Object overlap count
    overlap_AC = len(objA.intersection(objC))
    overlap_BC = len(objB.intersection(objC))

    print(f"CLIP Similarity C↔A: {sim_AC:.4f}")
    print(f"CLIP Similarity C↔B: {sim_BC:.4f}")
    print("\nObjects in A:", objA)
    print("Objects in B:", objB)
    print("Objects in C:", objC)

    print(f"\nObject Overlap C↔A: {overlap_AC}")
    print(f"Object Overlap C↔B: {overlap_BC}")

    # ---- Decision Logic ----
    score_A = sim_AC + (0.05 * overlap_AC)
    score_B = sim_BC + (0.05 * overlap_BC)

    print(f"\nFinal Score (with objects): A={score_A:.4f}  B={score_B:.4f}")

    if score_A > score_B:
        print("\n✅ Image C is more similar to Image A")
    elif score_B > score_A:
        print("\n✅ Image C is more similar to Image B")
    else:
        print("\n⚖️ Image C is equally similar to both")


# ---------- RUN ----------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_three_images_yolo_clip.py imgA.jpg imgB.jpg imgC.jpg")
    else:
        compare_three_images(sys.argv[1], sys.argv[2], sys.argv[3])
