# generate_dataset_final.py
import os
import cv2
import numpy as np
import random
from pathlib import Path

# Configurações
IMG_SIZE = 128
CLASSES = ["circle", "square", "triangle"]
IMAGES_PER_CLASS = 200
SPLIT = (0.5, 0.25, 0.25)  # treino, val, teste
OUT_ROOT = Path("data")

def ensure_dirs():
    for split in ("train", "val", "test"):
        for c in CLASSES:
            d = OUT_ROOT / split / c
            d.mkdir(parents=True, exist_ok=True)

def draw_shape(img, shape):
    h, w = img.shape[:2]
    color = (0, 0, 0)
    thickness = -1
    if shape == "circle":
        radius = random.randint(20, min(w,h)//3)
        center = (random.randint(radius, w-radius), random.randint(radius, h-radius))
        cv2.circle(img, center, radius, color, thickness)
    elif shape == "square":
        size = random.randint(30,70)
        x = random.randint(0, w-size)
        y = random.randint(0, h-size)
        cv2.rectangle(img, (x,y), (x+size,y+size), color, thickness)
    elif shape == "triangle":
        margin = 15
        pts = np.array([
            [random.randint(margin, w-margin), random.randint(margin, h-margin)],
            [random.randint(margin, w-margin), random.randint(margin, h-margin)],
            [random.randint(margin, w-margin), random.randint(margin, h-margin)]
        ], np.int32)
        cv2.fillPoly(img, [pts], color)
    return img

def randomize_image(img):
    angle = random.uniform(0,360)
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=(255,255,255))
    if random.random() < 0.3:
        noise = np.random.normal(0,10,rotated.shape).astype(np.int16)
        rotated = np.clip(rotated.astype(np.int16)+noise,0,255).astype(np.uint8)
    return rotated

def generate():
    ensure_dirs()
    for cls in CLASSES:
        for i in range(IMAGES_PER_CLASS):
            img = np.ones((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)*255
            img = draw_shape(img, cls)
            img = randomize_image(img)
            r = random.random()
            if r < SPLIT[0]:
                split = "train"
            elif r < SPLIT[0]+SPLIT[1]:
                split = "val"
            else:
                split = "test"
            fname = OUT_ROOT / split / cls / f"{cls}_{i:05d}.png"
            cv2.imwrite(str(fname), img)

if __name__ == "__main__":
    print("Gerando dataset final...")
    generate()
    print("✅ Dataset criado em 'data/train', 'data/val' e 'data/test' (3 classes).")
