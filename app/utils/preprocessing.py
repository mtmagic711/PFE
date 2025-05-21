import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch_geometric.data import Data

def preprocess_image(image_path, size=(128, 128)):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Image path is invalid ou le fichier n'existe pas.")
    img = Image.open(image_path).convert('RGB').resize(size)
    arr = np.array(img) / 255.0
    arr = arr[np.newaxis, ...]  # shape (1, height, width, 3)
    return arr

def preprocess_stroke_image(image_path):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Image path is invalid ou le fichier n'existe pas.")
    return preprocess_image(image_path, size=(224, 224))

def preprocess_ms_image(image_path):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Image path is invalid ou le fichier n'existe pas.")
    img = Image.open(image_path).convert('L').resize((128, 128))  # Convert to grayscale
    arr = np.array(img) / 255.0
    arr = arr[np.newaxis, ..., np.newaxis]  # shape (1, 128, 128, 1)
    return arr

def preprocess_image_pytorch(image_path):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Image path is invalid ou le fichier n'existe pas.")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)  # shape: (1, 3, 224, 224)
    return input_tensor

def preprocess_image_pytorch_grayscale(image_path):
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("Image path is invalid ou le fichier n'existe pas.")
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = Image.open(image_path).convert('L')
    input_tensor = preprocess(img).unsqueeze(0)  # shape: (1, 1, 224, 224)
    return input_tensor

def image_to_graph(img_path, patch_size=32):
    img = Image.open(img_path).convert('L').resize((128, 128))
    img = np.array(img) / 255.0
    h, w = img.shape
    ph, pw = patch_size, patch_size

    features = []
    positions = []

    for i in range(0, h, ph):
        for j in range(0, w, pw):
            patch = img[i:i+ph, j:j+pw]
            vec = [
                np.mean(patch),
                np.std(patch),
                np.min(patch),
                np.max(patch),
                np.median(patch)
            ]
            features.append(vec)
            positions.append((i // ph, j // pw))

    x = torch.tensor(features, dtype=torch.float)

    # Connect adjacent patches (grid neighbors)
    edge_index = []
    grid_h = h // ph
    grid_w = w // pw

    for idx, (i, j) in enumerate(positions):
        neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
        for ni, nj in neighbors:
            if 0 <= ni < grid_h and 0 <= nj < grid_w:
                nid = ni * grid_w + nj
                edge_index.append([idx, nid])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index) 