import os
from .base import DiseaseBaseScreen
import torch
import numpy as np
from utils.preprocessing import preprocess_image, preprocess_image_pytorch
from models import alzheimer_cnn, alzheimer_vit

class AlzheimerScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        try:
            model_type = self.ids.model_selector.text
            if model_type == "CNN":
                arr = preprocess_image(image_path)
                pred = alzheimer_cnn.predict(arr)
                affected = np.argmax(pred) != 0
            else:  # ViT
                arr = preprocess_image_pytorch(image_path)
                with torch.no_grad():
                    output = alzheimer_vit(arr)
                    pred = torch.argmax(output, dim=1).item()
                affected = pred != 0
            self.ids.result_label.text = (
                "Vous êtes atteint(e) d'Alzheimer." if affected else "Vous n'êtes pas atteint(e) d'Alzheimer."
            )
        except ValueError as e:
            self.ids.result_label.text = str(e)
            return 