import os
from .base import DiseaseBaseScreen
import torch
from utils.preprocessing import preprocess_image_pytorch, preprocess_image_pytorch_grayscale
from models import parkinson_model

class ParkinsonScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        if not image_path or not os.path.isfile(image_path):
            self.ids.result_label.text = "Veuillez d'abord sélectionner une image valide."
            return
        try:
            model_type = self.ids.model_selector.text
            if model_type == "VGG19":
                arr = preprocess_image_pytorch(image_path)
                with torch.no_grad():
                    output = parkinson_model(arr)
                    pred = torch.argmax(output, dim=1).item()
                affected = pred == 1
                self.ids.result_label.text = (
                    "Vous êtes atteint(e) de Parkinson." if affected else "Vous n'êtes pas atteint(e) de Parkinson."
                )
            elif model_type == "CNN":
                arr = preprocess_image_pytorch_grayscale(image_path)
                with torch.no_grad():
                    output = parkinson_model(arr)
                    pred = torch.argmax(output, dim=1).item()
                affected = pred == 1
                self.ids.result_label.text = (
                    "Vous êtes atteint(e) de Parkinson (CNN)." if affected else "Vous n'êtes pas atteint(e) de Parkinson (CNN)."
                )
        except ValueError as e:
            self.ids.result_label.text = str(e)
            return 