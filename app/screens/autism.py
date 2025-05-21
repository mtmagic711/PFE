import os
from .base import DiseaseBaseScreen
import torch
import numpy as np
from PIL import Image
from torch_geometric.data import Data
from models import autism_model
from config import DISEASE_INFO, SCREEN_TO_DISEASE, MODEL_EXPLANATIONS
from utils.preprocessing import image_to_graph

class AutismScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        model_type = self.ids.model_selector.text
        if model_type == "GCN":
            try:
                data = image_to_graph(image_path)
                data = data.to(torch.device("cpu"))
                with torch.no_grad():
                    output = autism_model(data)
                    pred = torch.argmax(output, dim=1).item()
                affected = pred == 1
                self.ids.result_label.text = (
                    "Vous êtes atteint(e) d'autisme." if affected else "Vous n'êtes pas atteint(e) d'autisme."
                )
            except Exception as e:
                self.ids.result_label.text = f"Erreur lors de l'analyse : {str(e)}" 