import os
from .base import DiseaseBaseScreen
import numpy as np
from utils.preprocessing import preprocess_image
from models import tumor_unet

class BrainTumorScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        if not image_path or not os.path.isfile(image_path):
            self.ids.result_label.text = "Veuillez d'abord sélectionner une image valide."
            return
        try:
            model_type = self.ids.model_selector.text
            if model_type == "U-Net":
                arr = preprocess_image(image_path)
                pred = tumor_unet.predict(arr)
                affected = np.argmax(pred) == 1
                self.ids.result_label.text = (
                    "Vous êtes atteint(e) d'une tumeur cérébrale." if affected else "Vous n'êtes pas atteint(e) d'une tumeur cérébrale."
                )
        except ValueError as e:
            self.ids.result_label.text = str(e)
            return 