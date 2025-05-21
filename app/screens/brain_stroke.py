import os
from .base import DiseaseBaseScreen
import numpy as np
from utils.preprocessing import preprocess_stroke_image
from models import stroke_cnn

class BrainStrokeScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        if not image_path or not os.path.isfile(image_path):
            self.ids.result_label.text = "Veuillez d'abord sélectionner une image valide."
            return
        try:
            model_type = self.ids.model_selector.text
            if model_type == "CNN":
                arr = preprocess_stroke_image(image_path)
                pred = stroke_cnn.predict(arr)
                affected = np.argmax(pred) == 1
                self.ids.result_label.text = (
                    "Vous êtes atteint(e) d'AVC." if affected else "Vous n'êtes pas atteint(e) d'AVC."
                )
        except ValueError as e:
            self.ids.result_label.text = str(e)
            return 