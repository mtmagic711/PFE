import os
from .base import DiseaseBaseScreen
import numpy as np
from utils.preprocessing import preprocess_ms_image
from models import ms_cnn, ms_unet

class MScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        try:
            model_type = self.ids.model_selector.text
            if model_type == "CNN":
                arr = preprocess_ms_image(image_path)
                pred = ms_cnn.predict(arr)
            else:  # U-Net
                arr = preprocess_ms_image(image_path)
                pred = ms_unet.predict(arr)
            affected = np.argmax(pred) == 1
            self.ids.result_label.text = (
                "Vous êtes atteint(e) de la SEP." if affected else "Vous n'êtes pas atteint(e) de la SEP."
            )
        except ValueError as e:
            self.ids.result_label.text = str(e)
            return 