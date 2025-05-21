import os
from kivy.uix.screenmanager import Screen
from config import DISEASE_INFO, SCREEN_TO_DISEASE, MODEL_EXPLANATIONS
from utils.preprocessing import (
    preprocess_image,
    preprocess_stroke_image,
    preprocess_ms_image,
    preprocess_image_pytorch,
    preprocess_image_pytorch_grayscale
)

class DiseaseBaseScreen(Screen):
    def get_model_explanation(self, model_name):
        if model_name.replace("-", "").upper() == "UNET":
            return MODEL_EXPLANATIONS.get("U-Net", "")
        return MODEL_EXPLANATIONS.get(model_name, "")

    def on_enter(self):
        disease_key = SCREEN_TO_DISEASE.get(self.name, self.name)
        info = DISEASE_INFO.get(disease_key, {})
        text = f"[b]Définition :[/b]\n{info.get('definition', '')}\n\n[b]Symptômes :[/b]\n{info.get('symptoms', '')}"
        if 'disease_info' in self.ids:
            self.ids.disease_info.text = text
        if 'model_selector' in self.ids and 'model_explanation' in self.ids:
            model_name = self.ids.model_selector.text
            self.ids.model_explanation.text = self.get_model_explanation(model_name)
            self.ids.model_selector.unbind(text=self.on_model_change)
            self.ids.model_selector.bind(text=self.on_model_change)

    def on_model_change(self, instance, value):
        self.ids.model_explanation.text = self.get_model_explanation(value)

    def open_filechooser(self):
        from kivy.uix.filechooser import FileChooserListView
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button

        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg'])
        content.add_widget(filechooser)

        btn = Button(text='Sélectionner', size_hint_y=0.1)
        btn.bind(on_press=lambda x: self.select_image(filechooser.path, filechooser.selection))
        content.add_widget(btn)

        self.popup = Popup(title="Sélectionner une image cérébrale", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def select_image(self, path, selection):
        if selection:
            self.ids.uploaded_image.source = selection[0]
        self.popup.dismiss()

    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        if not image_path or not os.path.isfile(image_path):
            self.ids.result_label.text = "Veuillez d'abord sélectionner une image valide."
            return
        self.ids.result_label.text = "Analyse terminée ! (résultat simulé)" 