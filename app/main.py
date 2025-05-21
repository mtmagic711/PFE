import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import timm
from torchvision import transforms

from screens.autism import AutismScreen
from screens.alzheimer import AlzheimerScreen
from screens.brain_stroke import BrainStrokeScreen
from screens.ms import MScreen
from screens.parkinson import ParkinsonScreen
from screens.brain_tumor import BrainTumorScreen
from screens.base import DiseaseBaseScreen
from utils import preprocessing
from config import DISEASE_INFO, SCREEN_TO_DISEASE, MODEL_EXPLANATIONS
from models import alzheimer_cnn, alzheimer_vit, stroke_cnn, ms_cnn, ms_unet, tumor_unet, parkinson_model, autism_model
from utils.preprocessing import (
    preprocess_image,
    preprocess_stroke_image,
    preprocess_ms_image,
    preprocess_image_pytorch,
    preprocess_image_pytorch_grayscale
)

Window.clearcolor = (0.95, 0.95, 0.97, 1)

# Load the style.kv file
Builder.load_file('style_new.kv')

class ClassicalCNN(nn.Module):
    def __init__(self):
        super(ClassicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))

class QuantumLayer(nn.Module):
    def __init__(self):
        super(QuantumLayer, self).__init__()
        self.fc = nn.Linear(64, 2)  # Output for binary classification

    def forward(self, x):
        return self.fc(x)

class GCN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second conv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Third conv layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global mean pooling
        x = x.mean(dim=0)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=0).unsqueeze(0)

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.cnn = ClassicalCNN()
        self.quantum = QuantumLayer()

    def forward(self, x):
        cnn_output = self.cnn(x)
        return self.quantum(cnn_output)

class MainMenu(Screen):
    pass

class DiseaseBaseScreen(Screen):
    def get_model_explanation(self, model_name):
        # Normalize model name for lookup
        if model_name.replace("-", "").upper() == "UNET":
            return MODEL_EXPLANATIONS.get("U-Net", "")
        return MODEL_EXPLANATIONS.get(model_name, "")

    def on_enter(self):
        disease_key = SCREEN_TO_DISEASE.get(self.name, self.name)
        info = DISEASE_INFO.get(disease_key, {})
        text = f"[b]Définition :[/b]\n{info.get('definition', '')}\n\n[b]Symptômes :[/b]\n{info.get('symptoms', '')}"
        if 'disease_info' in self.ids:
            self.ids.disease_info.text = text
        # Set initial model explanation
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
        self.ids.result_label.text = "Analyse terminée ! (résultat simulé)"

class AutismScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        model_type = self.ids.model_selector.text
        if model_type == "GCN":
            # Convert image to graph data
            img = Image.open(image_path).convert('L').resize((128, 128))
            arr = np.array(img) / 255.0
            # Create graph data (simplified example)
            x = torch.tensor(arr.reshape(-1, 1), dtype=torch.float)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Example edge connections
            data = Data(x=x, edge_index=edge_index)
            
            with torch.no_grad():
                output = autism_model(data)
                pred = torch.argmax(output, dim=1).item()
            affected = pred == 1
            self.ids.result_label.text = (
                "Vous êtes atteint(e) d'autisme." if affected else "Vous n'êtes pas atteint(e) d'autisme."
            )

class AlzheimerScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        model_type = self.ids.model_selector.text
        if model_type == "CNN":
            arr = preprocess_image(image_path)
            pred = alzheimer_cnn.predict(arr)
            affected = np.argmax(pred) != 0  # 0 = not affected, others = affected
        else:  # ViT
            arr = preprocess_image_pytorch(image_path)
            with torch.no_grad():
                output = alzheimer_vit(arr)
                pred = torch.argmax(output, dim=1).item()
            affected = pred != 0  # 0 = not affected, others = affected
        self.ids.result_label.text = (
            "Vous êtes atteint(e) d'Alzheimer." if affected else "Vous n'êtes pas atteint(e) d'Alzheimer."
        )

class BrainStrokeScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        model_type = self.ids.model_selector.text
        if model_type == "CNN":
            arr = preprocess_stroke_image(image_path)
            pred = stroke_cnn.predict(arr)
            affected = np.argmax(pred) == 1
            self.ids.result_label.text = (
                "Vous êtes atteint(e) d'AVC." if affected else "Vous n'êtes pas atteint(e) d'AVC."
            )

class MScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
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

class ParkinsonScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
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

class BrainTumorScreen(DiseaseBaseScreen):
    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        model_type = self.ids.model_selector.text
        if model_type == "U-Net":
            arr = preprocess_image(image_path)
            pred = tumor_unet.predict(arr)
            affected = np.argmax(pred) == 1
            self.ids.result_label.text = (
                "Vous êtes atteint(e) d'une tumeur cérébrale." if affected else "Vous n'êtes pas atteint(e) d'une tumeur cérébrale."
            )

class DiseaseScreen(Screen):
    def __init__(self, disease_name, **kwargs):
        super().__init__(**kwargs)
        self.disease_name = disease_name

    def on_enter(self):
        self.ids.disease_title.text = self.disease_name
        self.update_disease_info()

    def update_disease_info(self):
        info = DISEASE_INFO.get(self.disease_name, {})
        text = f"[b]Définition :[/b]\n{info.get('definition', '')}\n\n[b]Symptômes :[/b]\n{info.get('symptoms', '')}"
        self.ids.disease_info.text = text

    def analyze_image(self):
        image_path = self.ids.uploaded_image.source
        try:
            arr = preprocess_image_pytorch(image_path)
            self.ids.result_label.text = "Analyse terminée ! (résultat simulé)"
        except ValueError as e:
            self.ids.result_label.text = str(e)
            return

class BrainDiseaseApp(App):
    def build(self):
        self.title = 'Analyseur Cérébral'
        sm = ScreenManager()
        sm.add_widget(MainMenu(name='menu'))
        sm.add_widget(AutismScreen(name='autism'))
        sm.add_widget(AlzheimerScreen(name='alzheimer'))
        sm.add_widget(BrainStrokeScreen(name='brain_stroke'))
        sm.add_widget(MScreen(name='ms'))
        sm.add_widget(ParkinsonScreen(name='parkinson'))
        sm.add_widget(BrainTumorScreen(name='brain_tumor'))

        for disease in DISEASE_INFO.keys():
            screen_name = disease.lower().replace(' ', '_')
            sm.add_widget(DiseaseScreen(disease, name=screen_name))

        sm.current = 'menu'
        return sm

if __name__ == '__main__':
    BrainDiseaseApp().run()