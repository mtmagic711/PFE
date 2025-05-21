from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import timm

# Model class definitions
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
        self.fc = nn.Linear(64, 2)

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
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = x.mean(dim=0)
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

# Model loading
alzheimer_cnn = load_model('models/Alzheimer_CNN.h5', compile=False)
alzheimer_vit = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,
    num_classes=4
)
alzheimer_vit.load_state_dict(torch.load('models/vit_Alzheimer.pth', map_location=torch.device('cpu')))
alzheimer_vit.eval()
stroke_cnn = load_model('models/stroke_model.h5', compile=False)
ms_cnn = load_model('models/CNN_MS.h5', compile=False)
ms_unet = load_model('models/unet_MS.h5', compile=False)
tumor_unet = load_model('models/Brain_Tumor_U-NET.h5', compile=False)

parkinson_model = HybridModel()
parkinson_model.load_state_dict(torch.load('models/parkinson.pth', map_location=torch.device('cpu')))
parkinson_model.eval()

autism_model = GCN(input_dim=5, hidden_dim=64, output_dim=2)
autism_model.load_state_dict(torch.load('models/autism_GCN.pth', map_location=torch.device('cpu')))
autism_model.eval() 