# bonus_model.py
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from trainer import Trainer
from utils import load_dataset, get_nof_params


def my_bonus_model():
    checkpoint_path = "checkpoints/fakes_dataset_Bonus_Adam.pt"
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(nn.Dropout(p=0.2),nn.Linear(in_features, 2))
    if os.path.isfile(checkpoint_path):
        print(f"Checkpoint found at: {checkpoint_path}. Loading...")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])
        print("Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found at: {checkpoint_path}. Proceeding without loading.")
    return model
