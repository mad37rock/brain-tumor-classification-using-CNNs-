!pip install split-folders
!pip install torch-summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

import copy
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torchvision import utils
from torchvision.datasets import ImageFolder
import splitfolders
from torchsummary import summary
import torch.nn.functional as F
import pathlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
from tqdm.notebook import trange
from tqdm.notebook import tqdm
from torch import optim
import warnings
warnings.filterwarnings('ignore')


array_module = np
dataframe_module = pd
plotting_module = plt
visualization_module = sns
deep_copy_module = copy
filesystem_module = os
torch_module = torch
image_module = Image
dataset_module = Dataset
torchvision_module = torchvision
transforms_module = transforms
split_function_module = random_split
scheduler_module = ReduceLROnPlateau
neural_network_module = nn
utils_module = utils
imagefolder_module = ImageFolder
splitfolders_module = splitfolders
model_summary_module = summary
functional_module = F
pathlib_module = pathlib
confusion_matrix_module = confusion_matrix
classification_report_module = classification_report
itertools_module = itertools
progress_trange_module = trange
progress_tqdm_module = tqdm
optimizer_module = optim
warnings_module = warnings

sns.set_context("notebook")
sns.set_style("darkgrid")

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np_random_state = np.random.RandomState(seed=42)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

cwd_path = os.getcwd()
root_path = pathlib.Path(cwd_path)

project_name = "ImageClassificationProject"
project_dir = root_path / project_name
os.makedirs(project_dir, exist_ok=True)

data_dir = project_dir / "dataset"
train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"

splitfolders.ratio(
    input=str(data_dir),
    output=str(project_dir),
    seed=42,
    ratio=(0.7, 0.2, 0.1),
    move=False
)

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = ImageFolder(root=str(train_dir), transform=transform_pipeline)
val_dataset = ImageFolder(root=str(val_dir), transform=transform_pipeline)
test_dataset = ImageFolder(root=str(test_dir), transform=transform_pipeline)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

class_names = train_dataset.classes
num_classes = len(class_names)

class CustomCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = CustomCNNModel(num_classes=num_classes)
model = model.to(device)

summary(model, input_size=(3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

num_epochs = 25

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_train += labels.size(0)

    epoch_loss = running_loss / total_train
    epoch_acc = running_corrects.double() / total_train
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())

    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
            total_val += labels.size(0)

    val_loss = val_running_loss / total_val
    val_acc = val_running_corrects.double() / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
