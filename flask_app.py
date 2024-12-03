# %%
import os                       # for working with files

import numpy as np              # for numerical computationss

import torch                    # Pytorch module

import torch.nn as nn           # for creating  neural networks

from PIL import Image           # for checking images

import torch.nn.functional as F # for functions for calculating loss

import torchvision.transforms as transforms   # for transforming images into tensors


#from torchsummary import summary              # for getting the summary of our model



#%matplotlib inline

# %% [markdown]
# #### Residual Block code implementation

# %%
class SimpleResidualBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.relu2 = nn.ReLU()



    def forward(self, x):

        out = self.conv1(x)

        out = self.relu1(out)

        out = self.conv2(out)

        return self.relu2(out) + x # ReLU can be applied before or after adding the input

# %%
# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# base class for the model
class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

# %% [markdown]
# # Defining the model final architecture

# %%

#convolution block with BatchNormalization

def ConvBlock(in_channels, out_channels, pool=False):

    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

             nn.BatchNorm2d(out_channels),

             nn.ReLU(inplace=True)]

    if pool:

        layers.append(nn.MaxPool2d(4))

    return nn.Sequential(*layers)


# resnet architecture

class ResNet9(ImageClassificationBase):

    def __init__(self, in_channels, num_diseases):

        super().__init__()



        self.conv1 = ConvBlock(in_channels, 64)

        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64

        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))



        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16

        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44

        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))



        self.classifier = nn.Sequential(nn.MaxPool2d(4),

                                       nn.Flatten(),

                                       nn.Linear(512, num_diseases))



    def forward(self, xb): # xb is the loaded batch

        out = self.conv1(xb)

        out = self.conv2(out)

        out = self.res1(out) + out

        out = self.conv3(out)

        out = self.conv4(out)

        out = self.res2(out) + out

        out = self.classifier(out)

        return out

# %% [markdown]
# \# Testing model on test data

# %%
import torch

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# %%
device=get_default_device()
print(device)

# %%
# model = to_device(ResNet9(3,17), device)
# model.load_state_dict(torch.load('/Volumes/Data/Pest_disease/trained_model.pth',map_location=torch.device(device)))
     
from huggingface_hub import hf_hub_download

# Download the trained model file
model_path = hf_hub_download(
    repo_id="NLPGenius/resnet-PlantDiseaseDetect",
    filename="trained_model.pth"
)

# Load the model state dictionary
import torch
state_dict = torch.load(model_path, map_location=device)

# Load the state dictionary into your model
model = ResNet9(3, 17).to(device)
model.load_state_dict(state_dict)
# %%
# Example classes for your model
classes = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# %%
metadata = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "treatment": "Apply fungicides like azoxystrobin or tebuconazole.",
        "precautions": "Avoid over-watering and rotate crops to reduce soil-borne spores."
    },
    "Corn_(maize)___Common_rust_": {
        "treatment": "Use fungicides such as propiconazole or chlorothalonil.",
        "precautions": "Plant resistant varieties and avoid planting susceptible hybrids."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "treatment": "Use fungicides containing propiconazole, tebuconazole, or azoxystrobin.",
        "precautions": "Practice crop rotation and remove infected plant debris after harvest."
    },
    "Corn_(maize)___healthy": {
        "treatment": "No treatment needed.",
        "precautions": "Ensure optimal growing conditions and monitor for signs of disease."
    },
    "Potato___Early_blight": {
        "treatment": "Apply fungicides such as chlorothalonil or mancozeb.",
        "precautions": "Ensure proper spacing and remove infected plant debris."
    },
    "Potato___Late_blight": {
        "treatment": "Use fungicides like copper-based compounds or mefenoxam.",
        "precautions": "Avoid overhead irrigation and grow resistant varieties."
    },
    "Potato___healthy": {
        "treatment": "No treatment needed.",
        "precautions": "Monitor for early signs of blight and maintain good soil health."
    },
    "Tomato___Bacterial_spot": {
        "treatment": "Use copper-based fungicides or streptomycin.",
        "precautions": "Use resistant varieties, and avoid overhead watering."
    },
    "Tomato___Early_blight": {
        "treatment": "Use fungicides containing chlorothalonil or copper.",
        "precautions": "Ensure proper plant spacing and remove infected leaves promptly."
    },
    "Tomato___Late_blight": {
        "treatment": "Use fungicides like mancozeb or metalaxyl.",
        "precautions": "Avoid overhead watering and use disease-resistant varieties."
    },
    "Tomato___Leaf_Mold": {
        "treatment": "Use fungicides such as chlorothalonil or copper-based products.",
        "precautions": "Increase airflow and avoid wetting the foliage."
    },
    "Tomato___Septoria_leaf_spot": {
        "treatment": "Apply fungicides like chlorothalonil or copper.",
        "precautions": "Remove and destroy infected leaves and practice crop rotation."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "treatment": "Apply miticides or horticultural oils.",
        "precautions": "Maintain high humidity around plants and introduce natural predators like ladybugs."
    },
    "Tomato___Target_Spot": {
        "treatment": "Apply fungicides like azoxystrobin or boscalid.",
        "precautions": "Ensure proper spacing and remove infected leaves."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "treatment": "There is no cure; remove and destroy infected plants.",
        "precautions": "Disinfect tools and wash hands before handling plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "treatment": "There is no cure; remove and destroy infected plants.",
        "precautions": "Disinfect tools and wash hands before handling plants."
    },
    "Tomato___healthy": {
        "treatment": "No treatment needed.",
        "precautions": "Ensure optimal growing conditions and monitor for early signs of disease."
    }
}



# %%
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, allow_headers=["Content-Type"])

# Prediction function
def predict_image(img, model):
    # Resizing images, converting to tensor, and normalizing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Match training size
        transforms.ToTensor()
    ])    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get predictions from the model
    model.eval()
    with torch.no_grad():
        yb = model(img_tensor)
    
    # Apply softmax to get probabilities
    prob = nn.Softmax(dim=1)(yb)
    _, preds = torch.max(prob, dim=1)
    
    # Retrieve the class label
    predicted_class = classes[preds[0].item()]
    confidence = round(torch.max(prob).item() * 100, 4)
    
    # Map predicted class to metadata
    if predicted_class in metadata:
        treatment = metadata[predicted_class]['treatment']
        precautions = metadata[predicted_class]['precautions']
    else:
        treatment = "No treatment information available."
        precautions = "No precautions information available."
    
    return {
        "predicted_class": predicted_class,
        "confidence": f"{confidence}%",
        "treatment": treatment,
        "precautions": precautions
    }

# API endpoint
@app.route('/', methods=['GET','POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # Read image file
    file = request.files['image']
    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400
    
    # Make prediction
    try:
        result = predict_image(img, model)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run()



