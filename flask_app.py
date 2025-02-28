import os                       # for working with files

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
# import subprocess
# import sys

# try:
#     from huggingface_hub import hf_hub_download
# except ImportError:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
#     from huggingface_hub import hf_hub_download

from huggingface_hub import hf_hub_download

# Download the trained model file
model_path = hf_hub_download(
    repo_id="NLPGenius/DiseaseDetect-WheatRice-Resnet",
    filename="resnet9_WheatRice.pth"
)

# Load the model state dictionary
import torch
state_dict = torch.load(model_path, map_location=device)

# Load the state dictionary into your model
model = ResNet9(3, 17).to(device)
model.load_state_dict(state_dict)
# %%
# Example classes for your model
classes = ['Rice_Bacterialblight', 'Rice_Blast', 'Rice_Brownspot', 'Rice_Tungro', 'unknown', 'wheat_Aphid', 'wheat_Brown Rust', 'wheat_Common Root Rot', 'wheat_Fusarium Head Blight', 'wheat_Healthy', 'wheat_Mildew', 'wheat_Mite', 'wheat_Septoria', 'wheat_Smut', 'wheat_Stem fly', 'wheat_Tan spot', 'wheat_Yellow Rust']


# %%
metadata = {
    "Rice_Bacterialblight": {
        "treatment": "Use resistant rice varieties with known resistance genes such as Xa4, xa5, xa13, and Xa21. Apply balanced fertilizers and ensure proper water management.",
        "precautions": "Use certified disease-free seeds, maintain field sanitation, and avoid excessive irrigation.",
        "biological_solutions": "Utilize antagonistic bacteria that produce antimicrobial compounds, and apply fungal metabolites from fungi like Paraphaeosphaeria minitans."
    },
    "Rice_Blast": {
        "treatment": "Use resistant rice varieties, apply balanced fertilizers, and maintain proper field flooding.",
        "precautions": "Remove and destroy crop residues and weeds, and avoid excessive nitrogen application.",
        "biological_solutions": "Apply beneficial bacteria like Pseudomonas fluorescens to inhibit the growth of Magnaporthe oryzae, and use organic neem-based sprays."
    },
    "Rice_Brownspot": {
        "treatment": "Plant resistant rice varieties, use certified disease-free seeds, and apply balanced fertilizers.",
        "precautions": "Ensure proper field sanitation, maintain balanced soil fertility, and avoid excessive nitrogen application.",
        "biological_solutions": "Apply beneficial fungi such as Trichoderma viride and Trichoderma harzianum to inhibit the growth of Bipolaris oryzae, and use organic neem-based sprays."
    },
    "Rice_Tungro": {
        "treatment": "Cultivate resistant rice varieties, manage leafhopper populations, and implement synchronized planting.",
        "precautions": "Remove infected plant debris and volunteer rice plants, and maintain ecological balance in the field.",
        "biological_solutions": "Encourage natural predators of leafhoppers, like spiders and certain beetles, and apply endophytic bacteria like Streptomyces species to suppress tungro disease."
    },
    "wheat_Aphid": {
        "treatment": "Monitor aphid populations and apply insecticides if necessary. Use natural predators like lady beetles and lacewings to reduce aphid numbers.",
        "precautions": "Regularly inspect wheat fields for aphid presence, especially during early growth stages. Implement action thresholds for control measures.",
        "biological_solutions": "Encourage natural aphid predators such as lady beetles, lacewings, and parasitic wasps. Implement crop rotation and maintain optimal plant health through proper fertilization and irrigation."
    },
    "wheat_Brown_Rust": {
        "treatment": "Use fungicides such as systemic seed treatments and foliar fungicides to control brown rust. Apply fungicides around flag leaf emergence for effective control.",
        "precautions": "Cultivate resistant wheat varieties and monitor for early signs of brown rust. Implement action thresholds for fungicide applications.",
        "biological_solutions": "Research biological control agents like Bacillus subtilis for suppressing rust diseases. Crop rotation and avoiding excessive nitrogen can help mitigate disease severity."
    },
    "wheat_Common Root Rot": {
        "treatment": "Apply fungicidal seed treatments to protect seedlings. Use fungicides labeled for common root rot control and consult local guidelines.",
        "precautions": "Implement crop rotation with non-host crops and utilize resistant wheat varieties. Maintain proper plant health through fertilization and irrigation.",
        "biological_solutions": "Use beneficial fungi like Trichoderma species (e.g., T. viride, T. harzianum) and beneficial bacteria like Bacillus subtilis and Pseudomonas fluorescens as seed treatments or soil amendments."
    },
    "wheat_Fusarium Head Blight": {
        "treatment": "Apply fungicides at the flowering stage to control Fusarium Head Blight (FHB). Follow local guidelines and use fungicides labeled for FHB control.",
        "precautions": "Plant resistant wheat varieties and avoid planting wheat after corn or other cereals. Tilling residues into the soil reduces inoculum levels.",
        "biological_solutions": "Use antagonistic microorganisms like Bacillus velezensis and Clonostachys rosea. Apply these biological control agents during flowering to reduce disease incidence."
    },
    "wheat_Healthy": {
        "treatment": "No treatment needed.",
        "precautions": "Use well-balanced fertilizers and ensure proper irrigation. Avoid over-crowding to reduce disease risks.",
        "biological_solutions": "Introduce beneficial soil organisms like Trichoderma spp. to improve soil health and suppress pathogens. Incorporate organic matter and practice intercropping with legumes."
    },
    "wheat_Mildew": {
        "treatment": "Apply chemical fungicides at early signs of infection, particularly when white spots appear. Use resistant wheat varieties and early planting to reduce mildew incidence.",
        "precautions": "Maintain proper crop rotation and ensure proper plant spacing to improve airflow and reduce mildew growth.",
        "biological_solutions": "Use biological control agents like Ampelomyces quisqualis, a naturally occurring fungus that parasitizes powdery mildew. Introduce beneficial microorganisms like Bacillus subtilis and incorporate organic amendments like compost."
    },
    "wheat_Tan_spot": {
        "treatment": "Timely planting of wheat, apply systemic insecticides during early crop growth if infestation is suspected, and practice proper crop rotation. Remove infected or dead plants regularly.",
        "precautions": "Monitor fields regularly, especially during early growth stages. Delay planting to avoid peak fly activity, maintain proper field sanitation, and destroy crop residues after harvest.",
        "biological_solutions": "Introduce natural predators like parasitic wasps (Pteromalus spp.) to target stem fly larvae. Use beneficial nematodes (Steinernema and Heterorhabditis spp.) to target stem fly larvae in the soil. Incorporate biocontrol agents like Bacillus thuringiensis (Bt). Practice trap cropping and promote biodiversity within the farming ecosystem."
    },
    "wheat_Yellow Rust": {
        "treatment": "Plant resistant or tolerant wheat varieties. Apply fungicides like triazoles or strobilurins at the first signs of infection, preferably preventatively or at early stages.",
        "precautions": "Monitor weather conditions, especially during cool, wet periods (10-15°C). Practice crop rotation, remove and destroy infected crop residues after harvest.",
        "biological_solutions": "Introduce natural predators like Amblyseius mites or Phytoseiulus persimilis. Use biocontrol agents like Trichoderma spp. Incorporate beneficial microorganisms such as Bacillus subtilis or Pseudomonas fluorescens. Plant trap crops or intercropping with non-grass species."
    },
    "Wheat_Septoria": {
        "treatment": "Apply fungicides at key growth stages, particularly between stem elongation and flowering. Fungicides containing active ingredients such as azoles (e.g., prothioconazole) or SDHIs (e.g., bixafen) are commonly used. Monitor local guidelines for fungicide resistance management and adhere to recommended application rates and timings.",
        "precautions": "Select wheat varieties with partial resistance to Septoria tritici. Implement crop rotation with non-host species to reduce inoculum levels. Delay sowing dates to avoid peak periods of spore dispersal. Ensure adequate plant spacing to promote air circulation and reduce leaf wetness duration.",
        "biological_solutions": "Certain bacterial species, such as Bacillus megaterium, can reduce disease severity significantly. Pseudomonas species also show potential in suppressing Septoria tritici through antagonistic interactions."
    },
    "Wheat_Mite": {
        "treatment": "Apply miticides such as those containing abamectin or spiromesifen. Ensure timely application to control the infestation before it spreads significantly. Regular monitoring and early intervention are key to effective control.",
        "precautions": "Use resistant wheat varieties where available. Avoid planting wheat near recently harvested fields to reduce the chances of mite migration. Maintain crop rotation and proper spacing to improve airflow and disrupt mite habitats.",
        "biological_solutions": "Encourage natural predators, such as predatory mites (Amblyseius spp.) and lady beetles, which help keep mite populations in check. Additionally, apply neem oil or other bio-pesticides to suppress mite activity in an eco-friendly manner."
    },
    "Wheat_Stem Fly": {
        "treatment": "Apply insecticides such as chlorpyrifos or imidacloprid to control stem fly populations. Use systemic insecticides at the seedling stage for better protection.",
        "precautions": "Practice early sowing to avoid peak activity periods of stem flies. Use crop rotation with non-host crops to reduce the build-up of stem fly populations. Remove and destroy stubble and infested plants after harvest to eliminate breeding sites.",
        "biological_solutions": "Introduce natural predators such as parasitic wasps (Tetrastichus spp.) and ground beetles to manage stem fly larvae. Maintain field margins with flowering plants to support beneficial insect populations."
    },
    "Wheat_Smut": {
        "treatment": "Treat seeds with systemic fungicides such as carboxin, tebuconazole, or triticonazole before planting. These fungicides target smut spores and prevent infection.",
        "precautions": "Use certified disease-free seeds to minimize the risk of smut. Avoid planting infected seeds or using untreated seeds from a previous harvest. Implement crop rotation to reduce pathogen buildup in the soil and break the disease cycle.",
        "biological_solutions": "Apply biocontrol agents like Trichoderma harzianum or Pseudomonas fluorescens to seeds or soil. These beneficial microorganisms inhibit the growth of smut pathogens and reduce disease incidence naturally."
    }
}



# %%
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import OrderedDict

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, allow_headers=["Content-Type"])

from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Prediction function
def predict_image(img, model):

    if img.mode != "RGB":
        img = img.convert("RGB")
    
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
    #confidence = round(torch.max(prob).item() * 100, 4)
    
    # Map predicted class to metadata
    if predicted_class in metadata:
        treatment = metadata[predicted_class]['treatment']
        precautions = metadata[predicted_class]['precautions']
    else:
        treatment = "No treatment information available."
        precautions = "No precautions information available."
    
    return OrderedDict([
        ("predicted_class", predicted_class),
        #("confidence", f"{confidence}%"),
        ("treatment", treatment),
        ("precautions", precautions)
    ])

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
