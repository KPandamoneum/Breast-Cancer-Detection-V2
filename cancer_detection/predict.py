from PIL import Image

import torch
import torch.nn as nn
from torchvision.transforms import Resize, Compose, ToTensor
import pytorch_lightning as pl


# Histo Vs Random
model_histo_V_rando = "extra/histo_V_rando_model.pth"
transform_histo_V_rando = "extra/histo_V_rando_transform.pth"

def predict_histo_vs_rando(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define data transformation
    transform = torch.load(transform_histo_V_rando)  # Add batch dimension

    # Load the model
    model = torch.load(model_histo_V_rando)
    model.eval()

    # Apply the transformation to the image
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Interpret the model's output
    predicted_class = output.argmax().item()

    # Map the predicted class to human-readable labels
    label = 'Histopathological Image' if predicted_class == 0 else 'Not Histopathological Image'

    return label


# Histopathological
model_histo_path = "extra/breast_cancer_model_full.pth"
transform_histo_path = "extra/breast_cancer_transform.pth"


def predict_histopathological(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define data transformation
    transform = torch.load(transform_histo_path)  # Add batch dimension

    # Load the model
    model = torch.load(model_histo_path)
    model.eval()

    # Apply the transformation to the image
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass to get the predictions
        output = model(image)

    # Interpret the model's output
    predicted_class = output.argmax().item()

    # Map the class index to the corresponding label (e.g., cancer or non-cancer)
    label = "Image has cancer" if predicted_class == 1 else "Image has no cancer"

    return label


# Mammo vs Rando
model_mammo_V_rando = "extra/mammo_V_rando_model.pth"
transform_mammo_V_rando = "extra/mammo_V_rando_transform.pth"


def predict_mammo_vs_rando(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define data transformation
    transform = torch.load(transform_mammo_V_rando)  # Add batch dimension

    # Load the model
    model = torch.load(model_mammo_V_rando)
    model.eval()

    # Apply the transformation to the image
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Interpret the model's output
    predicted_class = output.argmax().item()

    # Map the predicted class to human-readable labels    
    label = 'Not Mammographical or Histopathological Image' if predicted_class == 1 else 'Mammographical Image'

    return label


# Mammographical
model_mammo_path = "extra/mammography.pth"
transform_mammo_path = "extra/mammography_transform.pth"

def predict_mammographical(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define data transformation
    transform = torch.load(transform_mammo_path)  # Add batch dimension

    # Load the model
    model = torch.load(model_mammo_path)
    model.eval()

    # Apply the transformation to the image
    image = transform(image).unsqueeze(0).to(next(model.parameters()).device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Interpret the model's output
    predicted_class = output.argmax().item()

    # Map the predicted class to human-readable labels    
    label = 'Image has cancer' if predicted_class == 1 else 'Image has no cancer'

    return label

