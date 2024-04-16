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
class CNN(pl.LightningModule):
    def __init__(self, width:int=227, height:int=227):
        super().__init__()
        def layer(
                input_channel: int, output_channel: int, kernel_size_conv: int=3, padding: str='same', 
                stride_conv: int=1, kernel_size_maxpool: int=2, stride_maxpool: int=2, normalize=True, maxpool=True
                ):
            layers = [nn.Conv2d(input_channel, 
                                output_channel, 
                                kernel_size_conv,
                                stride_conv,
                                padding
                                )]
            if normalize:
                layers.append(nn.BatchNorm2d(output_channel)) # type: ignore
            layers.append(nn.ReLU()) # type: ignore
            if maxpool:
                layers.append(nn.MaxPool2d(kernel_size=kernel_size_maxpool, stride=stride_maxpool)) # type: ignore
            return layers
        
        self.model = nn.Sequential(
                        *layer(input_channel=1, output_channel=8),
                        *layer(input_channel=8, output_channel=16),
                        *layer(input_channel=16, output_channel=32, maxpool=False)
                    )
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear((width//4) * (height//4) * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
#         print(type(x))
        x = self.model(x.float())
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
#         print(x)
        x = self.fc3(x)   
        x = self.softmax(x)
        return x

class FullModel(pl.LightningModule):
    def __init__(self, 
                    lr: float, 
                    total_steps: int,
                    width:int=227,
                    height:int=227
                ):
        super().__init__()
        self.lr = lr
        self.model = CNN(width=width, height=height)
        self.criterion = nn.CrossEntropyLoss()
        self.total_steps = total_steps
        self.save_hyperparameters()
        self.dem = 0
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), weight_decay=1e-4)
        #optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        #optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=self.total_steps, verbose=False,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [optimizer], [scheduler]
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        outputs = self.model(img)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss.item())
        #Accuracy
        output = torch.argmax(outputs, dim=1)
        correct = (output == labels).float().sum()
        self.log("train_loss", loss.item())
        self.log("train_acc", correct/len(labels))
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        outputs = self.model(img)
        loss = self.criterion(outputs, labels)
        #Accuracy
        output = torch.argmax(outputs, dim=1)
        correct = (output == labels).float().sum()
        self.log("val_loss", loss.item())
        self.log("val_acc", correct/len(labels))
        return loss

    def test_step(self, batch, batch_idx):
        img, labels = batch
        outputs = self.model(img)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss.item())
        self.log("test_loss", loss.item())
        self.log("test_acc", acc.item())  # type: ignore
        return loss

# Define a function to preprocess the input image
def preprocess_mammo_image(image_path):
    transform = Compose([
        Resize((227, 227)),
        ToTensor(),
        # Add any other transformations you used during training (e.g., normalization)
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # type: ignore
    return image

# Load the saved model checkpoint
model_mammo = FullModel.load_from_checkpoint("extra/model_ver4.ckpt")

# Set the model to evaluation mode
model_mammo.eval()


# Define a function to perform inference on a single image
def predict_mammographical(image_path):
    # Preprocess the input image
    image_tensor = preprocess_mammo_image(image_path)

    # Perform inference
    with torch.no_grad():
        output = model_mammo(image_tensor)

    # Interpret the model's output
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    # Map the predicted class to human-readable labels    
    label = 'Image has cancer' if predicted_class == 1 else 'Image has no cancer'

    return label

