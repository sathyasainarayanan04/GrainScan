from flask import Flask, request, render_template
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import gdown
import os

app = Flask(__name__)

# Define the LORALayer class
class LORALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LORALayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# Load a pre-trained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights='IMAGENET1K_V1')

# Replace the last fully connected layer with LORALayer for adaptation
model.fc = LORALayer(in_features=model.fc.in_features, out_features=1, rank=32)  # Adjust rank as necessary
model = model.to(device)

# Function to download weights using gdown
def download_weights():
    model_path = 'model_weights.pth'
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        file_id = "18bcak1hJkbtRf2lDLd274wCzEkWjN08Q"  # Replace with your actual file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
        print("Download complete!")
    return model_path

# Load the model weights
model_path = download_weights()
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Target size for resizing the uploaded images
target_size = (255, 255)  # Match the training size

# Image preprocessing function
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img.to(device)

# Adapted Prediction function
def predict(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output)  # Apply sigmoid for binary classification
        predicted_label = torch.round(prediction).item()  # Get 0 or 1 as the predicted label
    return predicted_label

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file part is in the request
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        
        # Check if the file has a name
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        # Process if the file is valid
        if file:
            img = Image.open(file.stream).convert('RGB')  # Ensure 3 channels
            img_tensor = preprocess_image(img)
            
            # Make prediction
            predicted_label = predict(img_tensor)
            
            # Render result based on prediction
            if predicted_label == 1:
                message = 'Prediction: Unhealthy'
            else:
                message = 'Prediction: Healthy'
            
            return render_template('index.html', message=message)
        
    return render_template('index.html', message='Upload an image')

if __name__ == '__main__':
    app.run(debug=True)
