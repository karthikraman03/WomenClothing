from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet15

classes = ['white_pants', 'black_pants', 'white_shoes', 'brown_shoes', 'blue_shirt', 'green_shoes', 'blue_pants', 'white_shorts', 'red_pants', 'blue_shoes', 'black_dress', 'green_pants', 'black_shorts', 'brown_shorts', 'red_shoes', 'blue_dress', 'black_shirt', 'red_dress', 'green_shirt', 'white_dress', 'green_shorts', 'blue_shorts', 'brown_pants', 'black_shoes']


model_path = 'plant-disease-model.pth'
models = ResNet15(3, len(classes))
models.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
models.eval()


def predict_image(img, model=models):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = models(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = classes[preds[0].item()]
    # Retrieve the class label
    return prediction


@app.route('/predict', methods=['GET', 'POST'])
def clothes_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('index.html')
        try:
            img = file.read()

            prediction = predict_image(img)

            return render_template('result.html', prediction=prediction)
        except:
            pass
        
        return render_template('result.html')
    
# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)

