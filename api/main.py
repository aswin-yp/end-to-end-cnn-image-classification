from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from torchvision import transforms
from src.model import TinyVGG

app = FastAPI()

classes = ["pizza", "steak", "sushi"]

model = TinyVGG(num_classes=3)
model.load_state_dict(torch.load("cnn_model.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor).argmax(dim=1).item()

    return {"prediction": classes[pred]}
