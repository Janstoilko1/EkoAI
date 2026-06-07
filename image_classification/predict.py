import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
import io
import base64

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "image_model.pth"

model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=3)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

classes = checkpoint["classes"]

val_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Ni slike"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor = val_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0] * 100
        _, predicted = torch.max(output, 1)

    razred = classes[predicted.item()]
    verjetnosti = {c: round(probs[i].item(), 1) for i, c in enumerate(classes)}

    return jsonify({
        "razred": razred,
        "verjetnosti": verjetnosti
    })

if __name__ == "__main__":
    app.run(debug=True)