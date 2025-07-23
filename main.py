from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import json, numpy as np, os
from utils import preprocess_patch_image, preprocess_whole_image

app = FastAPI()

# Allow Flutter access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MODEL_DIR = "models"

models_info = [
    {
        "name": "CNN",
        "filename": "best_model_cnn.keras",
        "class_path": "class_indices_cnn.json",
        "patch_based": False
    },
    {
        "name": "CNN + BiLSTM",
        "filename": "best_model_cnn_bilstm.keras",
        "class_path": "class_indices_cnn_bilstm.json",
        "patch_based": True
    }
]

# Load models
for model in models_info:
    model["model"] = load_model(os.path.join(MODEL_DIR, model["filename"]))
    with open(os.path.join(MODEL_DIR, model["class_path"]), 'r') as f:
        class_indices = json.load(f)
        model["class_names"] = (
            list(class_indices.keys()) if model["patch_based"]
            else {v: k for k, v in class_indices.items()}
        )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)

    results = []
    for model in models_info:
        input_data = preprocess_patch_image("temp.jpg") if model["patch_based"] else preprocess_whole_image("temp.jpg")
        prediction = model["model"].predict(input_data)
        pred_index = int(np.argmax(prediction))
        confidence = float(prediction[0][pred_index])
        label = (
            model["class_names"][pred_index]
            if isinstance(model["class_names"], list)
            else model["class_names"].get(pred_index)
        )
        results.append({
            "model": model["name"],
            "label": label,
            "confidence": round(confidence * 100, 2)
        })

    os.remove("temp.jpg")
    return {"result": results}
