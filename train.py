import os
import torch
from ultralytics import YOLO
from roboflow import Roboflow

# --------------------------------------------------
# CONFIG
# --------------------------------------------------


# --------------------------------------------------
# Device
# --------------------------------------------------
device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
rf = Roboflow(api_key="TuxgCnWHI3urT9Ux1yNf")
project = rf.workspace("test-taqza").project("unified_chemistry_tools")
version = project.version(1)
dataset = version.download("yolo26")

# --------------------------------------------------
# Model
# --------------------------------------------------
model = YOLO("yolo26m.pt")


# --------------------------------------------------
# Train
# --------------------------------------------------
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=80,
    imgsz=640,
    batch=32,
    device=device,
    optimizer="SGD",
    patience=10,
    exist_ok=True,
    name="yolo_runs",
    workers=8,
    
)

print("Training Complete.")