from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model choise: 

# Use the model
results = model.train(data="config.yaml", epochs=300, device=1, patience=50, verbose=True) 
# device can be: 0 (first GPU), [0,1] (first two GPU), cpu, mps (for appli silicon)
# patience is number of epochs without improvement to stop


# ## Resume Training Process ##
# model = YOLO("path/to/last.pt")  # load a partially trained model
# results = model.train(resume=True)