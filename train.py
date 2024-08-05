from ultralytics import YOLO



#model = YOLO('yolov8n-obb.yaml')
#results = model.train(data='dataset.yaml', epochs=200, imgsz=640, device='mps', degrees=90, translate=.2, scale=.1, mixup=1, batch=8)


# Load a model
model = YOLO('./runs/obb/train11/weights/last.pt')  # load a partially trained model

# Resume training
results = model.train(resume=True)
#results = model.train(data='dataset.yaml', epochs=200, imgsz=640, device='mps')
