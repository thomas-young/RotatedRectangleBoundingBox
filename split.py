
from ultralytics.data.split_dota import split_trainval, split_test

# split train and val set, with labels.
split_trainval(
    data_root='./datasets/DiploidHaploid',
    save_dir='./datasets/DiploidHaploid-split/',
    rates=[0.5, 1.0, 1.5],    # multiscale
    gap=500
)
# split test set, without labels.
split_test(
    data_root='./datasets/DiploidHaploid',
    save_dir='./datasets/DiploidHaploid-split/',
    rates=[0.5, 1.0, 1.5],    # multiscale
    gap=500
)

#model = YOLO('yolov8n-obb.yaml')
#results = model.train(data='dataset.yaml', epochs=100, imgsz=640)
