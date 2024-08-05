import os
import random
import supervision as sv
import cv2
from ultralytics import YOLO
import csv
from pathlib import Path 
from PIL import Image
import pprint

pp = pprint.PrettyPrinter(indent=4)

model = YOLO('./runs/obb/train7/weights/best.pt')


red_folders = os.listdir(f"../MemisData2/Red")
red_folders = [os.path.join("../MemisData2/Red/", name) for name in red_folders if name != '.DS_Store']

red_images_dict = {}
for folder in red_folders:
    if folder[0] == '.':
        pass
    red_images = os.listdir(folder)
    red_images = [os.path.join(folder, name) for name in red_images if name != '.DS_Store']
    red_images_dict[folder] = red_images

    #red_images_dict[folder] = [os.path.join("../MemisData2/Red/", name) for name in folder]

white_folders = os.listdir(f"../MemisData2/White")
white_folders = [os.path.join("../MemisData2/White/", name) for name in white_folders if name != '.DS_Store']

white_images_dict = {}
for folder in white_folders:
    if folder[0] == '.':
        pass
    white_images = os.listdir(folder)
    white_images = [os.path.join(folder, name) for name in white_images if name != '.DS_Store']

    white_images_dict[folder] = white_images

pp.pprint(red_images_dict)
pp.pprint(white_images_dict)


for folder in white_folders:
    all_test_images =white_images_dict[folder]
    for test_image in all_test_images:
        results = model(test_image)
        baseName = os.path.basename(test_image)[:-4]

        print("predicting on file " + str(baseName))
        predAnnotatePath = folder + '/PredAnnotated/'
        Path(predAnnotatePath).mkdir(parents=True, exist_ok=True)

        annotate_file_path =  folder + '/PredAnnotated/' + baseName + '.jpg'
        image_file_path =  folder + '/PredImages/' + baseName + '.jpg'
        image_file_path_base = folder + '/PredImages/'
        label_file_path =  folder + '/PredLabels/' + baseName + '.csv'
        label_file_path_base =  folder + '/PredLabels/' 

        Path(image_file_path_base).mkdir(parents=True, exist_ok=True)
        Path(label_file_path_base).mkdir(parents=True, exist_ok=True)

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            class_array = [box.cls for box in obb]

            #for box in obb:
            #    print(box.xyxyxyxy)
            #result.show()  # display to screen
            result.save(filename=annotate_file_path)  # save to disk


        image = Image.open(test_image)
        image.save(image_file_path)

        # Open the file in write mode
        with open(label_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['point1 x', 'point1 y', 'point2 x', 'point2 y', 
                            'point3 x', 'point3 y', 'point4 x', 'point4 y'])
            for box in obb:
                # Flatten the tensor and convert to list
                coords = box.xyxyxyxy.view(-1).tolist()
                # Write to the file with a newline
                writer.writerow([f'{coord:.4f}' for coord in coords])
