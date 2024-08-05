from tkinter import Tk
from tkinter.filedialog import askdirectory
from pathlib import Path 
import os
import pandas as pd


# Immediately hide the root window
root = Tk()
root.withdraw()


#imagePath1 = askdirectory(title='Select Folder') # shows dialog box and return the path
#imagePath2 = askdirectory(title='Select Folder') # shows dialog box and return the path
imagePathDiploid = '/Users/tomyoung/Desktop/RotatedRectangleBoundingBox/Diploid'
imagePathHaploid = '/Users/tomyoung/Desktop/RotatedRectangleBoundingBox/Haploid'
outputPath = '/Users/tomyoung/Desktop/RotatedRectangleBoundingBox/ParsedLabels/'
print(imagePathDiploid)
print(imagePathHaploid)
#print(imagePath)  
labelPathDiploid = str(imagePathDiploid) + "Labels/"
labelPathHaploid = str(imagePathHaploid) + "Labels/"

Path(labelPathDiploid).mkdir(parents=True, exist_ok=True)
Path(labelPathHaploid).mkdir(parents=True, exist_ok=True)
print(labelPathDiploid)
print(labelPathHaploid)

Path(outputPath).mkdir(parents=True, exist_ok=True)


labelFilesDiploid = os.listdir(labelPathDiploid)
labelFilesDiploid = [f for f in labelFilesDiploid if os.path.isfile(labelPathDiploid+'/'+f)] #Filtering only the files.
#print(labelFiles1)

labelFilesHaploid = os.listdir(labelPathHaploid)
labelFilesHaploid = [f for f in labelFilesHaploid if os.path.isfile(labelPathHaploid+'/'+f)] #Filtering only the files.
#print(labelFiles2)

def normalize_coordinate(coord, max_value):
    # Normalize the coordinate and clamp it between 0 and 1
    normalized = coord / max_value
    if normalized < 0:
        return 0
    elif normalized > 1:
        return 1
    return normalized

for labelFile in labelFilesDiploid:
    data = pd.read_csv(labelPathDiploid + labelFile)
    saveFile = labelFile[:-3] + 'txt'
    formatted_data = data.apply(lambda row: f"1 {normalize_coordinate(row['point1 x'], 3072)} {normalize_coordinate(row['point1 y'], 2048)} {normalize_coordinate(row['point2 x'], 3072)} {normalize_coordinate(row['point2 y'], 2048)} {normalize_coordinate(row['point3 x'], 3072)} {normalize_coordinate(row['point3 y'], 2048)} {normalize_coordinate(row['point4 x'], 3072)} {normalize_coordinate(row['point4 y'], 2048)}", axis=1)
    with open(outputPath + saveFile, 'w') as file:
        file.write("\n".join(formatted_data))

for labelFile in labelFilesHaploid:
    data = pd.read_csv(labelPathHaploid + labelFile)
    saveFile = labelFile[:-3] + 'txt'
    formatted_data = data.apply(lambda row: f"0 {normalize_coordinate(row['point1 x'], 3072)} {normalize_coordinate(row['point1 y'], 2048)} {normalize_coordinate(row['point2 x'], 3072)} {normalize_coordinate(row['point2 y'], 2048)} {normalize_coordinate(row['point3 x'], 3072)} {normalize_coordinate(row['point3 y'], 2048)} {normalize_coordinate(row['point4 x'], 3072)} {normalize_coordinate(row['point4 y'], 2048)}", axis=1)
    with open(outputPath + saveFile, 'w') as file:
        file.write("\n".join(formatted_data))