# Rotated Rectangle Bounding Box Annotation Tool

## How to Use

### Select Image Folder
- After running the script, you will be prompted to select a folder containing the images you wish to label
- Selecting the folder will create two directories, [foldername]Labels and [foldername]Backup
  
### Adding a Rectangle
- Press 'a'
- Click the image where you want to place the first corner of the rectangle
- Click the image where you want to place the second corner of the rectangle
- Click anywhere on the dotted yellow line to choose the length of the rectangle

### Removing a Rectangle
- Press 'x'
- Click on either of the red dots that mark the bounds of a given rectangles base

### Changing Images
- Press the left or right arrow keys

### Editing Rectangles
- Move the base line of a rectangle by clicking and dragging either of the red dots
- Resize a rectangle by clicking on one of the base line's red dots to select the rectangle, then click anywhere along the yellow line to resize it

### Recovering Labels
- Did something go wrong? I hope not. But if it did, you can restore the contents of the 'labels' folder with any of the timestamped subdirectories in the '[foldername]Backup' folder.
- Copy and paste the .csv files in from the timestamped directory of your choice into the top level 'labels' folder. 
