from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.lines import Line2D
import csv 
from pathlib import Path 
from tkinter.filedialog import askdirectory
import logging
from scipy.stats import normaltest
from tkinter import Tk
from mpl_toolkits.mplot3d import Axes3D
import re
import pandas as pd
# Immediately hide the root window
root = Tk()
root.withdraw()

TopLevelPath = askdirectory(title='Select Folder Top Level Folder') # shows dialog box and return the path
imageTestPathRed = str(TopLevelPath) + "/Red/"
imageTestPathWhite = str(TopLevelPath) + "/White/"

RedImageFolders = [os.path.join(imageTestPathRed, item + "/PredImages/") for item in os.listdir(imageTestPathRed) if os.path.isdir(os.path.join(imageTestPathRed, item))]
WhiteImageFolders = [os.path.join(imageTestPathWhite, item + "/PredImages/") for item in os.listdir(imageTestPathWhite) if os.path.isdir(os.path.join(imageTestPathWhite, item))]

RedImageFolders = [path for path in RedImageFolders if "DS_Store" not in path]
WhiteImageFolders = [path for path in WhiteImageFolders if "DS_Store" not in path]

print(RedImageFolders)
print(WhiteImageFolders)

index = ["count", "mean", "std_dev", "min", "25%", "50%", "75%", "max", "median"]

# Create a DataFrame
summaryStatDF = pd.DataFrame(index=index)


# Function to calculate the statistics
def calculate_statistics(stomata_sizes):
    total_number = len(stomata_sizes)
    mean_size = np.mean(stomata_sizes)
    std_dev = np.std(stomata_sizes)
    lower_quartile = np.percentile(stomata_sizes, 25)
    median = np.median(stomata_sizes)
    upper_quartile = np.percentile(stomata_sizes, 75)
    max_size = np.max(stomata_sizes)
    return {
        "total_number": total_number,
        "mean_size": mean_size,
        "std_dev": std_dev,
        "lower_quartile": lower_quartile,
        "median": median,
        "50%_value": median,
        "upper_quartile": upper_quartile,
        "max_size": max_size
    }

def extract_identifier(path):
    match = re.search(r'PHI-\d+', path)
    return match.group() if match else None

def extract_study_name(path):
    match = re.search(r'PHI-\d+', path)
    return match.group() if match else "Unknown_Study"

def extract_study_name_full(path):
    match = re.search(r'(Red_PHI-\d+|White_PHI-\d+)', path)
    return match.group() if match else None

def extract_color_name(path):
    # Split the path by slashes
    parts = path.split('/')
    
    # The color name is the third element in the split path
    color_name = parts[5]
    
    return color_name


# Sort both lists by the identifier
RedImageFolders = sorted(RedImageFolders, key=extract_identifier)
WhiteImageFolders = sorted(WhiteImageFolders, key=extract_identifier)

for imageTestPath1, imageTestPath2 in zip(RedImageFolders, WhiteImageFolders):
    print(imageTestPath1)
    print(imageTestPath2)
    print(f"Match: {extract_identifier(imageTestPath1) == extract_identifier(imageTestPath2)}")
    if extract_identifier(imageTestPath1) != extract_identifier(imageTestPath2):
        test = 1/0


    #imageTestPath1 = askdirectory(title='Select Folder RED Test Images') # shows dialog box and return the path
    #imageTestPath2 = askdirectory(title='Select Folder WHITE Test Images') # shows dialog box and return the path

    #imageTrainPath1 = askdirectory(title='Select Folder Diploid Train Images') # shows dialog box and return the path
    #imageTrainPath2 = askdirectory(title='Select Folder Haploid Train Images') # shows dialog box and return the path

    #labelTrainPath1 = str(imageTrainPath1) + "Labels/"
    #labelTrainPath2 = str(imageTrainPath2) + "Labels/"

    #print(imagePath)  
    labelTestPath1 = str(imageTestPath1[:-7]) + "Labels/"
    labelTestPath2 = str(imageTestPath2[:-7]) + "Labels/"
    print(labelTestPath1)
    print(labelTestPath2)

    Path(labelTestPath1).mkdir(parents=True, exist_ok=True)
    Path(labelTestPath2).mkdir(parents=True, exist_ok=True)
    #Path(labelTrainPath1).mkdir(parents=True, exist_ok=True)
    #Path(labelTrainPath2).mkdir(parents=True, exist_ok=True)

    #files1 = os.listdir(imageTrainPath1)
    #files1 = [f for f in files1 if os.path.isfile(imageTrainPath1+'/'+f) and f != '.DS_Store'] #Filtering only the files.


    #files2 = os.listdir(imageTrainPath2)
    #files2 = [f for f in files2 if os.path.isfile(imageTrainPath2+'/'+f and f) != '.DS_Store'] #Filtering only the files.

    filesTest1 = os.listdir(imageTestPath1)
    filesTest1 = [f for f in filesTest1 if os.path.isfile(imageTestPath1+'/'+f) and f != '.DS_Store'] #Filtering only the files.


    filesTest2 = os.listdir(imageTestPath2)
    filesTest2 = [f for f in filesTest2 if os.path.isfile(imageTestPath2+'/'+f) and f != '.DS_Store'] #Filtering only the files.


    class Rectangle():

        def __init__(self, line1, line2, line3, line4):
            self.lines = [line1, line2, line3, line4]
            self.points = self.setPointsFromLines()
            self.currBaseLineXs = line1.get_xdata()
            self.currBaseLineYs = line1.get_ydata()
            self.currSlope = self.getCurrSlope()
            self.perpLineLength = self.getPerpLineLength()

        def getLength(self):
            # Assuming points are [top-left, top-right, bottom-right, bottom-left]
            # Calculate widths and heights of the rectangle using distance formula
            width = np.sqrt((self.points[1][0] - self.points[0][0])**2 + (self.points[1][1] - self.points[0][1])**2)
            height = np.sqrt((self.points[3][0] - self.points[0][0])**2 + (self.points[3][1] - self.points[0][1])**2)

            # Return the longer side as the length
            return max(width, height)
        
        def aspect_ratio(self):
            # Assuming points are [top-left, top-right, bottom-right, bottom-left]
            # Calculate widths and heights of the rectangle using distance formula
            width = np.sqrt((self.points[1][0] - self.points[0][0])**2 + (self.points[1][1] - self.points[0][1])**2)
            height = np.sqrt((self.points[3][0] - self.points[0][0])**2 + (self.points[3][1] - self.points[0][1])**2)
            
            # Ensure width is the longer side
            if width < height:
                width, height = height, width
            
            # Return aspect ratio
            return width / height if height != 0 else 0

        def getRectangleCSVLine(self):
            allPoints = []
            for point in self.points:
                allPoints.append(point[0])
                allPoints.append(point[1])
            return allPoints

        def calculateArea(self):
            # Calculate the area using the Shoelace formula
            x = [p[0] for p in self.points]
            y = [p[1] for p in self.points]
            return 0.5 * np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

        def setPointsFromLines(self):
            point1 = [self.lines[0].get_xdata()[0], self.lines[0].get_ydata()[0]]
            point2 = [self.lines[0].get_xdata()[1], self.lines[0].get_ydata()[1]]
            point3 = [self.lines[1].get_xdata()[1], self.lines[1].get_ydata()[1]]
            point4 = [self.lines[2].get_xdata()[1], self.lines[2].get_ydata()[1]]
            return [point1, point2, point3, point4]


        def getCurrSlope(self):
            xsCurr = list(self.lines[0].get_xdata())
            ysCurr = list(self.lines[0].get_ydata())
            deltaXCurr = xsCurr[1] - xsCurr[0]
            deltaYCurr = ysCurr[1] - ysCurr[0]

            baseLineSlopeCurr = None
            if deltaXCurr != 0:
                baseLineSlopeCurr = deltaYCurr/deltaXCurr
            else:
                baseLineSlopeCurr = 999999999
            return baseLineSlopeCurr

        def getPerpLineLength(self):
            xs = list(self.lines[1].get_xdata())
            ys = list(self.lines[1].get_ydata())
            deltaX = xs[1] - xs[0]
            deltaY = ys[1] - ys[0]
            return math.sqrt(deltaX**2 + deltaY**2)

    class LineBuilder():

        def __init__(self, imageName, number):
            self.imageName = imageName
            self.lines = []
            self.xs = []
            self.ys = []
            self.currLineSlope = None
            self.currentLineInvNegSlope = None
            self.currLine = None
            self.rectDict = {}
            self.loadRectangles(number)

        def addLine(self, x, y):
            if len(x) != 2 or len(y) != 2:
                return
            line = Line2D(x, y, marker='o', markerfacecolor='red')
            self.currLine = line
            self.lines.append(line)
            self.xs.append(list(line.get_xdata()))
            self.ys.append(list(line.get_ydata()))
    
            
        def loadRectangles(self, number):
            filename, file_extension = os.path.splitext(self.imageName)
            if number == 1:
                csvName = labelTrainPath1 + filename + ".csv"
            elif number == 2:
                csvName = labelTestPath1 + filename + ".csv"
            elif number == 3:
                csvName = labelTrainPath2 + filename + ".csv"
            elif number == 4:
                csvName = labelTestPath2 + filename + ".csv"

            #print(csvName)
            try:
                with open(csvName, 'r', newline='') as csvFile:
                    reader = csv.reader(csvFile)
                    for row in reader:
                        if row[0] == 'point1 x':
                            continue
                        line2 = Line2D([float(row[2]), float(row[4])], [float(row[3]),float(row[5])], color="blue")
                        line3 = Line2D([float(row[4]), float(row[6])], [float(row[5]),float(row[7])], color="blue")
                        line4 = Line2D([float(row[6]), float(row[0])], [float(row[7]),float(row[1])], color="blue")
                        self.addLine([float(row[0]), float(row[2])], [float(row[1]),float(row[3])])
                        rectangle = Rectangle(self.currLine, line2, line3, line4)
                        self.rectDict[self.currLine] = rectangle
            except FileNotFoundError:
                logging.error(f"Could not find the file: {csvName}")
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")

        def average_aspect_ratio(self):
            aspect_ratios = [rect.aspect_ratio() for rect in self.rectDict.values()]
            return np.mean(aspect_ratios) if aspect_ratios else 0
        
        def countLabelsArea(self):

            #print(f"Reading {len(self.rectDict)} labels.")
            rectangleAreas = []
            for rect in self.rectDict.values():
                #print(f"Rectangle size: {rect.calculateArea()} square units.")
                rectangleAreas.append(rect.calculateArea())

            return (len(self.rectDict), rectangleAreas)
            
        def countLabelsLen(self):
            rectanlgeLenths = []
            for rect in self.rectDict.values():
                rectanlgeLenths.append(rect.getLength())
            return rectanlgeLenths

    def calculate_averages(lineBuilders):
        num_rectangles = []
        average_sizes = []
        for builder in lineBuilders:
            _, rectangle_areas = builder.countLabelsArea()
            if rectangle_areas:  # Avoid division by zero
                num_rectangles.append(len(rectangle_areas))
                if len(rectangle_areas) == 0:
                    print("!!! No rectangles found in " + builder.ImageName)
                average_sizes.append(np.mean(rectangle_areas))
            else:
                num_rectangles.append(0)
                average_sizes.append(0)
        return num_rectangles, average_sizes



    #lineBuilders1 = []
    #for file in files1:
    #    lineBuilders1.append(LineBuilder(file, 1))

    lineBuildersTest1 = []
    for file in filesTest1:
        lineBuildersTest1.append(LineBuilder(file, 2))

    #numImages1 = len(files1)

    '''
    all_rectangle_areas1 = []
    total_num_labels1 = 0
    for builder in lineBuilders1:
        numLabels1, rectangle_areas1 = builder.countLabelsArea()
        rectangle_lengths1 = builder.countLabelsLen()
        print(builder.imageName)
        print(rectangle_lengths1)
        total_num_labels1 += numLabels1
        all_rectangle_areas1.extend(rectangle_areas1)
    '''

    numImagesTest1 = len(filesTest1)
    all_rectangle_areasTest1 = []
    total_num_labelsTest1 = 0
    all_rectangle_lengthsTest1 = []
    all_rectangle_densityTest1 = []
    for builder in lineBuildersTest1:
        numLabelsTest1, rectangle_areasTest1 = builder.countLabelsArea()
        rectangle_lengthsTest1 = builder.countLabelsLen()
        #print(builder.imageName)
        #print(rectangle_lengthsTest1)
        rectangle_densityTest1 = numLabelsTest1 / (1536 * 1024)
        all_rectangle_densityTest1.append(rectangle_densityTest1)
        total_num_labelsTest1 += numLabelsTest1
        all_rectangle_lengthsTest1 += rectangle_lengthsTest1
        all_rectangle_areasTest1.extend(rectangle_areasTest1)


    print(f"Total rectangle areas from all images: {len(all_rectangle_areasTest1)}")
    # Continue from your existing code where you aggregate all_rectangle_areas and total_num_labels

    # Calculate the average number of rectangles per image
    average_num_rectangles_per_image1 = total_num_labelsTest1 / len(filesTest1) if len(filesTest1) > 0 else 0

    # Calculate the average size of rectangles
    average_rectangle_size1 = np.mean(all_rectangle_areasTest1) if all_rectangle_areasTest1 else 0

    # Calculate the standard deviation of rectangle sizes
    std_dev_rectangle_size1 = np.std(all_rectangle_areasTest1) if all_rectangle_areasTest1 else 0

    print(f"Average number of rectangles per image: {average_num_rectangles_per_image1}")
    print(f"Average rectangle size: {average_rectangle_size1} square units")
    print(f"Standard deviation of rectangle sizes: {std_dev_rectangle_size1} square units")

    #lineBuilders2 = []
    #for file in files2:
    #    lineBuilders2.append(LineBuilder(file, 3))


    lineBuildersTest2 = []
    for file in filesTest2:
        lineBuildersTest2.append(LineBuilder(file, 4))


    '''

    numImages2 = len(files2)
    print(f"Total rectangle areas from all images: {numImages2}")
    all_rectangle_areas2 = []
    total_num_labels2 = 0
    for builder in lineBuilders2:
        numLabels2, rectangle_areas2 = builder.countLabelsArea()
        rectangle_lengths2 = builder.countLabelsLen()
        print(builder.imageName)
        print(rectangle_lengths2)
        total_num_labels2 += numLabels2
        all_rectangle_areas2.extend(rectangle_areas2)
    '''

    numImagesTest2 = len(filesTest2)

    print(f"Total rectangle areas from all images: {numImagesTest2}")
    all_rectangle_areasTest2 = []
    total_num_labelsTest2 = 0
    all_rectangle_lengthsTest2 = []
    all_rectangle_densityTest2 = []
    for builder in lineBuildersTest2:
        numLabelsTest2, rectangle_areasTest2 = builder.countLabelsArea()
        rectangle_lengthsTest2 = builder.countLabelsLen()
        rectangle_densityTest2 = numLabelsTest2 / (1536 * 1024)
        all_rectangle_densityTest2.append(rectangle_densityTest2)
        #print(builder.imageName)
        #print(rectangle_lengthsTest2)
        all_rectangle_lengthsTest2 += rectangle_lengthsTest2
        total_num_labelsTest2 += numLabelsTest2
        all_rectangle_areasTest2.extend(rectangle_areasTest2)

    Test1Stats = calculate_statistics(all_rectangle_areasTest1)
    Test2Stats = calculate_statistics(all_rectangle_areasTest2)


    summaryStatDF[extract_study_name_full(imageTestPath1)] = [
        Test1Stats["total_number"],
        Test1Stats["mean_size"],
        Test1Stats["std_dev"],
        min(all_rectangle_areasTest1),  # Assuming min value calculation
        Test1Stats["lower_quartile"],
        Test1Stats["50%_value"],
        Test1Stats["upper_quartile"],
        Test1Stats["max_size"],
        Test1Stats["median"]
    ]

    summaryStatDF[extract_study_name_full(imageTestPath2)] = [
        Test2Stats["total_number"],
        Test2Stats["mean_size"],
        Test2Stats["std_dev"],
        min(all_rectangle_areasTest2),  # Assuming min value calculation
        Test2Stats["lower_quartile"],
        Test2Stats["50%_value"],
        Test2Stats["upper_quartile"],
        Test2Stats["max_size"],
        Test2Stats["median"]
    ]
    #print(f"Total rectangle areas from all images: {len(all_rectangle_areas2)}")
    # Continue from your existing code where you aggregate all_rectangle_areas and total_num_labels

    # Calculate the average number of rectangles per image
    #average_num_rectangles_per_image2 = total_num_labels2 / len(files2) if len(files2) > 0 else 0

    # Calculate the average size of rectangles
    #average_rectangle_size2 = np.mean(all_rectangle_areas2) if all_rectangle_areas2 else 0

    # Calculate the standard deviation of rectangle sizes
    #std_dev_rectangle_size2 = np.std(all_rectangle_areas2) if all_rectangle_areas2 else 0

    #print(f"Average number of rectangles per image: {average_num_rectangles_per_image2}")
    #print(f"Average rectangle size: {average_rectangle_size2} square units")
    #print(f"Standard deviation of rectangle sizes: {std_dev_rectangle_size2} square units")

    #num_rectangles1, average_sizes1 = calculate_averages(lineBuilders1)
    num_rectanglesTest1, average_sizesTest1 = calculate_averages(lineBuildersTest1)

    #num_rectangles2, average_sizes2 = calculate_averages(lineBuilders2)
    num_rectanglesTest2, average_sizesTest2 = calculate_averages(lineBuildersTest2)

    from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

    # Assuming calculate_averages function has been defined and executed as previous examples

    # Normality Test
    print("Normality Test (Shapiro-Wilk):")
    print("Num Rectangles Dataset 1:", shapiro(num_rectanglesTest1))
    print("Num Rectangles Dataset 2:", shapiro(num_rectanglesTest2))
    print("Average Sizes Dataset 1:", shapiro(average_sizesTest1))
    print("Average Sizes Dataset 2:", shapiro(average_sizesTest2))

    # Levene's Test for Equality of Variances
    print("\nLevene's Test for Equality of Variances:")
    print("Num Rectangles:", levene(num_rectanglesTest1, num_rectanglesTest2))
    print("Average Sizes:", levene(average_sizesTest1, average_sizesTest2))

    # Independent Samples T-Test or Mann-Whitney U Test
    # Decide based on the results of normality and homogeneity tests
    # Here, using t-test as an example, but this choice depends on your test results
    print("\nIndependent Samples T-Test:")
    print("Num Rectangles:", ttest_ind(num_rectanglesTest1, num_rectanglesTest2, equal_var=True))
    print("Average Sizes:", ttest_ind(average_sizesTest1, average_sizesTest2, equal_var=True))

    # If normality or equality of variances assumptions are not met, use Mann-Whitney U Test instead
    print("\nMann-Whitney U Test:")
    print("Num Rectangles:", mannwhitneyu(num_rectanglesTest1, num_rectanglesTest2))
    print("Average Sizes:", mannwhitneyu(average_sizesTest1, average_sizesTest2))

    #average_aspect_ratios1 = [builder.average_aspect_ratio() for builder in lineBuilders1]
    average_aspect_ratiosTest1 = [builder.average_aspect_ratio() for builder in lineBuildersTest1]

    #average_aspect_ratios2 = [builder.average_aspect_ratio() for builder in lineBuilders2]
    average_aspect_ratiosTest2 = [builder.average_aspect_ratio() for builder in lineBuildersTest2]

    #print("Average Aspect Ratio per Image for Dataset 1:", np.mean(average_aspect_ratios1))
    #all_rectangle_areas1print("Average Aspect Ratio per Image for Dataset 2:", np.mean(average_aspect_ratios2))
    folder_nameTest1 = extract_color_name(imageTestPath1)
    folder_nameTest2 = extract_color_name(imageTestPath2)

    #folder_nameTest1 = os.path.basename(os.path.normpath(imageTestPath1))
    #folder_nameTest2 = os.path.basename(os.path.normpath(imageTestPath2))

    # Define a function to save plots
    def save_plot(data, color, study_name, plot_type):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, color=color, edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of Stomata Sizes - {study_name} - {plot_type}')
        plt.xlabel('Stomata Area (square units)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        #plt.savefig(f"{study_name}_{plot_type}.png")
        plt.close()

    # Plot and save histograms separately for Red and White datasets
    if all_rectangle_areasTest1:
        study_name1 = extract_study_name(imageTestPath1)
        save_plot(all_rectangle_areasTest1, 'red', study_name1, 'Red')

    if all_rectangle_areasTest2:
        study_name2 = extract_study_name(imageTestPath2)
        save_plot(all_rectangle_areasTest2, 'skyblue', study_name2, 'White')

    # Plot and save histogram for Red vs. White
    if all_rectangle_areasTest1 and all_rectangle_areasTest2:
        plt.figure(figsize=(10, 6))
        study_name = extract_study_name(imageTestPath1)  # Assuming both paths have the same study name
        plt.hist(all_rectangle_areasTest1, bins=30, color='red', edgecolor='black', alpha=0.7, label=f'{study_name} - Red')
        plt.hist(all_rectangle_areasTest2, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label=f'{study_name} - White')
        plt.title(f'Distribution of Stomata Sizes - {study_name} - Red vs. White')
        plt.xlabel('Stomata Area (square units)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        #plt.savefig(f"{study_name}_Red_vs_White.png")
        plt.close()
    else:
        print("No rectangle areas to plot for Red vs. White.")

    #if all_rectangle_areasTest1 and all_rectangle_areasTest2:
    #    plt.figure(figsize=(10, 6))
    #    
        # Extract folder names from the directory paths
        # folder_name1 = os.path.basename(os.path.normpath(imageTrainPath1))
        #folder_nameTest1 = os.path.basename(os.path.normpath(imageTestPath1))
    #    folder_nameTest1 = extract_color_name(imageTestPath1)

        # folder_name2 = os.path.basename(os.path.normpath(imageTrainPath2))
        #folder_nameTest2 = os.path.basename(os.path.normpath(imageTestPath2))
    #    folder_nameTest2 = extract_color_name(imageTestPath2)

        # Use folder names as labels
        #plt.hist(all_rectangle_areasTest1, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label=f'{folder_nameTest1}')
        #plt.hist(all_rectangle_areasTest2, bins=30, color='red', edgecolor='black', alpha=0.7, label=f'{folder_nameTest2}')

        #plt.title('Distribution of Stomata Sizes')
        #plt.xlabel('Stomata Area (square units)')
        #plt.ylabel('Frequency')
        #plt.grid(axis='y', alpha=0.75)
        
        # Display the legend with folder names
        #plt.legend()
        
        #plt.show()
    #else:
        #print("No rectangle areas to plot.")

    #plt.figure(figsize=(10, 6))

    #fig,ax1 = plt.subplots()

    # Plotting for the first dataset
    #sc2 = plt.scatter(num_rectanglesTest2, average_sizesTest2, color='red', label=f'{extract_color_name(imageTestPath1)}')
    #sc = plt.scatter(num_rectanglesTest1, average_sizesTest1, color='blue', label=f'{extract_color_name(imageTestPath2)}')
    #annot = ax1.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
    #                    bbox=dict(boxstyle="round", fc="w"),
    #                    arrowprops=dict(arrowstyle="->"))
    #annot.set_visible(False)

    #def update_annot(ind):
    #    
    #    pos = sc.get_offsets()[ind["ind"][0]]
    #    annot.xy = pos
    #    text =  lineBuilders1[ind["ind"][0]].imageName
    #    annot.set_text(text)
    #    annot.get_bbox_patch().set_alpha(0.4)

    #def update_annot2(ind):
    #    
    #    pos = sc2.get_offsets()[ind["ind"][0]]
    #    annot.xy = pos
    #    text =  lineBuilders2[ind["ind"][0]].imageName
    #    annot.set_text(text)
    #    annot.get_bbox_patch().set_alpha(0.4)
        

    #def hover(event):
    #    vis = annot.get_visible()
    #    if event.inaxes == ax1:
    #        cont, ind = sc.contains(event)
    #        if cont:
    #            update_annot(ind)
    #            annot.set_visible(True)
    #            fig.canvas.draw_idle()
    #        else:
    #            cont2, ind2 = sc2.contains(event)
    #            if cont2:
    #                update_annot2(ind2)
    #                annot.set_visible(True)
    #                fig.canvas.draw_idle()
    #            else:
    #                if vis:
    #                    annot.set_visible(False)
    #                    fig.canvas.draw_idle()

    #fig.canvas.mpl_connect("motion_notify_event", hover)
        
    # Plotting for the second dataset

    #plt.title('Number of Stomata vs. Average Stomata Size')
    #plt.xlabel('Number of Stomata')
    #plt.ylabel('Average Stomata Size (square units)')
    #plt.legend()
    #plt.grid(True)
    #plt.show()


    # Assuming average_aspect_ratios1, average_aspect_ratios2, num_rectangles1, num_rectangles2,
    # average_sizes1, and average_sizes2 are already calculated as described before

    #fig = plt.figure(figsize=(12, 8))
    #ax = fig.add_subplot(111, projection='3d')

    # Dataset 1
    #ax.scatter(num_rectangles1, average_aspect_ratios1, average_sizes1, color='blue', alpha=0.2, label=f'{folder_name1}')
    #ax.scatter(num_rectanglesTest1, average_aspect_ratiosTest1, average_sizesTest1, color='blue', alpha=0.7, label=f'{folder_nameTest1}')

    # Dataset 2
    #ax.scatter(num_rectangles2, average_aspect_ratios2, average_sizes2, color='red', alpha=0.2, label=f'{folder_name2}')
    #ax.scatter(num_rectanglesTest2, average_aspect_ratiosTest2, average_sizesTest2, color='red', alpha=0.7, label=f'{folder_nameTest2}')

    #ax.set_xlabel('Number of Stomata')
    #ax.set_ylabel('Average Aspect Ratio')
    #ax.set_zlabel('Average Stomata Size (square units)')

    #plt.title('3D Plot of Aspect Ratio, Number of Stomata, and Size')
    #plt.legend()
    #plt.show()

    import numpy as np
    from sklearn.model_selection import train_test_split

    # Example feature preparation
    # features1 and features2 are lists of [num_rectangles, average_aspect_ratio, average_size] for each image
    #features1 = np.array([[num_rectangles1[i], average_aspect_ratios1[i], average_sizes1[i]] for i in range(len(num_rectangles1))])
    featuresTest1 = np.array([[num_rectanglesTest1[i], average_aspect_ratiosTest1[i], average_sizesTest1[i]] for i in range(len(num_rectanglesTest1))])

    #features2 = np.array([[num_rectangles2[i], average_aspect_ratios2[i], average_sizes2[i]] for i in range(len(num_rectangles2))])
    featuresTest2 = np.array([[num_rectanglesTest2[i], average_aspect_ratiosTest2[i], average_sizesTest2[i]] for i in range(len(num_rectanglesTest2))])

    # Combine the features from both datasets
    #features = np.concatenate((features1, features2), axis=0)

    # Create labels (0 for Dataset 1, 1 for Dataset 2)
    #labels = np.array([0]*len(features1) + [1]*len(features2))

    #labels1 = np.array([0]*len(features1))
    labelsTest1 = np.array([0]*len(featuresTest1))

    #labels2 = np.array([1]*len(features2))
    labelsTest2 = np.array([1]*len(featuresTest2))

    #unique_labels = [labels1, labels2]

    # Split the data into training and testing sets
    #features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    labels_test = np.concatenate((labelsTest1, labelsTest2), axis=0)
    #labels_train = np.concatenate((labels1, labels2), axis=0)

    features_test = np.concatenate((featuresTest1, featuresTest2), axis=0)
    #features_train = np.concatenate((features1, features2), axis=0)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Initialize the Random Forest classifier with 100 trees
    #clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier on the training data
    #clf.fit(features_train, labels_train)

    # Predict the labels for the testing set
    #labels_pred = clf.predict(features_test)

    # Calculate the accuracy of the predictions
    # accuracy = accuracy_score(labels_test, labels_pred)
    # print(f"RF Accuracy: {accuracy*100:.2f}%")

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
    import matplotlib.pyplot as plt

    # Precision, Recall, and F1 Score
    #precision = precision_score(labels_test, labels_pred)
    #recall = recall_score(labels_test, labels_pred)
    #f1 = f1_score(labels_test, labels_pred)

    #print(f"RF Precision: {precision:.4f}")
    #print(f"RF Recall: {recall:.4f}")
    #print(f"RF F1 Score: {f1:.4f}")


    # AUC-ROC
    # Calculate probabilities for the positive class
    #labels_prob = clf.predict_proba(features_test)[:, 1]

    # Calculate ROC curve
    #fpr, tpr, thresholds = roc_curve(labels_test, labels_prob)

    # Calculate AUC
    #roc_auc = auc(fpr, tpr)

    #print(f"RF AUC-ROC: {roc_auc:.4f}")

    # Plotting ROC curve
    #plt.figure(figsize=(8, 6))
    #plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    #plt.legend(loc="lower right")
    #plt.show()

    from sklearn.neighbors import KNeighborsClassifier

    # Initialize the KNN classifier with k neighbors. Let's start with k=5, which is a common default choice
    #knn_clf = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier on the training data
    #knn_clf.fit(features_train, labels_train)

    # Predict the labels for the testing set
    #labels_pred_knn = knn_clf.predict(features_test)

    # Calculate the accuracy of the predictions
    #accuracy_knn = accuracy_score(labels_test, labels_pred_knn)
    #print(f"KNN Accuracy: {accuracy_knn*100:.2f}%")
    # Precision, Recall, and F1 Score for KNN
    #precision_knn = precision_score(labels_test, labels_pred_knn)
    #recall_knn = recall_score(labels_test, labels_pred_knn)
    #f1_knn = f1_score(labels_test, labels_pred_knn)

    #print(f"KNN Precision: {precision_knn:.4f}")
    #print(f"KNN Recall: {recall_knn:.4f}")
    #print(f"KNN F1 Score: {f1_knn:.4f}")

    # AUC-ROC for KNN
    #labels_prob_knn = knn_clf.predict_proba(features_test)[:, 1]
    #fpr_knn, tpr_knn, thresholds_knn = roc_curve(labels_test, labels_prob_knn)
    #roc_auc_knn = auc(fpr_knn, tpr_knn)

    #print(f"KNN AUC-ROC: {roc_auc_knn:.4f}")

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # Initialize the SVM classifier
    # Using a pipeline to standardize features by removing the mean and scaling to unit variance
    #svm_clf = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))

    # Train the classifier on the training data
    #svm_clf.fit(features_train, labels_train)

    # Predict the labels for the testing set
    #labels_pred_svm = svm_clf.predict(features_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, auc
    import matplotlib.pyplot as plt

    # Accuracy
    #accuracy_svm = accuracy_score(labels_test, labels_pred_s vm)
    #print(f"SVM Accuracy: {accuracy_svm*100:.2f}%")

    # Precision, Recall, and F1 Score
    #precision_svm = precision_score(labels_test, labels_pred_svm)
    #recall_svm = recall_score(labels_test, labels_pred_svm)
    #f1_svm = f1_score(labels_test, labels_pred_svm)

    #print(f"SVM Precision: {precision_svm:.4f}")
    #print(f"SVM Recall: {recall_svm:.4f}")
    #print(f"SVM F1 Score: {f1_svm:.4f}")

    # AUC-ROC
    #labels_prob_svm = svm_clf.predict_proba(features_test)[:, 1]
    #fpr_svm, tpr_svm, thresholds_svm = roc_curve(labels_test, labels_prob_svm)
    #roc_auc_svm = auc(fpr_svm, tpr_svm)

    #print(f"SVM AUC-ROC: {roc_auc_svm:.4f}")

    # Plotting ROC curve for SVM
    #plt.figure(figsize=(8, 6))
    #plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label=f'SVM ROC curve (area = {roc_auc_svm:.2f})')
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve for SVM')
    #plt.legend(loc="lower right")
    #plt.show()

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.ensemble import RandomForestClassifier

    # Assuming clf is your trained RandomForestClassifier
    # And features_train, labels_train are your training dataset and labels

    # Select three dimensions/features to visualize
    x_index, y_index, z_index = 0, 1, 2  # Adjust based on your dataset

    # Generate mesh grid for the selected features (first decision boundary)
    #x_min, x_max = features[:, x_index].min() - 1, features[:, x_index].max() + 1
    #y_min, y_max = features[:, y_index].min() - 1, features[:, y_index].max() + 1
    #z_min, z_max = features[:, z_index].min() - 1, features[:, z_index].max() + 1

    #xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
    #                    np.linspace(y_min, y_max, 50))

    # Set z (third feature) to its median value or another representative value
    #zz = np.median(features[:, z_index])

    # Flatten the grid to predict over it (first decision boundary)
    #grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, zz)]

    # Predict class for each point in the mesh grid (first decision boundary)
    #Z = clf.predict(grid)
    #Z = Z.reshape(xx.shape)

    #fig = plt.figure(figsize=(10, 7))
    #ax = fig.add_subplot(111, projection='3d')

    # Plot the first decision boundary
    #ax.contourf(xx, yy, Z, zdir='z', offset=zz, alpha=0.3, cmap='coolwarm')

    # Generate mesh grid for the second decision boundary (fixing Feature 2)
    #xx2, zz2 = np.meshgrid(np.linspace(x_min, x_max, 50),
    #                    np.linspace(z_min, z_max, 50))

    #yy2 = np.median(features[:, y_index])  # Fixed value for Feature 2

    # Flatten the grid to predict over it (second decision boundary)
    #grid_new = np.c_[xx2.ravel(), np.full(xx2.ravel().shape, yy2), zz2.ravel()]

    # Predict class for each point in the mesh grid (second decision boundary)
    #Z_new = clf.predict(grid_new)
    #Z_new = Z_new.reshape(xx2.shape)

    # Superimpose the second orthogonal decision boundary
    #ax.contourf(xx2, Z_new, zz2, zdir='y', offset=yy2, alpha=0.3, cmap='coolwarm')

    # Generate mesh grid for the second decision boundary (fixing Feature 2)
    #yy3, zz3 = np.meshgrid(np.linspace(y_min, y_max, 50),
    #                    np.linspace(z_min, z_max, 50))

    #xx3 = np.median(features[:, y_index])  # Fixed value for Feature 2
    #grid_3 = np.c_[yy3.ravel(), np.full(yy3.ravel().shape, xx3), zz3.ravel()]
    # Predict class for each point in the mesh grid (second decision boundary)
    #Z_3 = clf.predict(grid_3)
    #Z_3 = Z_new.reshape(yy3.shape)
    #ax.contourf(Z_3, yy3, zz3, zdir='x', offset=xx3, alpha=0.3, cmap='coolwarm')

    # Plot the training points
    #ax.scatter(features[:, x_index], features[:, y_index], features[:, z_index], c=labels, s=20, edgecolor='k', cmap='coolwarm')

    #ax.set_xlabel('Feature 1')
    #ax.set_ylabel('Feature 2')
    #ax.set_zlabel('Feature 3')
    #ax.set_title('Random Forest Decision Boundaries (3D Projection)')

    # Adjust the viewing angle if necessary to better visualize the interaction of decision boundaries
    #ax.view_init(elev=20., azim=30)

    # Set the limits of the plot to match our data
    #ax.set_xlim(x_min, x_max)
    #ax.set_ylim(y_min, y_max)
    #ax.set_zlim(z_min, z_max)

    #plt.show()



    # Assuming clf is your trained RandomForestClassifier
    # features and labels are your datasets

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier

    # Assuming clf is your trained RandomForestClassifier
    # features and labels are your datasets

    # Naming the features for clarity according to the corrected order
    feature_names = ["Stomata Count", "Avg Aspect Ratio", "Avg Size of Stomata"]

    # Fixed values for each feature, for plotting decision boundaries
    #fixed_values = {
    #    "Stomata Count": np.median(features[:, 0]),
    #    "Avg Aspect Ratio": np.median(features[:, 1]),
    #    "Avg Size of Stomata": np.median(features[:, 2])
    #}

    #dataset_labels = np.array([folder_name1] * len(features1) + [folder_name2] * len(features2))

    # Helper function to plot decision boundaries
    def plot_decision_boundary(x_index, y_index, fixed_feature_index, fixed_value):
        xx, yy = np.meshgrid(np.linspace(features[:, x_index].min() - 1, features[:, x_index].max() + 1, 100),
                            np.linspace(features[:, y_index].min() - 1, features[:, y_index].max() + 1, 100))
        grid = np.empty((xx.ravel().shape[0], 3))
        grid[:, x_index] = xx.ravel()
        grid[:, y_index] = yy.ravel()
        grid[:, fixed_feature_index] = fixed_value  # Set fixed feature value for all points in the grid

        Z = clf.predict(grid).reshape(xx.shape)

        #plt.figure(figsize=(8, 6))
        #plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')

        # Plot points from different datasets with different colors and labels
        #unique_datasets = np.unique(dataset_labels)
        #for dataset in unique_datasets:
        #    idx = dataset_labels == dataset
        #    if dataset == folder_name1:
        #        plt.scatter(features[idx, x_index], features[idx, y_index],  c='blue', label=dataset, edgecolor='k', cmap='warmcool')
        #    else:
        #        plt.scatter(features[idx, x_index], features[idx, y_index],  c='red', label=dataset, edgecolor='k', cmap='warmcool')

        #plt.scatter(features[:, x_index], features[:, y_index], c=labels, edgecolor='k', cmap='coolwarm')
        #plt.xlabel(feature_names[x_index])
        ##plt.ylabel(feature_names[y_index])
        #plt.title(f'Decision Boundary with {feature_names[fixed_feature_index]} Fixed at {fixed_value:.2f}')
        #plt.legend()
        #plt.show()

    # Plot decision boundaries for each pair of features with the third feature fixed
    #plot_decision_boundary(0, 1, 2, fixed_values["Avg Size of Stomata"])  # Num Rectangles vs Aspect Ratio (Avg Size fixed)
    #plot_decision_boundary(0, 2, 1, fixed_values["Avg Aspect Ratio"])  # Num Rectangles vs Avg Size (Aspect Ratio fixed)
    #plot_decision_boundary(1, 2, 0, fixed_values["Stomata Count"])  # Aspect Ratio vs Avg Size (Num Rectangles fixed)

output_path = "stomataSizeStatistics.csv"
summaryStatDF.to_csv(output_path)