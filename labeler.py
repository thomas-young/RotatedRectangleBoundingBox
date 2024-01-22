from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import atexit
import signal
import sys
import math
from matplotlib.lines import Line2D
import csv 
from pathlib import Path 
from datetime import datetime
import shutil

Path("./labels/").mkdir(parents=True, exist_ok=True)
now = datetime.now()
date_time = now.strftime("%m_%d_%y_%H_%M_%S")
backupFolder = "./backup/" + date_time + "/"
Path("./backup/").mkdir(parents=True, exist_ok=True)
#currentLabels = os.listdir("./labels/")
shutil.copytree("./labels/", backupFolder)

files = os.listdir('./images')
files = [f for f in files if os.path.isfile('./images'+'/'+f)] #Filtering only the files.
print(*files, sep="\n")

class Rectangle():

    def __init__(self, line1, line2, line3, line4):
        self.lines = [line1, line2, line3, line4]
        self.points = self.setPointsFromLines()
        self.currBaseLineXs = line1.get_xdata()
        self.currBaseLineYs = line1.get_ydata()
        self.currSlope = self.getCurrSlope()
        self.perpLineLength = self.getPerpLineLength()

    def getRectangleCSVLine(self):
        allPoints = []
        for point in self.points:
            allPoints.append(point[0])
            allPoints.append(point[1])
        return allPoints


    def setPointsFromLines(self):
        point1 = [self.lines[0].get_xdata()[0], self.lines[0].get_ydata()[0]]
        point2 = [self.lines[0].get_xdata()[1], self.lines[0].get_ydata()[1]]
        point3 = [self.lines[1].get_xdata()[1], self.lines[1].get_ydata()[1]]
        point4 = [self.lines[2].get_xdata()[1], self.lines[2].get_ydata()[1]]
        return [point1, point2, point3, point4]

    def draw(self):
        ax = plt.gca()
        for line in self.lines:
            ax.add_line(line)
        plt.draw()

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

    def clearRectange(self):
        for line in self.lines:
            line.remove()

    def recalculateLines(self, baseLine, pointId):
        #print("ATTEMPTING TO ROTATE ABOUT POINT: ", pointId)
        xs = list(baseLine.get_xdata())
        ys = list(baseLine.get_ydata())
        deltaX = xs[1] - xs[0]
        deltaY = ys[1] - ys[0]
        baseLineSlope = None
        if deltaX != 0:
            baseLineSlope = deltaY/deltaX
        else:
            #print("DELTA X = 0")
            if self.currSlope >= 0:
                baseLineSlope = 999999999
            else:
                baseLineSlope = -999999999

        if baseLineSlope != 0:
            perpSlope = -1/baseLineSlope
        else:
            #print("DELTA Y = 0")

            perpSlope = 999999999


        #print("old slope: ", self.currSlope)
        #print('new slope: ', baseLineSlope)
        rotationAngle = -1 * (math.atan(self.currSlope) - math.atan(baseLineSlope))
        #print("rotationAngle: ",rotationAngle)
        if pointId == 1:

            # 
            px = self.lines[1].get_xdata()[1]
            py = self.lines[1].get_ydata()[1]

            # bottom right point before rotation
            tx = self.currBaseLineXs[1]
            ty = self.currBaseLineYs[1]
            #print("tx: ", tx)
            #print("xs: ", xs)
            #print("ty: ", ty)
            #print("ys: ", ys)

            fx=math.cos(rotationAngle)*(tx-xs[0])-math.sin(rotationAngle)*(ty-ys[0])   + xs[0]
            fy=math.sin(rotationAngle)*(tx-xs[0])+math.cos(rotationAngle)*(ty-ys[0])  + ys[0]

            transX = xs[1] - fx
            transY = ys[1] - fy
            #print("transX: ", transX)
            #print("transY: ", transY)
            px += abs(transX)
            py += abs(transY)
         
            qx=math.cos(rotationAngle)*(px-xs[0])-math.sin(rotationAngle)*(py-ys[0])+xs[0]
            qy=math.sin(rotationAngle)*(px-xs[0])+math.cos(rotationAngle)*(py-ys[0])+ys[0]

            ox2, px2 = self.lines[2].get_xdata()
            oy2, py2 = self.lines[2].get_ydata()

            qx2 = math.cos(rotationAngle)*(px2-xs[0])-math.sin(rotationAngle)*(py2-ys[0])+xs[0] 
            qy2 = math.sin(rotationAngle)*(px2-xs[0])+math.cos(rotationAngle)*(py2-ys[0])+ys[0]

            #print("old perp line data: ", px, py)
            #print("new perp line data: ", qx, qy)
            topLineIntercept = qy2 - (baseLineSlope * qx2)
            perpIntercept = ys[1] - (perpSlope * xs[1])
            x2 = (perpIntercept - topLineIntercept) / (baseLineSlope - perpSlope)
            y2 = (x2 * baseLineSlope) + topLineIntercept
            self.lines[0].set_data(xs, ys)
            self.currSlope = self.getCurrSlope()
            self.currBaseLineXs = self.lines[0].get_xdata()
            self.currBaseLineYs = self.lines[0].get_ydata()

            self.lines[1].set_data([xs[1], x2], [ys[1],y2])
            self.lines[2].set_data([x2, qx2], [y2,qy2])
            self.lines[3].set_data([xs[0], qx2], [ys[0],qy2])
            self.points = self.setPointsFromLines()
        if pointId == 0:
            px = self.lines[2].get_xdata()[1]
            py = self.lines[2].get_ydata()[1]

            # bottom left point before rotation
            tx = self.currBaseLineXs[0]
            ty = self.currBaseLineYs[0]

            fx=math.cos(rotationAngle)*(tx-xs[1])-math.sin(rotationAngle)*(ty-ys[1])   + xs[1]
            fy=math.sin(rotationAngle)*(tx-xs[1])+math.cos(rotationAngle)*(ty-ys[1])  + ys[1]

            transX = xs[0] - fx
            transY = ys[0] - fy
            #print("transX: ", transX)
            #print("transY: ", transY)
            px += transX
            py += transY

            qx=math.cos(rotationAngle)*(px-xs[1])-math.sin(rotationAngle)*(py-ys[1])+xs[1]
            qy=math.sin(rotationAngle)*(px-xs[1])+math.cos(rotationAngle)*(py-ys[1])+ys[1]

            ox2, px2 = self.lines[1].get_xdata()
            oy2, py2 = self.lines[1].get_ydata()

            qx2 = math.cos(rotationAngle)*(px2-xs[1])-math.sin(rotationAngle)*(py2-ys[1])+xs[1] 
            qy2 = math.sin(rotationAngle)*(px2-xs[1])+math.cos(rotationAngle)*(py2-ys[1])+ys[1]

            topLineIntercept = qy2 - (baseLineSlope * qx2)
            perpIntercept = ys[0] - (perpSlope * xs[0])
            x2 = (perpIntercept - topLineIntercept) / (baseLineSlope - perpSlope)
            y2 = (x2 * baseLineSlope) + topLineIntercept
            self.lines[0].set_data(xs, ys)
            self.currSlope = self.getCurrSlope()
            self.currBaseLineXs = self.lines[0].get_xdata()
            self.currBaseLineYs = self.lines[0].get_ydata()

            self.lines[1].set_data([xs[1], qx2], [ys[1],qy2])
            self.lines[2].set_data([qx2, x2], [qy2,y2])
            self.lines[3].set_data([xs[0], x2], [ys[0],y2])
            self.points = self.setPointsFromLines()


class LineBuilder():

    epsilon = 30

    def __init__(self, imageName):
        fig = plt.gcf()
        canvas = fig.canvas
        self.imageName = imageName
        self.canvas = canvas
        self.lines = []
        self.axes = plt.gca()
        self.xs = []
        self.ys = []
        self.perpLine = None
        self.currLineSlope = None
        self.currentLineInvNegSlope = None
        self.currLine = None
        self.rectDict = {}
        self.animated_index = None
        self.animated_ind = None
        self.addingLine = False
        for line in self.lines:
            self.xs.append(list(line.get_xdata()))
            self.ys.append(list(line.get_ydata()))
    
        self.ind = None
        self.idx = None
        canvas.mpl_connect('button_press_event', self.buttonPressCallback)
        canvas.mpl_connect('button_release_event', self.buttonReleaseCallback)
        canvas.mpl_connect('motion_notify_event', self.motionNotifyCallback)
    
    def printLineCount(self):
        print("NUM LINES: ", len(self.lines))

    def clearLines(self):
        #print("num lines: ", len(self.lines))
        for line in self.lines:
            print(line)
            line.remove() 
        if self.perpLine is not None:
            try:
                self.perpLine.remove()
            except:
                print("no good!")
        #print("num lines: ", len(self.lines))

    def drawLines(self):
        ax = plt.gca()
        for line in self.lines:
            ax.add_line(line)
        if len(self.lines) >= 1:
            self.currLine = self.lines[-1]
            self.drawPerpLine(self.lines[-1])
        plt.draw()

    def addLine(self, x, y):
        if not self.addingLine:
            return
        if len(x) != 2 or len(y) != 2:
            return
        line = Line2D(x, y, marker='o', markerfacecolor='red')
        self.currLine = line
        ax = plt.gca()
        ax.add_line(line)
        self.lines.append(line)
        self.xs.append(list(line.get_xdata()))
        self.ys.append(list(line.get_ydata()))
        if self.perpLine is not None:
            try:
                self.perpLine.remove()
            except:
                print("no good!")
        self.drawPerpLine(line) 
        self.addingLine = False
        plt.draw()

    def stopLinePlacement(self):
        if self.addingLine:
            self.addingLine = False

    def deleteLine(self, x, y):
        self.ind, self.idx = self.getInd(x,y)
        if self.idx is None:
            return
        #print("attemping to remove line: ", self.idx)
        currDelLine = self.lines[self.idx]
        currDelLine.remove()
        self.xs.pop(self.idx)
        self.ys.pop(self.idx)
        if currDelLine in self.rectDict:
            self.rectDict[currDelLine].clearRectange()
            del self.rectDict[currDelLine]
        self.perpLine = None
        self.lines.pop(self.idx)
        self.canvas.draw()

    def deleteLines(self):
        #print("num lines: ", len(self.lines))
        if self.perpLine is not None:
            try:
                self.perpLine.remove()
            except:
                print("no good!")
        for line in self.lines:
            if line in self.rectDict:
                self.rectDict[line].clearRectange()
                del self.rectDict[line]
            line.remove()

        self.lines = []
        self.xs = []
        self.ys = []
        self.canvas.draw()
        self.perpLine = None
        self.currLineSlope = None
        self.currentLineInvNegSlope = None
        self.currLine = None
        #print("num lines: ", len(self.lines))

    def drawPerpLine(self, line):
        xs = list(self.currLine.get_xdata())
        ys = list(self.currLine.get_ydata())
        deltaX = xs[1] - xs[0]
        deltaY = ys[1] - ys[0]
        if deltaX != 0:
            self.currLineSlope = deltaY/deltaX
        else:
            self.currLineSlope = 999999999
        intercept = ys[0] - (self.currLineSlope * xs[0])
        if self.currLineSlope != 0:
            self.currentLineInvNegSlope = -1.0 / self.currLineSlope
        else:
            self.currentLineInvNegSlope = 999999999
        self.perpIntercept = ys[1] - (self.currentLineInvNegSlope * xs[1])
        #print("currentLineInvNegSlope: ", self.currentLineInvNegSlope)
        lineLength = math.sqrt(deltaX**2 + deltaY**2)
        lenScaler = 1/self.currentLineInvNegSlope
        perpLineEndpointX = xs[1] + ((2000)*lenScaler)
        perpLineStartpointX = xs[1] - (2000*lenScaler)
        perpLineEndpointY = ys[1] + (2000*lenScaler* (self.currentLineInvNegSlope))
        perpLineStartpointY = ys[1] - (2000*lenScaler* (self.currentLineInvNegSlope))
        perpLine = Line2D([perpLineStartpointX, perpLineEndpointX], [perpLineStartpointY,perpLineEndpointY], alpha=.7, linestyle='dashed', color="yellow")
        ax = plt.gca()
        ax.add_line(perpLine)
        self.perpLine = perpLine
        plt.draw()
    
    def getInd(self, xdata, ydata, getClosest=False):
        returnVal = None
        returnIdx = None
        minD = 999999999
        currIdx = 0
        for line in self.lines:
            x = np.array(line.get_xdata())
            y = np.array(line.get_ydata())
            d = np.sqrt((x-xdata)**2 + (y - ydata)**2)
            if d[0] < d[1]:
                if d[0] < minD:
                    minD = d[0]
                    #print("found new min dist line at idx: %d", currIdx)
                    returnVal = 0
                    returnIdx = currIdx
            else:
                if d[1] < minD:
                    minD = d[1]
                    returnVal = 1
                    #print("found new min dist line at idx: %d", currIdx)
                    returnIdx = currIdx
            currIdx += 1
        #print("final minD: ", minD)
        #print("epsilon: ", self.epsilon)
        if minD > self.epsilon and getClosest is False:
                returnVal = None
                returnIdx = None
        return returnVal, returnIdx

    def getDistanceToPerpLine(self, event):
        clickX = event.xdata
        clickY = event.ydata
        rectLineSlope = -1.0 / self.currentLineInvNegSlope
        rectLineIntercept = clickY - (clickX * rectLineSlope)
        x2 = (self.perpIntercept - rectLineIntercept) / (rectLineSlope - self.currentLineInvNegSlope)
        y2 = (x2 * rectLineSlope) + rectLineIntercept
        #print("x1: ", clickX)
        #print("y1: ", clickY)
        #print("x2: ",x2)
        #print("y2: ",y2)
        rectLineLength = abs(((x2-clickX)**2 + (y2-clickY)**2)**.5)
        #print("Attempting to place rectangle, distance from perp line: ", rectLineLength)
        #print("calling getInd from getDistanceToPerpLine")
        #print("index: ", self.getInd(event.xdata, event.ydata)[0])
        if rectLineLength < 25 and self.getInd(event.xdata, event.ydata)[0] is None:
            #print("Placing new rectangle")
            if self.currLine in self.rectDict:
                self.rectDict[self.currLine].clearRectange()
            self.addRectangle(x2, y2, rectLineIntercept, rectLineSlope)

    def addRectangle(self, intersectX, intersectY, rectIntercept, rectSlope):
        xs = list(self.currLine.get_xdata())
        ys = list(self.currLine.get_ydata())
        perpIntercept2 = ys[0] - (self.currentLineInvNegSlope * xs[0])
        x2 = (perpIntercept2 - rectIntercept) / (rectSlope - self.currentLineInvNegSlope)
        y2 = (x2 * rectSlope) + rectIntercept
        firstLine = self.currLine
        perpLineRect = Line2D([xs[1], intersectX], [ys[1],intersectY], color="blue")
        rectLineExtension = Line2D([intersectX, x2], [intersectY,y2], color="blue")
        perpLine2 = Line2D([xs[0], x2], [ys[0],y2],  color="blue")
        rectangle = Rectangle(firstLine, perpLineRect, rectLineExtension, perpLine2)
        rectangle.draw()

        self.rectDict[self.currLine] = rectangle

    def clearRectangles(self):
        for rectangle in self.rectDict.values():
            rectangle.clearRectange()

    def drawRectangles(self):
        for rectangle in self.rectDict.values():
            rectangle.draw()

    def saveRectangles(self):
        filename, file_extension = os.path.splitext(self.imageName)
        csvName = "labels/" + filename + ".csv"

        print(csvName)
        with open(csvName, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["point1 x", "point1 y", "point2 x", "point2 y", "point3 x", "point3 y", "point4 x", "point4 y"])
            for rectangle in self.rectDict.values():
                writer.writerow(rectangle.getRectangleCSVLine())

    def loadRectangles(self):
        filename, file_extension = os.path.splitext(self.imageName)
        csvName = "labels/" + filename + ".csv"
        print(csvName)
        try:
            with open(csvName, 'r', newline='') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    if row[0] == 'point1 x':
                        continue
                    #baseLine = Line2D([float(row[0]), float(row[2])], [float(row[1]),float(row[3])], marker='o', markerfacecolor='red')
                    line2 = Line2D([float(row[2]), float(row[4])], [float(row[3]),float(row[5])], color="blue")
                    line3 = Line2D([float(row[4]), float(row[6])], [float(row[5]),float(row[7])], color="blue")
                    line4 = Line2D([float(row[6]), float(row[0])], [float(row[7]),float(row[1])], color="blue")
                    self.addingLine = True
                    self.addLine([float(row[0]), float(row[2])], [float(row[1]),float(row[3])])
                    rectangle = Rectangle(self.currLine, line2, line3, line4)
                    rectangle.draw()
                    self.rectDict[self.currLine] = rectangle
        except:
            print("Couldnt find: ", csvName)


    def buttonPressCallback(self, event):
        if event.button != 1:
            return
        self.ind, self.idx = self.getInd(event.xdata, event.ydata)
        if self.idx is None:
            if self.currentLineInvNegSlope is not None:
                self.getDistanceToPerpLine(event)
            return
        self.animated_index = self.idx
        self.animated_ind = self.ind
        self.lines[self.idx].set_animated(True)

        if self.lines[self.idx] in self.rectDict:
            for rectLine in self.rectDict[self.lines[self.idx]].lines:
                rectLine.set_animated(True)
        if self.perpLine is not None:
            self.perpLine.remove()    
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.lines[self.idx].axes.bbox)
        self.axes.draw_artist(self.lines[self.idx])
        self.canvas.blit(self.axes.bbox)

    def buttonReleaseCallback(self, event):
        oldIdx = self.idx
        self.animated_index = None
        self.animated_ind = None

        self.ind, self.idx = self.getInd(event.xdata,event.ydata)
        if event.button != 1:
            return
        if self.idx is None:
            return
        if self.lines[self.idx] in self.rectDict and self.idx is not None:
            currRect = self.rectDict[self.lines[self.idx]]
            currRect.recalculateLines(self.lines[self.idx], self.ind)
        self.ind = None
        self.lines[self.idx].set_animated(False)
        if self.lines[self.idx] in self.rectDict:
            for rectLine in self.rectDict[self.lines[self.idx]].lines:
                rectLine.set_animated(False)
        self.background = None

        if self.lines[self.idx].figure is not None:
            self.lines[self.idx].figure.canvas.draw()
        self.currLine = self.lines[self.idx]
        if oldIdx is not None:
            self.drawPerpLine(self.lines[oldIdx])
        self.idx = None

    def motionNotifyCallback(self, event):
        if self.animated_index is None:
            return
        if self.animated_ind is None:
            return
        if self.idx is None:
            return
        if event.inaxes != self.lines[self.idx].axes:
            return
        if event.button != 1:
            return

        self.xs[self.animated_index][self.animated_ind] = event.xdata
        self.ys[self.animated_index][self.animated_ind] = event.ydata
        self.lines[self.animated_index].set_data(self.xs[self.animated_index], self.ys[self.animated_index])
        if self.lines[self.animated_index] in self.rectDict:
            self.rectDict[self.lines[self.animated_index]].recalculateLines(self.lines[self.animated_index], self.animated_ind)

        self.canvas.restore_region(self.background)

        self.axes.draw_artist(self.lines[self.animated_index])
        if self.lines[self.animated_index] in self.rectDict:
            for rectLine in self.rectDict[self.lines[self.animated_index]].lines:
                self.axes.draw_artist(rectLine)

        self.canvas.blit(self.axes.bbox)

lineBuilders = []
for file in files:
    lineBuilders.append(LineBuilder(file))

current_image_index = 0
img = np.asarray(Image.open('./images/' + files[current_image_index]))
numImages = len(files)
imgplot = plt.imshow(img)
linebuilder = lineBuilders[0]
linebuilder.loadRectangles()

def onKeyPress(event):
    global current_image_index
    global linebuilder
    linebuilder.stopLinePlacement()

    if event.key == 'left':
        current_image_index += -1
        if current_image_index < 0:
            current_image_index = len(files) -1
        #print(f"Displaying image {files[current_image_index]}: {current_image_index + 1} of {len(files)}")
        img = np.asarray(Image.open('./images/' + files[current_image_index]))
        fig.canvas.manager.set_window_title(files[current_image_index])
        plt.xlabel(files[current_image_index], fontsize=18)
        plt.title("Image " + str(current_image_index  + 1) + " of " + str(numImages))

        linebuilder.saveRectangles()
        linebuilder.deleteLines()

        linebuilder = lineBuilders[current_image_index]
        linebuilder.loadRectangles()

        imgplot.set_data(img)
        plt.draw()
    if event.key == 'right':
        current_image_index += 1
        if current_image_index >= len(files):
            current_image_index = 0
        #print(f"Displaying image {files[current_image_index]}: {current_image_index + 1} of {len(files)}")
        img = np.asarray(Image.open('./images/' + files[current_image_index]))
        fig.canvas.manager.set_window_title(files[current_image_index])
        plt.xlabel(files[current_image_index], fontsize=18)
        plt.title("Image " + str(current_image_index  + 1) + " of " + str(numImages))

        linebuilder.saveRectangles()
        linebuilder.deleteLines()

        linebuilder = lineBuilders[current_image_index]
        linebuilder.printLineCount()
        linebuilder.loadRectangles()

        imgplot.set_data(img)
        plt.draw()
    if event.key == 'a':
        linebuilder.addingLine = True
        xy = plt.ginput(2)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        linebuilder.addLine(x,y)
    elif event.key == 'x':
        xy = plt.ginput(1)
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        linebuilder.deleteLine(x,y)
    else:
        print(f"Invalid key: {event.key}")

def on_close(event):
    linebuilder.saveRectangles()

    print('Closed Figure!')


if __name__ == '__main__':
    fig = plt.gcf()   
    fig.canvas.mpl_connect('key_press_event', onKeyPress)
    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.manager.set_window_title(files[current_image_index])
    plt.xlabel(files[current_image_index], fontsize=18)

    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.title("Image " + str(current_image_index  + 1) + " of " + str(numImages))

    plt.show()

