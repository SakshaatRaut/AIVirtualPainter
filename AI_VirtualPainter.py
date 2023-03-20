import cv2
import numpy as np
import time
import HandTrackingModule as htm
import os

################
brushThickness = 15  #Brush thickness used as per convenience#
eraserThickness = 85  #Eraser thickness used as per convenience#
################
FolderPath = "Header_Painter"
myList = os.listdir(FolderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{FolderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# CREATED A CANVAS IMAGE FOR PRINTING THE DRAWING WE DRAW ON THE SCREEN
# WE USE THE NUMPY ZEROS METHOD HERE WE WRITE HEIGHT B4 THE WIDTH AND HAS 3 CHANNELS BCOZ 3 COLOURS
# NUMPY ZEROS --
# NP.UINT8 (UNSIGNED INTEGER OF 8 BITS) IS USED FOR COLOUR DETECTION AS WE WANT VALUES FROM 0 TO 255

while True:
    # 1. Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.FindHands(img)
    lmList = detector.FindPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList)

        # Tip of index finger(x1,y1) and middle finger(x2,y2)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        # 3. Check which fingers are up
        fingers = detector.FingersUp()
        #print(fingers)

        # 4. If selection mode - 2 fingers are up.
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("Selection Mode")
            # Checking for the click & The if elif ranges give the range as per the Header of draw screen.
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), drawColor, cv2.FILLED)


        # 5. If Drawing mode - Index finger is up.
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) ###IMG INVERSE IS DONE COZ CAMERA SHOULD CAPTURE A MIRROR IMAGE OF DRAWING FOR COMFORT OF USER
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    #Setting the header image
    img[0:125, 0:1280] = header ### FIXING THE HEADER OF CANVA PRINT I MADE ###
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    #final outputting the cv images canvas#
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)

