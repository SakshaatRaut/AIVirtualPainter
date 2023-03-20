import cv2
import mediapipe as mp
import time

### Hand detection module inside cv2 module ###
### It also gives the detection Confidence ###
class HandDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity=1 , detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

### Colour of the FingerTips and the axis of alignment ###

    def FindHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def FindPosition(self, img, handNo=0, draw=True ):

        self.LmList = []
        if self.results.multi_hand_landmarks:
            MyHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(MyHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.LmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return self.LmList

    def FingersUp(self):
        fingers = []

        # Thumb is up or not #
        if self.LmList[self.tipIds[0]][1] < self.LmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers are up or not #
        for id in range(1, 5):
            if self.LmList[self.tipIds[id]][2] < self.LmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

### Shows the FPS used by the program ###
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  #Used for camera capture#
    detector = HandDetector()  #calling handdetector fxn#

    while True:
        success, img = cap.read()
        img = detector.FindHands(img)
        LmList = detector.FindPosition(img)
        if len(LmList) != 0:
            print(LmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
### The FPS text Font, Color, Size and alignment ###
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1) ###Displays Camera###

if __name__ == "__main__":
    main()

