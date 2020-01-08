import cv2
import imutils
import numpy as np
import DetectPlate2
import SegmentCharacters
import  PredictCharacters
#import TrainRecognizeCharacters

filename = 'xe00.mp4'
cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        frame1 = imutils.rotate(frame,270)

        #frame1 = frame.copy
        image, plate, likeObj = DetectPlate2.detectPlate(frame1)
        plate = plate[0]

        width = plate[3] -plate[1]
        height = plate[2] - plate[0]
        roi = frame1[plate[0]+ height//8:plate[2] - height//8, plate[1]+ width//30:plate[3]- width//30]
        # print(roi.shape)
        # image1, character = SegmentCharacters.segmentCharacters(likeObj, roi)
        image1, roiCharacter, column_list = SegmentCharacters.segmentPlate2(roi)
        licensePlate = PredictCharacters.predict(roiCharacter, column_list)
        cv2.putText(image, licensePlate, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
        if roi.shape[0] !=0 and roi.shape[1] != 0:
            cv2.imshow('roi', roi)
        cv2.imshow('detect Plate', image)
        cv2.imshow('detect Character', image1)
        count +=2
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()