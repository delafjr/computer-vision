import cv2, numpy as np
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam, coba index 1/2")

detector = PoseDetector(staticMode=False, modelComplexity=1,
                        enableSegmentation=False, detectionCon=0.5, trackCon=0.5)

while True:
    # tangkap setiap frame dari webcam
    success, img = cap.read()

    # find pose manusia dalam frame
    img = detector.findPose(img)

    # find landmark, bounding box, dan pusat tubuh dalam frame
    # set draw=true untuk menggambar landmark dan bounding box pada gambar
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=True)

    # periksa apakah ada landmark tubuh terdeteksi
    if lmList:
        # get pusat bounding box di sekitar tubuh
        center = bboxInfo["center"]

        # draw lingkaran di pusat bounding box
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # hitung jarak antara landmark 11 dan 15 dan gambarkan pd gambar
        length, img, info = detector.findDistance(lmList[11][0:2],
                                                  lmList[15][0:2], 
                                                  img=img,
                                                  color=(255, 0, 255),
                                                  scale=10)

        # hitung sudut anda antara landmark 11, 13, dan 15 dan gambarkan pd gambar
        angle, img = detector.findAngle(lmList[11][0:2],
                                              lmList[13][0:2],
                                              lmList[15][0:2],
                                              img=img,
                                              color=(255, 0, 255),
                                              scale=10)

        # periksa apakah sudut mendekati 50 derajat dengan offset 10
        isCloseAngle50 = detector.angleCheck(myAngle=angle, targetAngle=50, offset=10)

        # print hasil periksa sudut
        print(isCloseAngle50)

    cv2.imshow("Pose + Angel", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()