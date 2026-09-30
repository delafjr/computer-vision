import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open video device")

detector = HandDetector(
    staticMode=False, 
    maxHands=1, 
    modelComplexity=1, 
    detectionCon=0.5, 
    minTrackCon=0.5
)

while True:
    ok, img = cap.read()
    if not ok:
        break

    hands, img = detector.findHands(
        img,
        draw=True,
        flipType=True           # flip type untuk mirror ui
    )

    if hands:
        hand = hands[0]     # dict berisi lmList, bbox, dll
        fingers = detector.fingersUp(hand)  # list berisi 0/1 untuk setiap jari
        count = sum(fingers)  # jumlah jari yang terdeteksi
        cv2.putText(
            img,
            f'Fingers: {count}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2    
        )

    cv2.imshow("Hand Fingers Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()