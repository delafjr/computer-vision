import cv2, numpy as np
from cvzone.HandTrackingModule import HandDetector

def dist(a,b):
    return np.linalg.norm(np.array(a) - np.array(b))

def classify_gesture(hand):
    # hand["lmList"] contains the list of landmarks for the detected hand
    lm = hand["lmList"]
    wrist = np.array(lm[0][:2])  # Wrist landmark
    thumb_tip = np.array(lm[4][:2])  # Thumb tip landmark
    index_tip = np.array(lm[8][:2])  # Index finger tip landmark
    middle_tip = np.array(lm[12][:2])  # Middle finger tip landmark
    ring_tip = np.array(lm[16][:2])  # Ring finger tip landmark
    pinky_tip = np.array(lm[20][:2])  # Pinky finger tip landmark

    # heuristik jarak relatif antara jari untuk mengklasifikasikan gesture
    r_mean = np.mean([dist(index_tip, wrist),
                      dist(middle_tip, wrist),
                      dist(ring_tip, wrist),
                      dist(pinky_tip, wrist),
                      dist(thumb_tip, wrist)])

    if dist(thumb_tip, index_tip) < 35:
        return "OK"
    elif (thumb_tip[1] < wrist[1]-40) and (dist(thumb_tip, wrist) > 0.8 * dist(index_tip, wrist)):
        return "THUMBS UP"
    elif r_mean < 120:
        return "ROCK"
    elif r_mean > 200:
        return "PAPER"
    elif dist(index_tip, wrist) > 180 and dist(middle_tip, wrist) > 180 and dist(ring_tip, wrist) < 160 and dist(pinky_tip, wrist) < 160:
        return "SCISSORS"
    return "UNKNOWN"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

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
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        label = classify_gesture(hands[0])
        cv2.putText(img, f"Gesture: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()