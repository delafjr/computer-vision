import cv2, numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# index mata kiri (contoh)
L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133

def dist(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam, coba index 1/2")

# init object FaceMeshDetector
# staticMode=True -> deteksi hanya sekali; jika false -> deteksi setiap frame
# maxFaces: jumlah max wajah yg didteksi
# minDetectionCon: ambang kepercayaan deteksi minimal
detector = FaceMeshDetector(
    staticMode=False,
    maxFaces=2,
    minDetectionCon=0.5,
    minTrackCon=0.5
)

# var untuk menghitung kedipan sederhana
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3  # jumlah frame mata tertutup untuk dihitung sebagai kedipan
EYE_AR_THRESHOLD = 0.2  # ambang eye aspect ratio (EAR) untuk menilai mata tertutup
is_closed = False  # status mata saat ini

while True:
    ok, img = cap.read()
    if not ok:
        break
    img, faces = detector.findFaceMesh(img, draw=True)
    if faces:
        face = faces[0] # list of 468 (x, y)
        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])
        ear = v / (h + 1e-8)  # eye aspect ratio (EAR)
        cv2.putText(img, f"EAR(L): {ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # contoh ambang kedipan sdehana dan logika counter
        # jika EAR < EYE_AR_THRESHOLD selama CLOSED_FRAMES_THRESHOLD frame -> hitung kedipan 
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        # tampilakna jumlah kedipan mata pd frame
        cv2.putText(img, f"Blink: {blink_count}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Face Mesh + EAR", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()