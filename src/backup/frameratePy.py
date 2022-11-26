import cv2
import time

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(3, hCam)

pTime = 0

while True:
    success, img = cap.read()
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    if success:
        cv2.putText(img, f'FPS: {int(fps)}', (420, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 1), 3)
        cv2.imshow("webcam", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print('no camera')
        break
cap.release()
cv2.destroyAllWindows()