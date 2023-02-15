import cv2


cv2.startWindowThread()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

