import cv2      #import openCV library

alg = "Face Detection And Tracking\haarcascade_frontalface_default.xml"     #accessing data from the model
haar_cascade = cv2.CascadeClassifier(alg)       #loading the model

cam = cv2.VideoCapture(0)       #initializing camera
while True :        #infinite loop
    _,img = cam.read()      #read the frames from the camera
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #converting color image into gray scale image
    face = haar_cascade.detectMultiScale(grayimg, 1.3, 4)       #getting coordinates of face

    for (x, y, w, h) in face :
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("FaceDetection", img)
    key = cv2.waitKey(10)
    if key == 27 :
        break

cam.release()
cv2.destroyAllWindows()