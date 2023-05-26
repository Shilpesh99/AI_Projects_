import cv2      #import opencv library
import time     #import time library
import imutils      #import resizing library

cam = cv2.VideoCapture(0)       #initializing the camera
time.sleep(1)       #1 second delay

FirstFrame = None       #initializing there is no object
area = 500      #threshold

while True :        #infinite loop
    _,img = cam.read()      #read the frame from camera
    text = "Normal"      #initializing the text
    img = imutils.resize(img, width=500)        #resizing the image
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #convert the colored image to gray scale image
    gaussianimg = cv2.GaussianBlur(grayimg, (21, 21), 0)        #smoothening

    if FirstFrame is None :
        FirstFrame = gaussianimg       #gaussian image will be saved as first frame
        continue

    imgdiff = cv2.absdiff(FirstFrame, gaussianimg)      #substracting current frame from first frame
    thresimg = cv2.threshold(imgdiff, 25, 255, cv2.THRESH_BINARY)[1]        #applying threshold
    thresimg = cv2.dilate(thresimg, None, iterations=2)     #removing holes

    cnts = cv2.findContours(thresimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        #covering up the whole moving object as a single area
    cnts = imutils.grab_contours(cnts)
    for c in cnts :
        if cv2.contourArea(c) < area :
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = "Moving Object Detected"
    print(text)
    cv2.putText(img, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow("CameraFeed", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") :
        break
cam.release()
cv2.destroyAllWindows()