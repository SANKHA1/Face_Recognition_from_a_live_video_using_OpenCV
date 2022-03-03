import cv2
import label_image
import pandas as pd
import os
size = 4
label_dict  = os.listdir("images")
img_array = []
# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0) #Using default WebCam connected to the PC.
df = pd.DataFrame()
df["Name"] = label_dict
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_face_recognition.avi', fourcc, 6, (640, 480))
textlist = []
while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,0) #Flip to act as a mirror
    # Resize the image to speed up detection
    mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face

    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
        
        #Save just the rectangle faces in SubRecFaces
        sub_face = im[y:y+h, x:x+w]

        FaceFileName = "test.jpg" #Saving the current image from the webcam for testing.
        cv2.imwrite(FaceFileName, sub_face)
        
        text = label_image.main(FaceFileName)# Getting the Result from the label_image file, i.e., Classification Result.
        text = text.title()# Title Case looks Stunning.
        textlist.append(text)
        font = cv2.FONT_HERSHEY_TRIPLEX
        # cv2.putText(im, text,(x+w,y), font, 1, (0,0,255), 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), (0, 255, 0), -2)
        cv2.putText(im, text, (x, y - 10), font, 0.75, (255, 255, 255), 1,cv2.LINE_AA)

        cv2.imwrite(FaceFileName, im)
    # Show the image
    cv2.imshow('Capture',   im)
    out.write(im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key3
        textlist = list(set(textlist))
        textlist = [x.lower() for x in textlist]
        # print(f"label dict {label_dict}")
        # print(f"textlist {textlist}")
        df["Attendance"] = df.apply(lambda x: "Present" if x["Name"] in textlist else "Absent", axis=1)
        df.to_csv("Attendance.csv", index= False)
        break
out.release()
webcam.release()
cv2.destroyAllWindows()
