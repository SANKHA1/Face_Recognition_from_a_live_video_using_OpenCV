# Face Recognition from Live video

I am trying to recognize my face from live video.


## STEP 1 - Collection of images for training

Run the 'datacollect.py' file to save images from webcam. It will take 500 images for each set and save them in the 'images' folder. I have created two sets of images 'Sankha' and 'Sankha without specs'.

## STEP 2 - Implementation of OpenCV HAAR CASCADES

I am using the "Frontal Face Alt" Classifier for detecting the presence of Face in the WebCam. 

## STEP 3 - ReTraining the Network - Tensorflow Image Classifier

To retrain the network I am using Mobilenet Model which is quite fast and accurate. To run the training, I am running the following command in terminal:-

      python retrain.py --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --architecture=MobileNet_1.0_224 --image_dir=images


## STEP 4 - Importing the ReTrained Model and Setting Everything Up

I am runing the following command in terminal:
      
     python label.py
     
It wll open a new window of OpenCV and then identifies my face. The video is saved with the name 'output_face_recognition'.avi'. The names of the persons present in the video are saved in 'Attendance.csv' file.

