# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:59:31 2023

@author: acer
"""
import cv2
import streamlit as st
import pickle as p
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

Categories= ['droppy', 'normal_1']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

model = p.load(open('E:/anaconda3/envs/biopy/Project_Skripsi/img_model.p','rb'))
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces: 
            #roi_rgb = rgb[y:y+h, x:x+w]
            # Draw a rectangle around the face
            color = (255, 0, 0) # in BGR
            stroke = 2
            cv2.rectangle(img, (x, y), (x + w, y + h), color, stroke)

            # resize the image
            size=(150,150,3)
            resized_image = resize(img, size)
            image_array = np.array(resized_image, "uint8")
            #img = image_array.reshape(150,150,3)
            img_flatten=[image_array.flatten()]
   

            # predict the image
            label=model.predict(img_flatten)[0]
            label = str(label)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

#def video_stream():
#    cap = cv2.VideoCapture(0)

#    while True:
#        ret, frame = cap.read()
#       frame = detect_faces(frame)

#       st.image(frame)

st.title("Real-time Face Recognition")
webrtc_streamer(key="key", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False},
    async_processing=False, rtc_configuration=RTC_CONFIGURATION,)
#video_stream()
