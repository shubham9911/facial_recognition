# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:08:29 2019

@author: bunty
"""

import numpy as np
import cv2
import os



##Knn algorythm
def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())
    
    
def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        
        ix = train[i, :-1]
        iy = train[i, -1]
        
        d = distance(test, ix)
        dist.append([d, iy])
        
    dk = sorted(dist, key=lambda x: x[0])[:k]

    labels = np.array(dk)[:, -1]
    
    output = np.unique(labels, return_counts=True)
    
    index = np.argmax(output[1])
    
    return output[0][index]
######################################

video = cv2.VideoCapture('http://192.168.0.101:8080/video')

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier("C:\\Users\\bunty\\Desktop\\Untitled Folder\\.ipynb_checkpoints\\haar-cascade-files-master\\haar-cascade-files-master\\haarcascade_eye.xml")
#smile_cascade = cv2.CascadeClassifier("C:\\Users\\bunty\\Desktop\\Untitled Folder\\.ipynb_checkpoints\\haar-cascade-files-master\\haar-cascade-files-master\\haarcascade_smile1.xml")

skip = 0
face_data = []
labels = []
data_path = '.\\data\\'
names = {}

class_id = 0

###Data prep.

for fx in os.listdir(data_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(data_path+fx)
        face_data.append(data_item  )
        
        #Creat labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)
        
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)       
        
trainset= np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

###Testing###
import cv2
while True:
    ret,frame = video.read()
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR)
    
    if ret == False:
        continue  
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    
    faces = sorted(faces,key=lambda f:f[2]*f[3])
    
    for face in faces:
        x,y,w,h = face
        
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))



        out = knn(trainset,face_section.flatten())
        
        
        pred_name = names[int(out)]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    
    cv2.imshow("face",frame)
    key = cv2.waitKey(1)
    
    if key == ord('b'):
       break
    
video.release()
cv2.destroyAllWindows()
       
