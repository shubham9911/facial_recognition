#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2

#img = cv2.imread('abc.jpg')

#resized_img = cv2.resize(img,(600,600))


# In[4]:


#new_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)

#cv2.imshow("shubh",new_img)

#cv2.waitKey(0)

#cv2.destroyAllWindows()


# In[3]:


import cv2
import numpy as np

video = cv2.VideoCapture('http://192.168.0.101:8080/video')


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier("C:\\Users\\bunty\\Desktop\\Untitled Folder\\.ipynb_checkpoints\\haar-cascade-files-master\\haar-cascade-files-master\\haarcascade_eye.xml")
#smile_cascade = cv2.CascadeClassifier("C:\\Users\\bunty\\Desktop\\Untitled Folder\\.ipynb_checkpoints\\haar-cascade-files-master\\haar-cascade-files-master\\haarcascade_smile1.xml")

skip = 0
face_data = []
label = []
data_path = '.\\data\\'
file_name = input("Enter the name of the person in frame:  ")
names = {}

while True:
    ret,frame = video.read()
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR)
    if ret == False:
        continue
        
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    #eye = eye_cascade.detectMultiScale(frame,1.3,5)
    #smile = smile_cascade.detectMultiScale(frame,1.05,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])
    
    for (x,y,w,h) in faces[-1:]:
       
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    offset = 10   
    face = frame[y-offset:y+h+offset,x-offset:x+w+offset]
    face_section = cv2.resize(face,(100,100))
    
    skip += 1
    if skip%10==1:
        face_data.append(face_section)
        print(len(face_data))
        
    #for (ex,ey,ew,eh) in eye:
     #    cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
   
   # for (sx,sy,sw,sh) in smile:
    #    cv2.rectangle(frame,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    
    
    
    cv2.imshow("web_cam",frame)
    cv2.imshow("face_section",face_section)
    
    key = cv2.waitKey(1)
    
    if skip==400:
        break
        
        
face_data = np.asarray(face_data)        
face_data = face_data.reshape((face_data.shape[0],-1))

print(face_data.shape)

np.save(data_path+file_name+'.npy',face_data)

print("data is susessfully saved at",data_path+file_name+'.npy')
        
        
        
        
video.release()

cv2.destroyAllWindows()

      



# In[ ]:





# In[ ]:





# In[ ]:




