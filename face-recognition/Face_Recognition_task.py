import sys
sys.path.append('..')
import helpers
helpers.mask_busy_gpus(wait=False)


from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
import numpy as np


import cv2
import os
from copy import copy
from collections import Counter
import csv


def load_image_data(dir_name = 'Face_Recognition_data/image_classification'):
    x_train,y_train,x_test,y_test={},{},{},{}
    for filename in os.listdir(dir_name+"/train/images"):
        if 'jpg' not in filename:continue
        curr_img=cv2.imread(dir_name+"/train/images/"+filename,)
        RGB_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        x_train[filename]=RGB_img
    with open(dir_name+"/train/y_train.csv") as f:
        filerader=csv.reader(f)
        for line in filerader:
            y_train[line[0]]=line[1]
    for filename in os.listdir(dir_name+"/test/images"):
        if 'jpg' not in filename:continue
        curr_img=cv2.imread(dir_name+"/test/images/"+filename)

        RGB_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        x_test[filename]=RGB_img
    with open(dir_name+"/test/y_test.csv") as f:
        filerader=csv.reader(f)
        for line in filerader:
            y_test[line[0]]=line[1]
    return x_train, y_train, x_test, y_test


#x_train, y_train, x_test, y_test = load_image_data()
#print('%d'%len(x_train), '\ttraining images')
#print('%d'%len(x_test), '\ttesting images')






def load_video_data(dir_name = 'Face_Recognition_data/video_classification'):
    x_train,y_train,x_test,y_test={},{},{},{}
    for filename in os.listdir(dir_name+"/train/images"):
        curr_img=cv2.imread(dir_name+"/train/images/"+filename,)
        RGB_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        x_train[filename]=RGB_img
    with open(dir_name+"/train/y_train.csv") as f:
        filerader=csv.reader(f)
        for line in filerader:
            y_train[line[0]]=line[1]
    for video_id in os.listdir(dir_name+"/test/videos"):
        x_test[video_id]=[]
        for filename in os.listdir(dir_name+"/test/videos/"+video_id):
            curr_img=cv2.imread(dir_name+"/test/videos/"+video_id+"/"+filename,)
            RGB_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
            x_test[video_id].append(RGB_img)
    with open(dir_name+"/test/y_test.csv") as f:
        filerader=csv.reader(f)
        for line in filerader:
            y_test[line[0]]=line[1]
    return x_train, y_train, x_test, y_test



video_train, train_labels, video_test, test_labels = load_video_data()
print('%d'%len(video_train), '\ttraining images')
print ('%d'%len(video_test), '\ttesting videos')

import skimage.transform as tf
from skimage.transform import rotate
import math

def transform_face(image, eyes):
    """ Your implementation """

    #print(eyes)
    left_eye,right_eye=eyes
    y=-(right_eye[1]-left_eye[1])
    x=right_eye[0]-left_eye[0]
    angle=math.acos(abs(x/math.sqrt(x**2+y**2)))#*180/math.pi
    #print(angle,angle*180/pi)
    if  y>0:angle*=-1
    
    image=rotate(image,angle*180/math.pi,center=eyes[0]) #rotate image around left eye
    
    #get the transformed co-ords of right eye, left eye co-ordinate remains same
    tform_1=tf.SimilarityTransform(translation=[-left_eye[0],-left_eye[1]]) #first translate
    tform_2 = tf.SimilarityTransform(rotation=angle)#rotate
    tform_3 =tf.SimilarityTransform(translation=left_eye) #translate back
    right_eye_new=tform_3(tform_2([tform_1(right_eye)[0][0],-tform_1(right_eye)[0][1]]))
    right_eye=tuple([int(np.ceil(right_eye_new[0][0])),eyes[0][1]])

    eye_dist=right_eye[0]-left_eye[0]
    #crop
   # print(image.shape,eye_dist)
    #print(left_eye[1]-int(2.2*eye_dist),left_eye[1]+int(2.2*eye_dist),left_eye[0]-int(1.25*eye_dist),right_eye[0]+int(1.25*eye_dist))
    image=image[max(0,left_eye[1]-int(2*eye_dist)):min(left_eye[1]+int(2*eye_dist),image.shape[0]),max(0,left_eye[0]-int(1.3*eye_dist)):min(image.shape[1],right_eye[0]+int(1.3*eye_dist))]
    return image





import time
import cv2 as cv


def preprocess_img(img):
    face_cascade = cv.CascadeClassifier('/afs/crc.nd.edu/user/n/ndev/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('/afs/crc.nd.edu/user/n/ndev/.local/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')
    eyeglass_cascade = cv.CascadeClassifier('/afs/crc.nd.edu/user/n/ndev/.local/lib/python3.6/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
    
    
    scales=[1.3,1.25,1.2,1.15,1.1]
    min_neighbs=[5,4,3,2]
    faces=[]
    ret_img=np.copy(img)
    gray = cv.cvtColor(ret_img, cv.COLOR_BGR2GRAY)
    def helper_func1():
        
        for scale in scales:
            for num_neighbs in min_neighbs:
                temp_faces = face_cascade.detectMultiScale(gray, scale, num_neighbs)
                faces.extend(temp_faces)
                return
    
    helper_func1()
    
    if len(faces)==0:
        print("No face found, skipping")
        return ret_img

    scales=[1.3,1.275,1.25,1.225,1.2,1.175,1.15,1.125,1.1,1.075,1.05,1.025]
    min_neighbs=[5,4,3,2,1]
    detected_eyes=[]
    #ret_face=ret_img
    #print(ret_face.shape)
    def helper_func2():
        for face in faces:
            x,y,w,h=face
            
            ret_face=ret_img[int(y*0.85):int(y+h*1.15),int(x*0.95):int(x+w*1.05)]
            
            face_gray=np.copy(gray[int(y):int(y+h),int(x):int(x+w)])
            #print("face_width: ",w)
            for scale in scales:
                for num_neighbs in min_neighbs:
                
                    eyes = eye_cascade.detectMultiScale(face_gray,scale,num_neighbs)
                    if len(eyes)<2:continue
                    eye_centers=[(int(eye[0]+np.ceil(eye[2]/2.)+x),int(eye[1]+np.ceil(eye[3]/2.)+y)) for eye in  eyes]
                
                
                    for l in range(len(eye_centers)):
                        for m in range(len(eye_centers)):
                            if l>=m:continue
                            if eye_centers[l][0]<eye_centers[m][0]:
                                left_eye=eye_centers[l]
                                right_eye=eye_centers[m]
                            else:
                                left_eye=eye_centers[m]
                                right_eye=eye_centers[l]
                            y_diff=-(right_eye[1]-left_eye[1])
                            x_diff=right_eye[0]-left_eye[0]
                            if x_diff==0 and y_diff==0:continue
                            angle=math.acos(abs(x_diff/math.sqrt(x_diff**2+y_diff**2)))#*180/math.pi
                            if  y_diff>0:angle*=-1
                            #print("angle: ",angle*180/pi)
                           # print("dist: ",np.abs(eye_centers[l][0]-eye_centers[m][0]))
#                        if np.abs(eye_centers[l][1]-eye_centers[m][1])<20 and np.abs(eye_centers[l][0]-eye_centers[m][0])>15:
                            if abs(angle*180/math.pi)<35 and w*0.7>np.abs(x_diff)>w*0.25:
                               # print(scale,num_neighbs)
                                detected_eyes.append(left_eye)
                                detected_eyes.append(right_eye)
                                return True,None
            for scale in scales:
                for num_neighbs in min_neighbs:
                #print(scale,num_neighbs)
                    eyes = eyeglass_cascade.detectMultiScale(face_gray,scale,num_neighbs)
                    if len(eyes)<2:continue
                    eye_centers=[(int(eye[0]+np.ceil(eye[2]/2.)+x),int(eye[1]+np.ceil(eye[3]/2.)+y)) for eye in  eyes]
                #print("eye_centers: ",eye_centers)
                    for l in range(len(eye_centers)):
                        for m in range(len(eye_centers)):
                            if l>=m:continue
                            if eye_centers[l][0]<eye_centers[m][0]:
                                left_eye=eye_centers[l]
                                right_eye=eye_centers[m]
                            else:
                                left_eye=eye_centers[m]
                                right_eye=eye_centers[l]
                            y_diff=-(right_eye[1]-left_eye[1])
                            x_diff=right_eye[0]-left_eye[0]
                            if x_diff==0 and y_diff==0:continue
                            angle=math.acos(abs(x_diff/math.sqrt(x_diff**2+y_diff**2)))#*180/math.pi
                            if  y_diff>0:angle*=-1
                            #print("angle: ",angle*180/pi)
                            #print("dist: ",np.abs(eye_centers[l][0]-eye_centers[m][0]))
#                        if np.abs(eye_centers[l][1]-eye_centers[m][1])<20 and np.abs(eye_centers[l][0]-eye_centers[m][0])>15:
                            if abs(angle*180/math.pi)<35 and w*0.7>np.abs(x_diff)>w*0.25:
                                #print(scale,num_neighbs)
                                detected_eyes.append(left_eye)
                                detected_eyes.append(right_eye)
                                #print("glasses: ",scale,num_neighbs,detected_eyes)
                                return True,None
        
        return False,ret_face
    
    eyes_detected,cropped_face=helper_func2()

#    for (ex,ey) in detected_eyes:
#        cv.circle(ret_img,(ex,ey),3,(0,255,0))
    
    #return ret_img
    return transform_face(ret_img,detected_eyes) if eyes_detected else cropped_face

def preprocess_imgs(imgs):
    ret_list=[]
    for img in imgs:
        to_add=preprocess_img(img)
        to_add=cv.resize(to_add,(224,224))
        ret_list.append(to_add)
    #print(ret_list)
    return ret_list
        


import h5py
from keras.models import load_model
model = load_model('face_recognition_model.h5')
model.summary()




def get_layer_output(images, layer = 'fc6'):
    assert len(images.shape)==4, 'Wrong input dimentionality!'
    assert images.shape[1:]==(224,224,3), 'Wrong input shape!'
    
    network_output = model.get_layer(layer).output
    feature_extraction_model = Model(model.input, network_output)
    
    output = feature_extraction_model.predict(images)
    return output




from sklearn.neighbors import KNeighborsClassifier as kNN
from skimage.io import imread
import cv2
from os.path import join
import tqdm
class Classifier():
    def __init__(self, nn_model, layer = 'fc6',k=3):
        """Your implementation"""
        network_output = nn_model.get_layer(layer).output
        self.feature_extraction_model = Model(model.input, network_output)
        self.clf=kNN(n_neighbors=k,weights='distance')
    def fit(self, train_imgs, train_labels):
        """Your implementation"""
        train_array_x=[]
        train_array_y=[]
        i=1
        for key,value in tqdm.tqdm(train_imgs.items()):
            prep_image=preprocess_img(train_imgs[key])

            if prep_image.shape!=train_imgs[key].shape:

                prep_image=cv.resize(prep_image,(224,224))
                train_image_reshape=np.reshape(prep_image,newshape=(1,224,224,3))
                train_img_features=self.feature_extraction_model.predict(train_image_reshape)
                train_array_x.append(train_img_features)
                train_array_y.append(train_labels[key])
        train_array_x=np.array(train_array_x)
        train_array_y=np.array(train_array_y)
        train_array_x=np.reshape(train_array_x,newshape=(-1,4096))
        #print(train_array_x.shape)
        self.clf.fit(train_array_x,train_array_y)
    def classify_images(self, test_imgs):
        """Your implementation"""
        ret_val={}
        for key,value in test_imgs.items():
            prep_image=preprocess_img(test_imgs[key])
            prep_image=cv.resize(prep_image,(224,224))
            prep_image_reshape=np.reshape(prep_image,newshape=(1,224,224,3))
            test_img_features=self.feature_extraction_model.predict(prep_image_reshape)
            predicted_label=self.clf.predict(test_img_features)
            ret_val[key]=predicted_label
       
        return ret_val
    def classify_videos(self, test_video):
        """Your implementation"""
        ret_val={}
        for key,image_list in tqdm.tqdm(test_video.items()):
            curr_labels=[]
            for image in tqdm.tqdm(image_list):
                prep_image=preprocess_img(image)
                if prep_image.shape!=image.shape:#if no face found,skip
                    prep_image=cv.resize(prep_image,(224,224))
                    prep_image_reshape=np.reshape(prep_image,newshape=(1,224,224,3))
                    test_img_features=self.feature_extraction_model.predict(prep_image_reshape)
                    predicted_label=self.clf.predict(test_img_features)
                    curr_labels.append(predicted_label)
            unique_values,value_counts=np.unique(curr_labels,return_counts=True) 
            ret_val[key]=unique_values[np.argmax(value_counts)]

        return ret_val




#img_classifier = Classifier(model)
#img_classifier.fit(x_train, y_train)
#y_out = img_classifier.classify_images(x_test)




#y_out_train = img_classifier.classify_images(x_train)

#print(check_test(y_out_train,y_train)) #first check train accuracy, should be close to one really, kNN just "remembers" stuff




def check_test(output, gt):
    correct = 0.
    total = len(gt)
    for k, v in gt.items():
        if k=='filename':continue 
        if output[k] == v:
            correct += 1
        else:
            "It was ",v," but we predicted ",k

    accuracy = correct / total

    return 'Classification accuracy is %.4f' % accuracy






video_classifier = Classifier(model)
video_classifier.fit(video_train, train_labels)





y_video_out = video_classifier.classify_videos(video_test)





print(y_video_out)



print(check_test(y_video_out, test_labels))

