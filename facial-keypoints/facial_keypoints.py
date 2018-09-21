import sys
sys.path.append("..")
import helpers
helpers.mask_busy_gpus(wait=False)

import os
from os.path import join
import cv2
import numpy as np
import pandas as pd

def load_imgs_and_keypoints(dirname='data'):
    # Write your code for loading images and points here
    imgs=[]
    heights=[]
    widths=[]
    for img in os.listdir(dirname+"/images"):
        img=cv2.imread(join(dirname,"images",img))
        heights.append(img.shape[0])
        widths.append(img.shape[1])
        img=cv2.resize(img,dsize=(100,100))  #in BGR
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(RGB_img)
    ret_imgs=np.stack(imgs)
    kp=pd.read_csv(join('data','gt.csv'))
    kp=kp.iloc[:,1:]
    for ix in range(kp.shape[1]):
        if ix%2==0:
            kp.iloc[:,ix]=kp.iloc[:,ix]/widths
        else:
            kp.iloc[:,ix]=kp.iloc[:,ix]/heights
    kp=kp-0.5
    ret_kps=kp.values
    return ret_imgs,ret_kps

imgs, points = load_imgs_and_keypoints()






# ### Train/val split



from sklearn.model_selection import train_test_split
imgs_train, imgs_val, points_train, points_val = train_test_split(imgs, points, test_size=0.1)







def flip_img(img, img_points):
    img=cv2.flip(img,1)
    new_points=np.copy(img_points)
    for ix in range(len(new_points)):
        if ix%2==0:
            new_points[ix]*=-1
    return img,new_points

f_img, f_points = flip_img(imgs[1], points[1])


# Time to augment our training sample. Apply flip to every image in training sample. As a result you should obtain two arrays: `aug_imgs_train` and `aug_points_train` which contain original images and points along with flipped ones.


new_imgs,new_points=[],[]
for img,img_points in zip(imgs_train,points_train):
    f_img, f_points = flip_img(img, img_points)
    new_imgs.append(img)
    new_imgs.append(f_img)
    new_points.append(img_points)
    new_points.append(f_points)
    
aug_imgs_train=np.stack(new_imgs)
aug_points_train=np.stack(new_points)
aug_imgs_train.shape,aug_points_train.shape



# ### Network architecture and training
# 
# Now let's define neural network regressor. It will have 28 outputs, 2 numbers per point. The precise architecture is up to you. We recommend to add 2-3 (`Conv2D` + `MaxPooling2D`) pairs, then `Flatten` and 2-3 `Dense` layers. Don't forget about ReLU activations. We also recommend to add `Dropout` to every `Dense` layer (with p from 0.2 to 0.5) to prevent overfitting.
# 




from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,LeakyReLU,Dropout,Flatten,BatchNormalization
from keras import regularizers

input_img=Input(shape=aug_imgs_train[1].shape)

x=Conv2D(32,(3,3),strides=2,padding='same',kernel_initializer='he_uniform')(input_img)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.2)(x)



x=Conv2D(64,(2,2),padding='same',kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.2)(x)

x=MaxPooling2D((2,2))(x)


x=Conv2D(128,(3,3),strides=2,kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.2)(x)


x=Conv2D(256,(3,3),kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.2)(x)

x=MaxPooling2D((2,2))(x)

x=Flatten()(x)

x=Dense(256,name='denseA',kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.2)(x)
x=Dropout(0.3)(x)

x=Dense(512,name='denseB',kernel_initializer='he_uniform')(x)
x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.2)(x)
x=Dropout(0.4)(x)

regressed=Dense(28,name='output')(x)

regressor=Model(input_img, regressed)





regressor.summary()



from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import SGD, Adam
checkpoint=ModelCheckpoint('current_best_model3.h5',save_best_only=True)
e_stop=EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15, verbose=1, mode='min',baseline=0.03)

regressor.compile(optimizer=Adam(lr=0.001,decay=5e-6), loss='mse')
regressor.fit(aug_imgs_train,aug_points_train,epochs=1000,batch_size=32,shuffle=True,validation_data=(imgs_val, points_val),callbacks=[checkpoint])


regressor.save_weights('keypoints_regressor_trained_weights3.h5')
