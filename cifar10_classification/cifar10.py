
# coding: utf-8

# # Your first CNN on CIFAR-10
# 
# In this task you will: 
# * define your first CNN architecture for CIFAR-10 dataset
# * train it from scratch
# * visualize learnt filters
# 
# CIFAR-10 dataset contains 32x32 color images from 10 classes: __airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck__:
# <img src="images/cifar10.jpg" style="width:80%">

# # Import stuff

# In[1]:


import sys
sys.path.append("..")


import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import h5py
# In[2]:




from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[51]:


print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)
#print(x_train[0])
#print(y_train[0])


# In[37]:


NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]




# normalize inputs
x_train2 = x_train/255-0.5### YOUR CODE HERE
x_test2 =x_test/255-0.5 ### YOUR CODE HERE

# convert class labels to one-hot encoded, should have shape (?, NUM_CLASSES)
y_train2 =keras.utils.to_categorical(y_train,10) ### YOUR CODE HERE
y_test2 = keras.utils.to_categorical(y_test,10)### YOUR CODE HERE


# # Define CNN architecture

# In[61]:


# import necessary building blocks
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,Input,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def make_model():
    """
    Define your model architecture here.
    """
    input_img=Input(shape=(32,32,3))

    #conv layers
    x=Conv2D(16,(3,3),padding='same')(input_img)
#    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    #x=Dropout(0.25)(x)
       
    x=Conv2D(32,(3,3),padding='same')(x)
 #   x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    
    
    x=MaxPooling2D((2,2))(x)
    x=Dropout(0.25)(x)
    
    x=Conv2D(32,(3,3),padding='same')(x)
  #  x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    
    x=Conv2D(64,(3,3),padding='same')(x)
   # x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.1)(x)
    
    
    x=MaxPooling2D((2,2))(x)
    x=Dropout(0.25)(x)

    x=Flatten()(x)
    
    x=Dense(256,name='denseA')(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Dropout(0.5)(x)
    
    label=Dense(10,name='denseB',activation='softmax')(x)
    
    
    cifar_cnn=Model(input_img, label)

   
    return cifar_cnn


# In[78]:





INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 32
EPOCHS = 10


model = make_model()  # define our model
model.summary()
# prepare model for fitting (loss, optimizer, etc)
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adam(lr=INIT_LR),  # for SGD
    metrics=['accuracy']  # report accuracy during training
)

# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.99999 ** epoch

# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))




# fit model
model.fit(
    x_train2, y_train2,  # prepared data
    batch_size=32,
    epochs=200,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), 
               LrHistory(), 
               ],
    validation_data=(x_test2, y_test2),
    shuffle=True,
    verbose=1,
)


# In[ ]:


# save weights to file
model.save_weights("my_weights.h5")


