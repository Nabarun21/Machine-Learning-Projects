import sys
sys.path.append("..")
import helpers
helpers.mask_busy_gpus(wait=False)

from keras import backend as K




import numpy as np
from skimage import transform
import cv2





from get_data import load_dataset, unpack




train_images, train_bboxes, train_shapes = load_dataset("data", "train")
val_images, val_bboxes, val_shapes = load_dataset("data", "val")





SAMPLE_SHAPE = (32, 32, 3)





from scores import iou_score # https://en.wikipedia.org/wiki/Jaccard_index

def is_negative_bbox(new_bbox, true_bboxes, eps=1e-1):
    """Check if new bbox not in true bbox list.
    
    There bbox is 4 ints [min_row, min_col, max_row, max_col] without image index."""
    
    for bbox in true_bboxes:
        if iou_score(new_bbox, bbox) >= eps:
            return False
    return True



# Write this function
def gen_negative_bbox(image_shape,bbox_size=None,):
    """Generate negative bbox for image."""
    if not bbox_size:
        size=np.random.choice(np.arange(30,41))
    else:
        size=bbox_size
    max_row,max_col=image_shape
    
    min_row=np.random.choice(np.arange(0,max_row-size))
    min_col=np.random.choice(np.arange(0,max_col-size))
    
    return [min_row,min_col,min_row+size,min_col+size]

def get_positive_negative(images, true_bboxes, image_shapes, negative_bbox_count=None):
    """Retrieve positive and negative samples from image."""
    positive = []
    negative = []
    image_count = image_shapes.shape[0]
    #first get positive images
    for j in range(len(images)):
        image=images[j]
        true_bboxes_curr=true_bboxes[true_bboxes[:,0]==j][:,1:]
        
        for box in true_bboxes_curr:
            pos_sample=image[box[0]:box[2]+1,box[1]:box[3]+1,:]
            pos_sample_resized=cv2.resize(pos_sample,(32,32))
            positive.append(pos_sample_resized)
    if negative_bbox_count is None:
        negative_bbox_count = len(true_bboxes)
    i=0
    for dummy in range(negative_bbox_count):
        index=np.random.choice(image_count)
        image=images[index]
        image_shape=image_shapes[index]
        if image_shape[0]<50 or image_shape[1]<50:continue
        true_bboxes_curr=true_bboxes[true_bboxes[:,0]==index][:,1:]
       
        for _ in range(100):
            neg_box=gen_negative_bbox(image_shape)
            if is_negative_bbox(neg_box,true_bboxes_curr,0.05):
                neg_sample=image[neg_box[0]:neg_box[2],neg_box[1]:neg_box[3],:]
                neg_sample_resized=cv2.resize(neg_sample,(32,32))
                negative.append(neg_sample_resized)
                i+=1
                break
   
    return positive, negative



def get_samples(images, true_bboxes, image_shapes):
    """Usefull samples for learning.
    
    X - positive and negative samples.
    Y - one hot encoded list of zeros and ones. One is positive marker.
    """
    positive, negative = get_positive_negative(images=images, true_bboxes=true_bboxes, 
                                               image_shapes=image_shapes)
    X = positive
    Y = [[0, 1]] * len(positive)
    
    X.extend(negative)
    Y.extend([[1, 0]] * len(negative))
    
    return np.array(X), np.array(Y)




X_train, Y_train = get_samples(train_images, train_bboxes, train_shapes)
X_val, Y_val = get_samples(val_images, val_bboxes, val_shapes)



BATCH_SIZE = 64


from keras.preprocessing.image import ImageDataGenerator # Usefull thing. Read the doc.

datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             rotation_range=10
                            )
datagen.fit(X_train)



import os.path
from keras.optimizers import Adam
# Very usefull, pay attention
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from graph import save_hist


def fit(model, datagen, X_train, Y_train, X_val, Y_val, model_name=None, output_dir="/afs/crc.nd.edu/user/n/ndev/ML_exercises/face-detection/data/checkpoints", class_weight=None, epochs=2000, lr=0.0003):
    """Fit model.
    
    You can edit this function anyhow.
    """
    

    model.compile(optimizer=Adam(lr=lr,decay=2e-5), # You can use another optimizer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    model.summary()
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  validation_data=(datagen.standardize(X_val), Y_val),
                                  epochs=epochs, steps_per_epoch=len(X_train) // BATCH_SIZE,
                                  callbacks=[ModelCheckpoint(os.path.join(output_dir, "{model_name}").format(model_name=model_name) + "-{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True),
                                            ] if model_name is not None else [],
                                  class_weight=class_weight,
                                  verbose=1
                      
                                 )  # starts training
    
    save_hist(history,"5")




import keras
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Activation, Input, Dropout, Activation, BatchNormalization,LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import regularizers

def generate_model(sample_shape):
    # Classification model

    input_img=Input(shape=sample_shape)
    x=Conv2D(32,(3,3),kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.03))(input_img)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=Conv2D(64,(3,3),kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.03))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.03))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D()(x)
    x=Conv2D(128,(3,3),kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.03))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=Conv2D(256,(3,3),kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.03))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D()(x)
    x=Conv2D(256,(3,3),kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.03))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=BatchNormalization()(x)
    x=Conv2D(512,(2,2),kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.03))(x)
    x=LeakyReLU(alpha=0.1)(x)
    x=Flatten()(x)
    x=Dense(512, activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.03))(x)
    predictions = Dense(2, activation='softmax')(x)
    
    return Model(inputs=input_img, outputs=predictions)


model = generate_model(SAMPLE_SHAPE)


fit(model_name="nabnet_5", model=model, datagen=datagen, X_train=X_train, X_val=X_val, Y_train=Y_train, Y_val=Y_val)


