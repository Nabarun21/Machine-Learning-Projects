import sys
sys.path.append("..")
import helpers

helpers.mask_busy_gpus()

import tensorflow as tf
import keras, keras.layers as L, keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from lfw_dataset import load_lfw_dataset
import numpy as np


# In[ ]:



def reset_tf_session():
    K.clear_session()
    tf.reset_default_graph()
    s = K.get_session()
    return s


# # Load dataset
# - http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
# - http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
# - http://vis-www.cs.umass.edu/lfw/lfw.tgz

# In[ ]:




# load images
X = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)
IMG_SHAPE = X.shape[1:]

# center images
X = X.astype('float32') / 255.0 - 0.5

# split
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)



del X
import gc
gc.collect()





def build_deep_autoencoder(img_shape, code_size):
    """PCA's deeper brother. See instructions above. Use `code_size` in layer definitions."""
    H,W,C = img_shape
    
    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(32,(3,3),padding='same',activation='elu'))
    encoder.add(L.MaxPool2D((2,2),padding='same'))
    print(encoder.output_shape)
    encoder.add(L.Conv2D(64,(3,3),padding='same',activation='elu'))
    encoder.add(L.MaxPool2D((2,2),padding='same'))
    print(encoder.output_shape)
    encoder.add(L.Conv2D(128,(3,3),padding='same',activation='elu'))
    encoder.add(L.MaxPool2D((2,2),padding='same'))
    print(encoder.output_shape)
    encoder.add(L.Conv2D(256,(3,3),padding='same',activation='elu'))
    encoder.add(L.MaxPool2D((2,2),padding='same'))
    print(encoder.output_shape)
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size,activation='elu'))
    ### YOUR CODE HERE: define encoder as per instructions above ###

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(4*256,activation='elu'))
    decoder.add(L.Reshape((2,2,256)))
    #decoder.add(L.Conv2DTranspose(256,(3,3),padding='same',activation='elu',strides=2,))
    #print(decoder.output_shape)
    decoder.add(L.Conv2DTranspose(128,(3,3),padding='same',activation='elu',strides=2,))
    print(decoder.output_shape)
    decoder.add(L.Conv2DTranspose(64,(3,3),padding='same',activation='elu',strides=2,))
    print(decoder.output_shape)
    decoder.add(L.Conv2DTranspose(32,(3,3),padding='same',activation='elu',strides=2,))
    print(decoder.output_shape)
    decoder.add(L.Conv2DTranspose(3,(3,3),padding='same',strides=2,))
    print(decoder.output_shape)
    ### YOUR CODE HERE: define decoder as per instructions above ###
    
    return encoder, decoder


# In[ ]:


# Check autoencoder shapes along different code_sizes
get_dim = lambda layer: np.prod(layer.output_shape[1:])
for code_size in [1,8,32,128,512]:
    s = reset_tf_session()
    encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=code_size)
    print("Testing code size %i" % code_size)
    assert encoder.output_shape[1:]==(code_size,),"encoder must output a code of required size"
    assert decoder.output_shape[1:]==IMG_SHAPE,   "decoder must output an image of valid shape"
    assert len(encoder.trainable_weights)>=6,     "encoder must contain at least 3 layers"
    assert len(decoder.trainable_weights)>=6,     "decoder must contain at least 3 layers"
    
    for layer in encoder.layers + decoder.layers:
        assert get_dim(layer) >= code_size, "Encoder layer %s is smaller than bottleneck (%i units)"%(layer.name,get_dim(layer))

print("All tests passed!")
s = reset_tf_session()


# In[ ]:


# Look at encoder and decoder shapes.
# Total number of trainable parameters of encoder and decoder should be close.
s = reset_tf_session()
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
encoder.summary()
decoder.summary()


# Convolutional autoencoder training. This will take **1 hour**. You're aiming at ~0.0056 validation MSE and ~0.0054 training MSE.

# In[ ]:


s = reset_tf_session()

encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')



autoencoder.fit(x=X_train, y=X_train, epochs=50,
                validation_data=[X_test, X_test],
                verbose=1,
                )


# In[ ]:


reconstruction_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
print("Convolutional autoencoder MSE:", reconstruction_mse)


# save trained weights
encoder.save_weights("encoder.h5")
decoder.save_weights("decoder.h5")


