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





# # Optional: Denoising Autoencoder
# 
# This part is **optional**, it shows you one useful application of autoencoders: denoising. You can run this code and make sure denoising works :) 
# 
# Let's now turn our model into a denoising autoencoder:
# <img src="images/denoising.jpg" style="width:40%">
# 
# We'll keep the model architecture, but change the way it is trained. In particular, we'll corrupt its input data randomly with noise before each epoch.
# 
# There are many strategies to introduce noise: adding gaussian white noise, occluding with random black rectangles, etc. We will add gaussian white noise.

# In[ ]:


def apply_gaussian_noise(X,sigma=0.1):
    """
    adds noise from standard normal distribution with standard deviation sigma
    :param X: image tensor of shape [batch,height,width,3]
    Returns X + noise.
    """
    noise = np.random.standard_normal(X.shape)*sigma#
    return X + noise




s = reset_tf_session()

# we use bigger code size here for better quality
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=512)
assert encoder.output_shape[1:]==(512,), "encoder must output a code of required size"

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inp, reconstruction)
autoencoder.compile('adamax', 'mse')

for i in range(40):
    print("Epoch %i/25, Generating corrupted samples..."%(i+1))
    X_train_noise = apply_gaussian_noise(X_train)
    X_test_noise = apply_gaussian_noise(X_test)
    
    # we continue to train our model with new noise-augmented data
    autoencoder.fit(x=X_train_noise, y=X_train, epochs=1,
                    validation_data=[X_test_noise, X_test],
                    verbose=1)


# In[ ]:
# save trained weights
encoder.save_weights("denoising_encoder.h5")
decoder.save_weights("denoising_decoder.h5")


X_test_noise = apply_gaussian_noise(X_test)
denoising_mse = autoencoder.evaluate(X_test_noise, X_test, verbose=0)
print("Denoising MSE:", denoising_mse)
"""

# # Optional: Image retrieval with autoencoders
# 
# So we've just trained a network that converts image into itself imperfectly. This task is not that useful in and of itself, but it has a number of awesome side-effects. Let's see them in action.
# 
# First thing we can do is image retrieval aka image search. We will give it an image and find similar images in latent space:
# 
# <img src="images/similar_images.jpg" style="width:60%">
# 
# To speed up retrieval process, one should use Locality Sensitive Hashing on top of encoded vectors. This [technique](https://erikbern.com/2015/07/04/benchmark-of-approximate-nearest-neighbor-libraries.html) can narrow down the potential nearest neighbours of our image in latent space (encoder code). We will caclulate nearest neighbours in brute force way for simplicity.

# In[ ]:


# restore trained encoder weights
s = reset_tf_session()
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
encoder.load_weights("encoder.h5")


# In[ ]:


images = X_train
codes = ### YOUR CODE HERE: encode all images ###
assert len(codes) == len(images)


# In[ ]:


from sklearn.neighbors.unsupervised import NearestNeighbors
nei_clf = NearestNeighbors(metric="euclidean")
nei_clf.fit(codes)


# In[ ]:


def get_similar(image, n_neighbors=5):
    assert image.ndim==3,"image must be [batch,height,width,3]"

    code = encoder.predict(image[None])
    
    (distances,),(idx,) = nei_clf.kneighbors(code,n_neighbors=n_neighbors)
    
    return distances,images[idx]


# In[ ]:


def show_similar(image):
    
    distances,neighbors = get_similar(image,n_neighbors=3)
    
    plt.figure(figsize=[8,7])
    plt.subplot(1,4,1)
    show_image(image)
    plt.title("Original image")
    
    for i in range(3):
        plt.subplot(1,4,i+2)
        show_image(neighbors[i])
        plt.title("Dist=%.3f"%distances[i])
    plt.show()


# Cherry-picked examples:

# In[ ]:


# smiles
show_similar(X_test[247])


# In[ ]:


# ethnicity
show_similar(X_test[56])


# In[ ]:


# glasses
show_similar(X_test[63])


# # Optional: Cheap image morphing
# 

# We can take linear combinations of image codes to produce new images with decoder.

# In[ ]:


# restore trained encoder weights
s = reset_tf_session()
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)
encoder.load_weights("encoder.h5")
decoder.load_weights("decoder.h5")


# In[ ]:


for _ in range(5):
    image1,image2 = X_test[np.random.randint(0,len(X_test),size=2)]

    code1, code2 = encoder.predict(np.stack([image1, image2]))

    plt.figure(figsize=[10,4])
    for i,a in enumerate(np.linspace(0,1,num=7)):

        output_code = code1*(1-a) + code2*(a)
        output_image = decoder.predict(output_code[None])[0]

        plt.subplot(1,7,i+1)
        show_image(output_image)
        plt.title("a=%.2f"%a)
        
    plt.show()


# That's it!
# 
# Of course there's a lot more you can do with autoencoders.
# 
# If you want to generate images from scratch, however, we recommend you our honor track on Generative Adversarial Networks or GANs.
"""
