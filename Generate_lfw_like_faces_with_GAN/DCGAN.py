
# coding: utf-8

# ### Generating human faces with Adversarial Networks
import sys
sys.path.append("..")
import helpers
helpers.mask_busy_gpus(wait=False)

 

import numpy as np
#Those attributes will be required for the final part of the assignment (applying smiles), so please keep them in mind
#from lfw_dataset2 import load_lfw_dataset 
from lfw_dataset import load_lfw_dataset 
data,attrs = load_lfw_dataset(dimx=36,dimy=36)
#data = load_lfw_dataset(use_raw=True,dimx=36,dimy=36)
print(np.max(data),np.min(data))
#preprocess faces
#data = np.float32(data)
print(data[0])
data=(data-127.5)/float(127.5) #scale to between -1 and 1


print(data[0])
IMG_SHAPE = data.shape[1:]


# In[3]:


#print random image
print(data.shape)


import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.333)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
#s = tf.Session()

import keras
from keras.models import Sequential
from keras import layers as L
from keras.layers import LeakyReLU
from keras import backend as K
K.set_learning_phase(0)


CODE_SIZE = 256

generator = Sequential()
generator.add(L.InputLayer([CODE_SIZE],name='noise'))
generator.add(L.Dense(10*8*8,kernel_initializer='he_normal'))
generator.add(LeakyReLU(alpha=0.2))

generator.add(L.Reshape((8,8,10)))
print(generator.output_shape)
generator.add(L.Conv2DTranspose(64,kernel_size=(5,5),kernel_initializer='he_normal'))
generator.add(LeakyReLU(alpha=0.2))
print(generator.output_shape)
generator.add(L.Dropout(0.2))

generator.add(L.Conv2DTranspose(64,kernel_size=(5,5),kernel_initializer='he_normal'))
print(generator.output_shape)
generator.add(LeakyReLU(alpha=0.2))
generator.add(L.Dropout(0.1))

generator.add(L.Conv2DTranspose(32,kernel_size=(5,5),strides=2,padding='same',kernel_initializer='he_normal'))
print(generator.output_shape)
generator.add(LeakyReLU(alpha=0.2))


generator.add(L.Conv2DTranspose(32,kernel_size=3,kernel_initializer='he_normal'))
print(generator.output_shape)
generator.add(L.Dropout(0.1))
generator.add(LeakyReLU(alpha=0.2))

generator.add(L.Conv2DTranspose(32,kernel_size=3,kernel_initializer='he_normal'))
print(generator.output_shape)
generator.add(LeakyReLU(alpha=0.1))

generator.add(L.Conv2DTranspose(32,kernel_size=3,kernel_initializer='he_normal'))
generator.add(LeakyReLU(alpha=0.1))
print(generator.output_shape)
#generator.add(L.Dropout(0.3))                                                                                                                                                                        \
                                                                                                                                                                                                       

generator.add(L.Conv2D(3,kernel_size=3,activation='tanh',kernel_initializer='glorot_uniform'))
print(generator.output_shape)

generator.summary()


# In[6]:


assert generator.output_shape[1:] == IMG_SHAPE, "generator must output an image of shape %s, but instead it produces %s"%(IMG_SHAPE,generator.output_shape[1:])


# ### Discriminator
# * Discriminator is your usual convolutional network with interlooping convolution and pooling layers
# * The network does not include dropout/batchnorm to avoid learning complications.
# * We also regularize the pre-output layer to prevent discriminator from being too certain.

# In[7]:
discriminator = Sequential()

discriminator.add(L.InputLayer(IMG_SHAPE))

discriminator.add(L.Conv2D(32,(3,3),strides=2,padding='same',kernel_initializer='he_uniform'))
discriminator.add(LeakyReLU(alpha=0.1))
print(discriminator.output_shape)
discriminator.add(L.Conv2D(32,(3,3),strides=2,padding='same',kernel_initializer='he_uniform'))
discriminator.add(LeakyReLU(alpha=0.1))
discriminator.add(L.Conv2D(64,(3,3),strides=2,padding='same',kernel_initializer='he_uniform'))
discriminator.add(LeakyReLU(alpha=0.1))
print(discriminator.output_shape)
discriminator.add(L.AveragePooling2D((2,2),padding='same'))
print(discriminator.output_shape)
discriminator.add(L.Conv2D(128,(3,3),padding='same',kernel_initializer='he_uniform'))
discriminator.add(LeakyReLU(alpha=0.1))
print(discriminator.output_shape)
discriminator.add(L.Flatten())
print(discriminator.output_shape)
discriminator.add(L.Dense(256,activation='tanh',kernel_initializer='glorot_uniform'))
print(discriminator.output_shape)
discriminator.add(L.Dense(2,activation=tf.nn.log_softmax))
discriminator.summary()



noise = tf.placeholder('float32',[None,CODE_SIZE])
real_data = tf.placeholder('float32',[None,]+list(IMG_SHAPE))

logp_real = discriminator(real_data)
print(logp_real.shape)

generated_data = generator(noise)

logp_gen = discriminator(generated_data)
print(logp_gen.shape)


# In[9]:


########################
#discriminator training#
########################

surity1=np.random.uniform(0.8,1.15)
surity2=np.random.uniform(-1.0,0.2)
d_loss = -tf.reduce_mean(surity1*logp_real[:,1] + (1-surity2)*logp_gen[:,0])

#regularize
d_loss += tf.reduce_mean(discriminator.layers[-1].kernel**2)

#tf.summary.scalar('disc_loss', d_loss)

#optimize
disc_optimizer =  tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss,var_list=discriminator.trainable_weights)


# In[10]:


########################
###generator training###
########################

g_loss = -tf.reduce_mean((logp_gen[:,1]))

#tf.summary.scalar('gen_loss', g_loss)
gen_optimizer = tf.train.AdamOptimizer(1e-4).minimize(g_loss,var_list=generator.trainable_weights)

    


# In[11]:


#merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter('./logs',s.graph)

s.run(tf.global_variables_initializer())


# ### Auxiliary functions
# Here we define a few helper functions that draw current data distributions and sample training batches.

# In[12]:

def sample_noise_batch(bsize):
    return np.random.normal(size=(bsize, CODE_SIZE)).astype('float32')

def sample_data_batch(bsize):
    idxs = np.random.choice(np.arange(data.shape[0]), size=bsize)
    return data[idxs]

for _ in range(10):
    print(sample_noise_batch(1))


for i in range(10):
    feed_dict = {
        real_data:sample_data_batch(100),
        noise:sample_noise_batch(100)
    }
    disc_loss,_=s.run([d_loss,disc_optimizer],feed_dict)
    print("Epoch: ",i,"disc loss is: ",disc_loss)

gen_loss=0
best_loss=5


for epoch in range(100000):
    
    feed_dict = {
        real_data:sample_data_batch(100),
        noise:sample_noise_batch(100)
    }
    if gen_loss<5:
        if disc_loss>1:
            for i in range(5):
                s.run(disc_optimizer,feed_dict)
        elif disc_loss>0.2:
            for i in range(4):
                s.run(disc_optimizer,feed_dict)
        else:
            for i in range(3):
                s.run(disc_optimizer,feed_dict)
    if gen_loss<2:
        disc_loss,gen_loss,_=s.run([d_loss,g_loss,gen_optimizer],feed_dict)
    else:
        s.run(gen_optimizer,feed_dict)
        disc_loss,gen_loss,_=s.run([d_loss,g_loss,gen_optimizer],feed_dict)
    print("Epoch: ",epoch,"disc loss is: ",disc_loss,"gen loss is: ",gen_loss)
    
    if epoch %50==0:
        if epoch>249 and ((gen_loss<1.3*best_loss and gen_loss<4) or gen_loss<1.5):
            generator.save_weights("best_weights.h5")
            best_loss=gen_loss
        if gen_loss>4.9:
            generator.load_weights("best_weights.h5")
    
    if epoch%20000==0 and epoch!=0:
        generator.save_weights("generator_weights3_"+str(epoch)+".h5")
        discriminator.save_weights("discriminator_weights3_"+str(epoch)+".h5")
