
from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import time
import os
import ops 
from utils import *
import cv2

import sys
sys.path.append("..")
import helpers
helpers.mask_busy_gpus(wait=False)

DATA_PATH = ['./img_align_celeba_b/','./img_align_celeba/'] # Path to the dataset with celebA faces
Z_DIM=100 # Dimension of face's manifold
GENERATOR_DENSE_SIZE=1024 # Length of first tensor in generator

IMAGE_SIZE=64 # Shapes of input image
BATCH_SIZE=32 # Batch size
N_CHANNELS = 3 # Number channels of input image

MERGE_X = 8 # Number images in merged image



assert(os.path.exists(DATA_PATH[0])), 'Please, download aligned celebA to DATA_PATH folder'
assert(os.path.exists(DATA_PATH[1])), 'Please, download aligned celebA to DATA_PATH folder'




#center crop to 150*120 and resize to 64,64 the images to make them more manageable
def crop_and_scale(img):
    cropped_img=img[34:184,29:149]
    return cv2.resize(cropped_img,(64,64))




def transform(arr):
    arr=arr/float(127.5)-1.
    return arr

def get_image_batch(data_path,batch_size=32):
    ret_arr=[]
    ix=np.random.choice([0,1])
    for i in np.random.randint(len(os.listdir(data_path[ix])), size = batch_size):
        img=transform(crop_and_scale(plt.imread(data_path[ix]+os.listdir(data_path[ix])[i])))
        ret_arr.append(img)
    return np.array(ret_arr)




def generator(z, is_training):
    # Firstly let's reshape input vector into 3-d tensor. 
    

    z_ = ops.linear(z, GENERATOR_DENSE_SIZE*4*4, 'g_h0_lin')
    h_in = tf.reshape(z_, [-1, 4, 4, GENERATOR_DENSE_SIZE])
    g_batch_norm_in=ops.batch_norm(name='g_batch_norm_in')
    h_in_bn = g_batch_norm_in(h_in,is_training)
    h_in_z=ops.lrelu(x=h_in_bn,name='g_lr_1')
        
    h_1=ops.deconv2d(h_in_z,output_shape=[BATCH_SIZE,8,8,512],k_h=5,k_w=5,d_h=2, d_w=2,name="g_deconv_1")
    g_batch_norm_1=ops.batch_norm(name='g_batch_norm_1')
    h_1_bn = g_batch_norm_1(h_1,is_training)
    h_1_z=ops.lrelu(x=h_1_bn,name='g_lr_2')
    h_1_z_dr=tf.nn.dropout(h_1_z,0.3)
    
    h_2=ops.deconv2d(h_1_z_dr,output_shape=[BATCH_SIZE,16,16,256],k_h=5,k_w=5,d_h=2, d_w=2,name="g_deconv_2")
    g_batch_norm_2=ops.batch_norm(name='g_batch_norm_2')
    h_2_bn = g_batch_norm_2(h_2,is_training)
    h_2_z=ops.lrelu(x=h_2_bn,name='g_lr_3')
    h_2_z_dr=tf.nn.dropout(h_2_z,0.3)
    
    h_3=ops.deconv2d(h_2_z_dr,output_shape=[BATCH_SIZE,32,32,128],k_h=5,k_w=5,d_h=2, d_w=2,name="g_deconv_3")
    g_batch_norm_3=ops.batch_norm(name='g_batch_norm_3')   
    h_3_bn = g_batch_norm_3(h_3,is_training)
    h_3_z=ops.lrelu(x=h_3_bn,name='g_lr_4')
    
    h_out = ops.deconv2d(h_3_z, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS],
            name='g_out')

    return tf.nn.tanh(h_out)




def discriminator(image, is_training, batch_norms=None):

    dh_in = ops.conv2d(image,output_dim=32,k_h=5,k_w=5,d_h=2, d_w=2,name="d_conv_1")
    d_batch_norm_1=ops.batch_norm(name='d_batch_norm_1')
    dh_in_bn = d_batch_norm_1(dh_in,is_training)
    dh_in_z=ops.lrelu(x=dh_in_bn,name='d_lr_1')
    
    dh_1=ops.conv2d(dh_in_z,output_dim=64,k_h=5,k_w=5,d_h=2, d_w=2,name="d_conv_2")
    d_batch_norm_2=ops.batch_norm(name='d_batch_norm_2')
    dh_1_bn = d_batch_norm_2(dh_1,is_training)
    dh_1_z=ops.lrelu(x=dh_1_bn,name='d_lr_2')
    
    dh_2=ops.conv2d(dh_1_z,output_dim=128,k_h=5,k_w=5,d_h=2, d_w=2,name="d_conv_3")
    d_batch_norm_3=ops.batch_norm(name='d_batch_norm_3')
    dh_2_bn =d_batch_norm_3(dh_2,is_training)
    dh_2_z=ops.lrelu(x=dh_2_bn,name='d_lr_3')
    
    dh_3=ops.conv2d(dh_2_z,output_dim=256,k_h=5,k_w=5,d_h=2, d_w=2,name="d_conv_4")
    d_batch_norm_4=ops.batch_norm(name='d_batch_norm_4')
    dh_3_bn =d_batch_norm_4(dh_3,is_training)
    dh_3_z=ops.lrelu(x=dh_3_bn,name='d_lr_4')
    
    dh_4=ops.conv2d(dh_3_z,output_dim=512,k_h=5,k_w=5,d_h=2, d_w=2,name="d_conv_5")
    d_batch_norm_5=ops.batch_norm(name='d_batch_norm_5')
    dh_4_bn =d_batch_norm_5(dh_4,is_training)
    dh_4_z=ops.lrelu(x=dh_4_bn,name='d_lr_5')
    
    d_flat=tf.contrib.layers.flatten(dh_3_z)
    
    d_linear=ops.linear(d_flat, 256, 'd_lin_1')
    d_batch_norm_6=ops.batch_norm(name='d_batch_norm_6')
    d_linear_bn=d_batch_norm_6(d_linear,is_training)
    d_linear_z=ops.lrelu(x=d_linear_bn,name='d_lr_6')

    
    linear_out=ops.linear(d_linear_z,1,'d_lin_out')

    return tf.nn.sigmoid(linear_out), linear_out




tf.reset_default_graph()
is_training = tf.placeholder(tf.bool, name='is_training')

with tf.variable_scope("G") as scope:
    z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')

    G = generator(z, is_training)
    
with tf.variable_scope('D') as scope:
    images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
    
    # If you use batch norms from ops define them here (like batch_norms = [batch_norm(name='d_bn0')])
    # and pass to discriminator function instances.
    D_real, D_real_logits = discriminator(images, is_training)
    scope.reuse_variables()
    D_fake, D_fake_logits = discriminator(G, is_training)   

d_loss_real = -tf.reduce_mean(tf.log_sigmoid(D_real_logits))

d_loss_fake = -tf.reduce_mean(tf.log(1-D_fake+1e-30))

surity1=np.random.uniform(0.8,1.15) #label smoothing
surity2=np.random.uniform(-1.0,0.2)

d_loss = surity1*d_loss_real + (1-surity2)*d_loss_fake

g_loss = -tf.reduce_mean(tf.log_sigmoid(D_fake_logits))




tvars = tf.trainable_variables()
## All variables of discriminator
d_vars = [v for v in tvars if 'd_' in v.name]

print d_vars
## All variables of generator
g_vars = [v for v in tvars if 'g_' in v.name]

print g_vars
LEARNING_RATE = 0.0002 # Learning rate for adam optimizer
BETA = 0.5 # Beta paramater in adam optimizer

##Optimizers - ypu may use your favourite instead.
d_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA)                   .minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA)                   .minimize(g_loss, var_list=g_vars) 





data = glob(os.path.join(DATA_PATH[0], "*.jpg"))
print(len(data))
assert(len(data) > 0), "Length of training data should be more than zero"


# Functions for training and evaluations.

# In[43]:


def load(sess, load_dir,latest_fname=None):
    """load network's paramaters
    
    load_dir : path to load dir
    """
    saver = tf.train.Saver()
    # ckpt = tf.train.get_checkpoint_state(load_dir,latest_filename=latest_fname)
    # if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    saver.restore(sess,os.path.join(load_dir,latest_fname))





def train(sess, load_dir=None, save_frequency=100, sample_frequency=100, sample_dir='sample_faces3',
          save_dir='checkpoint3', max_to_keep=30, model_name='dcgan.model3',
          n_epochs=25, n_generator_update=2,latest_fname=None):
    """train gan
the plots in the paper/pas that giovanni was referring to have been fixed. as to his other question about why we haven't included the morphing plots this is my idea of a reply:

"

Dear Giovanni

The plots in the paper/pas have been fixed. Regarding the morphing, a detailed list of plots and observed an expected limits have been uploaded to the twiki and can be found in [1]. Our understanding was that we were doing the morphing as an exercise in order to see the behaviour at the intermediate mass points and if they deviate a lot from the simply interpolated (without morphed points) plots that we currently have. Given that the simply interpolated and morphed plots do not appear to vary all that much we thought it better to keep the plots without the morphing in the interest of keeping things simpler. However, we are open to discussing this further with you and the ARC.

""    Parameters
    -------------------------------------------
    load_dir : str, default = None
        path to the folder with parameters
    save_frequency: int, default = 100
        how often save parameters []
    sample_frequency: int, default = None (not sample)
        how often sample faces
    sample_dir: str, default = samples
        directory for sampled images
    save_dir: str, default = 'checkpoint'
        path where to save parameters
    max_to_keep: int, default = 1
        how many last checkpoints to store
    model_name: str, default='dcgan.model'
        name of model
    n_epochs: int, default = 25 
        number epochs to train
    n_generator_update: int, default = 2
        how many times run generator updates per one discriminator update
    -------------------------------------------
    """
    
    if save_frequency is not None:
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        
    if load_dir is not None:
        print("Reading checkpoints...")
        load(sess,load_dir,latest_fname)
        print("Loaded checkpoints")
    else:
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

    counter=1
    start_time = time.time()
    
    for epoch in range(n_epochs):
        batch_idxs = min(len(data), np.inf) // BATCH_SIZE
        
        for idx in range(batch_idxs*2):
           # batch_files = data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
           #batch = [get_image(batch_file, IMAGE_SIZE) for batch_file in batch_files]
           # batch_images = np.array(batch).astype(np.float32)
            batch_images=get_image_batch(DATA_PATH,BATCH_SIZE)
            batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)

            # Update D network
            sess.run(d_optim, feed_dict={images: batch_images, z: batch_z,is_training: True})
            
            if epoch>12:n_generator_update=1
            # Update G network
            for _ in range(n_generator_update):
                sess.run(g_optim,
                    feed_dict={z: batch_z, is_training: True})

            errD_fake = d_loss_fake.eval({z: batch_z, is_training: False})
            errD_real = d_loss_real.eval({images: batch_images, is_training: False})
            errG = g_loss.eval({z: batch_z, is_training: False})

            counter += 1
            print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

            if np.mod(counter, save_frequency) == 1:
                print("Saved model")
                saver.save(sess, 
                           os.path.join(save_dir, model_name+"_"+str(counter)))

            if np.mod(counter, sample_frequency) == 1:
                samples = sess.run(G, feed_dict={z: batch_z, is_training: False} )
                save_images(samples, [MERGE_X//2, MERGE_X],
                            os.path.join(sample_dir, 'train_{:02d}_{:04d}.png'.format(epoch, idx)))
                print("Sampled")





with tf.Session() as sess:
#    train(sess,load_dir='checkpoint3',save_dir='checkpoint3',sample_frequency=500,save_frequency=50,latest_fname=None)
    train(sess,save_dir='checkpoint3',sample_frequency=500,save_frequency=3000)

