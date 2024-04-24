import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm

from itertools import chain
import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb

from keras.models import Model, load_model,Sequential
from keras.layers import Input, Dense, UpSampling2D, RepeatVector, Reshape
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras import backend as K

import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('/content/drive/MyDrive/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
import pickle


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, 1)
TRAIN_PATH = '/content/drive/MyDrive/585_project_f04/images/train/c1'

train_ids = next(os.walk(TRAIN_PATH))[2]
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
missing_count = 0
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + '/'+id_+''
    try:
        img = imread(path)
        print(path)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n-missing_count] = img
    except:
#         print(" Problem with: "+path)
        missing_count += 1

X_train = X_train.astype('float32') / 255


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, 1)
VAL_PATH = '/content/drive/MyDrive/585_project_f04/images/val/c1'

test_ids = next(os.walk(VAL_PATH))[2]
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
missing_count = 0
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = VAL_PATH + '/'+id_+''
    try:
        img = imread(path)
        print(path)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n-missing_count] = img
    except:
        missing_count += 1

X_test = X_test.astype('float32') / 255

def Colorize():
    embed_input = Input(shape=(1000,))
    
    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_input)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(128, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    
    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
    
    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(32, (2,2), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output)


# preprocessing
from keras.preprocessing.image import ImageDataGenerator
# Create an ImageDataGenerator object with specified data augmentation parameters
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# Defining a function to generate Inception embeddings for grayscale images
def inception_predict(grayscaled_rgb):
    # A function to resize grayscale images
    def resize_gray(x):
        return resize(x, (299, 299, 3), mode='constant')

    grayscaled_rgb_resized = np.array([resize_gray(x) for x in grayscaled_rgb])
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    # Generating the Inception embeddings
    embed = inception.predict(grayscaled_rgb_resized)
    
    return embed

def generate_preprocess(dataset, batch_size = 20):
    # Generating augmented images using the ImageDataGenerator object
    for batch in datagen.flow(dataset, batch_size=batch_size):
        X_batch = rgb2gray(batch)
        grayscaled_rgb = gray2rgb(X_batch)
        lab_batch = rgb2lab(batch)
        # Extracting the L channel from the LAB images
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        # Extracting and normalizing the AB channels from the LAB images
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield [X_batch, inception_predict(grayscaled_rgb)], Y_batch


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# a ReduceLROnPlateau callback to reduce the learning rate when the loss stops improving
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5,
                                            min_lr=0.00001)

# ModelCheckpoint callback to save the best model during training
checkpoint = ModelCheckpoint("Colorization_Model.h5",
                             save_best_only=True,
                             monitor='loss',
                             mode='min')
model_callbacks = [learning_rate_reduction,checkpoint]

y_val = X_train[-200:]
y_train = X_train[:-200]

batch_size = 20
generator = generate_preprocess(y_val, batch_size)
num_batches = len(y_val) // batch_size

for i in range(num_batches):
    batch = next(generator)
    X_eva = batch[0]
    y_eva = batch[1]

def get_compiled_model(input_model):
    model = input_model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    
model = get_compiled_model(Colorize())

# Training
history = model.fit(generate_preprocess(y_train,20),
            epochs=30,
            verbose=1,
            steps_per_epoch=X_train.shape[0]/20,
             callbacks=model_callbacks,
             validation_data=(X_eva, y_eva)
)


if not os.path.exists('/content/drive/MyDrive/585_project_f04/Models/pickles'):
    os.makedirs('/content/drive/MyDrive/585_project_f04/Models/pickles')

train_losses = history.history['loss']
val_losses = history.history['val_loss']

with open('/content/drive/MyDrive/585_project_f04/Models/pickles/train_val_loss_plot_encoder_resnetv2.pkl', 'wb') as f:
    pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)
# Save losses to a pickle file
if not os.path.exists('/content/drive/MyDrive/585_project_f04/Models/saved_weights'):
    os.makedirs('/content/drive/MyDrive/585_project_f04/Models/saved_weights')

model.save('/content/drive/MyDrive/585_project_f04/Models/saved_models/saved_model_encoder_resnetv2.h5')
model.save_weights('/content/drive/MyDrive/585_project_f04/Models/saved_weights/saved_model_encoder_resnetv2.h5')

output_size = (256, 256)

sample = X_test[:, :output_size[0], :output_size[1], :]

color_me = gray2rgb(rgb2gray(sample))
color_me_embed = inception_predict(color_me)
color_me = rgb2lab(color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))
# Generating the colorized images
output = model.predict([color_me, color_me_embed])
output = output * 128
# Combining the input and output images to create the final colorized images
decoded_imgs = np.zeros((len(output), output_size[0], output_size[1], 3))
for i in range(len(output)):
    cur = np.zeros((output_size[0], output_size[1], 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    decoded_imgs[i] = lab2rgb(cur)

import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

save_dir = '/content/drive/MyDrive/585_project_f04/outputs_inception_resnetv2/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

test_ids = next(os.walk(VAL_PATH))[2]
for i in range(len(test_ids)):
    img = imread(VAL_PATH + '/' + test_ids[i])
    if len(img.shape) != 3:
      img = np.stack([img]*3, axis=-1)
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = img.shape
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    # grayscale
    plt.gray()
    plt.tight_layout()
    img_index = test_ids[i]
    img_index = img_index[:-4]
    fig_name = 'gray_'+str(img_index)+'.png'
    fig_path = os.path.join(save_dir, 'gray/')
    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    resized_img = resize(rgb2gray(X_test)[i], (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    plt.imsave(arr=resized_img, fname='{}{}'.format(fig_path, fig_name), cmap='gray')

    # recolorization
    plt.tight_layout()
    fig_name = 'color_'+str(img_index)+'.jpg'
    fig_path = os.path.join(save_dir, 'color/')
    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    resized_img = resize(decoded_imgs[i], (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    plt.imsave(arr=resized_img, fname='{}{}'.format(fig_path, fig_name))

    # original
    plt.tight_layout()
    fig_name = 'orig_'+str(img_index)+'.jpg'
    fig_path = os.path.join(save_dir, 'ground_truth/')
    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    plt.imsave(arr=img, fname='{}{}'.format(fig_path, fig_name))

    plt.figure(figsize=(20, 6))
for i in range(10):
    # grayscale
    plt.subplot(3, 10, i + 1)
    plt.imshow(rgb2gray(X_test)[i].reshape(256,256))
    plt.gray()
    plt.axis('off')
 
    # recolorization
    plt.subplot(3, 10, i + 1 +10)
    plt.imshow(decoded_imgs[i].reshape(256, 256,3))
    plt.axis('off')
    
    # original
    plt.subplot(3, 10, i + 1 + 20)
    plt.imshow(X_test[i].reshape(256, 256,3))
    plt.axis('off')
 
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/585_project_f04/outputs_inception_resnetv2/colorization_results.jpg')
plt.show()