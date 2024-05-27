from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout, Flatten
from keras.models import Model
from extra_keras_datasets import stl10
from skimage.io import imread, imsave
 
import numpy as np
import tensorflow as tf
from PIL import Image
import os

(x_train, _), (x_test, _) = stl10.load_data()
 
# Normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_image = Input(shape=(96, 96, 3))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image) #96 x 96 x 32
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1) #96 x 96 x 32
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1) #96 x 96 x 32
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1) #96 x 96 x 32
conv1 = BatchNormalization()(conv1)
conv1 = MaxPooling2D(pool_size=(2, 2))(conv1) #48 x 48 x 32
conv1 = Dropout(0.4)(conv1)
 
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1) #48 x 48 x 64
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2) #48 x 48 x 64
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2) #48 x 48 x 64
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2) #48 x 48 x 64
conv2 = BatchNormalization()(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2))(conv2) #24 x 24 x 64
conv2 = Dropout(0.4)(conv2)
 
conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2) #24 x 24 x 64
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #24 x 24 x 64
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #24 x 24 x 64
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #24 x 24 x 64
conv3 = BatchNormalization()(conv3)
conv3 = MaxPooling2D(pool_size=(2, 2))(conv3) #12 x 12 x 64
conv3 = Dropout(0.4)(conv3)
 
conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3) #12 x 12 x 32
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4) #12 x 12 x 32
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4) #12 x 12 x 32
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4) #12 x 12 x 32
 
encoded = BatchNormalization()(conv4)

conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4) #12 x 12 x 32
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5) #12 x 12 x 32
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5) #12 x 12 x 32
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5) #12 x 12 x 32
conv5 = BatchNormalization()(conv5)
conv5 = UpSampling2D((2,2))(conv5) #24 x 24 x 32
 
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #24 x 24 x 64
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6) #24 x 24 x 64
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6) #24 x 24 x 64
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6) #24 x 24 x 64
conv6 = BatchNormalization()(conv6)
conv6 = UpSampling2D((2,2))(conv6) #48 x 48 x 64
 
conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6) #48 x 48 x 64
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7) #48 x 48 x 64
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7) #48 x 48 x 64
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7) #48 x 48 x 64
conv7 = BatchNormalization()(conv7)
conv7 = UpSampling2D((2,2))(conv7) #96 x 96 x 64
 
conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7) #96 x 96 x 32
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8) #96 x 96 x 32
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8) #96 x 96 x 32
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8) #96 x 96 x 32
conv8 = BatchNormalization()(conv8)
 
decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv8) #96 x 96 x 3
 
model = Model(input_image , decoder)
model.compile(loss='mean_squared_error', optimizer='nadam')

# Training
model.fit(x_train, x_train, batch_size=16, epochs=50, validation_data=(x_test, x_test), shuffle=True)

model.save("model.keras")

"""
img = imread("input.png")

model = tf.keras.models.load_model("model.keras")

test = [(img/255.0).tolist(), (img/255.0).tolist()]

print(test)

decoded_imgs = model.predict(test)

img = Image.fromarray((decoded_imgs[0]* 255).astype(np.uint8))
img.save('res.png')
"""
"""
img = Image.fromarray((x_test[1]* 255).astype(np.uint8))
img.save('input1.png')
"""