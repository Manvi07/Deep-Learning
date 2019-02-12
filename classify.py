import numpy as np
from scipy.ndimage import zoom
import math
import sys
from keras.layers.normalization import BatchNormalization
from keras.layers.core import *
from keras.layers import Input, Dense, Flatten, Dropout, merge, Reshape, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Add
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


normal_images = np.load("/media/biometric/Data21/FM/Project/D_L/Micaai/Data/Train/Neuro_Image_2_Class/extracted/normal-20.npy")
patient_images = np.load("/media/biometric/Data21/FM/Project/D_L/Micaai/Data/Train/Neuro_Image_2_Class/extracted/patient-20.npy")

num_classes = 2

x_train = []
y_train = []

for i in range(1, 460, 20):
	x_train.append(normal_images[i])
	y_train.append(1)

for i in range(1, 460, 20):
	x_train.append(patient_images[i])
	y_train.append(0)

x_train = np.array(x_train, dtype = "float32")
y_train = np.array(y_train, dtype = "float32")
	
y_train = keras.utils.to_categorical(y_train)
print(x_train.shape, y_train.shape)

def CNN(input_image):
	EConv1_1 = Conv3D(16, (3,3, 3), activation = 'relu', padding='same', name="block1_conv1")(input_image)
	EConv1_1 = BatchNormalization()(EConv1_1)
	EConv1_2 = Conv3D(16, (3, 3, 3), activation = 'relu', padding = 'same', name = "block1_conv2")(EConv1_1)
	EConv1_2 = BatchNormalization()(EConv1_2)
	pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2,2, 2), padding = 'same', name = 'block1_pool1')(EConv1_2)

	Econv2_1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name="block2_conv1")(pool1)
	Econv2_1 = BatchNormalization()(Econv2_1)
	Econv2_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name="block2_conv2")(Econv2_1)
	Econv2_2 = BatchNormalization()(Econv2_2)
	pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), padding = 'same', name="block2_pool1")(Econv2_2)

	Econv3_1 = Conv3D(128, (3, 3 ,3),  activation='relu', padding='same', name="block3_conv1")(pool2)
	Econv3_1 = BatchNormalization()(Econv3_1)
	Econv3_2 = Conv3D(128, (3, 3, 3),  activation='relu', padding='same', name="block3_conv2")(Econv3_1)
	Econv3_2 = BatchNormalization()(Econv3_2)
	pool3 = MaxPooling3D(pool_size=(2,2, 2), strides=(2, 2, 2), padding = 'same', name="block3_pool1")(Econv3_2)

	flatten = Flatten()(pool3)
	fc1 = Dense(256, activation='sigmoid', use_bias=True, kernel_initializer='normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(flatten)
	fc2 = Dense(128,  activation='sigmoid', use_bias=True, kernel_initializer='normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(fc1)
	fc3 = Dense(num_classes, activation='sigmoid', use_bias=True, kernel_initializer='normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(fc2)
	print(type(fc3.shape))
	encoded = Model(inputs = input_image, outputs = (fc3))

	return encoded

input_layer = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))
model = CNN(input_layer)

model.compile(loss = categorical_crossentropy, optimizer=Adam(lr=0.01), metrics=['acc'])
model.summary()

# checkpoint = ModelCheckpoint("./classificationmodel.h5", monitor="val_loss", verbose=1, save_best_only=True, mode = 'min')
# callbacks_list = [checkpoint]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))

x_train, y_train = shuffle(x_train, y_train)
# print(y_train)
history=model.fit(x=x_train, y=y_train, batch_size=40, epochs=50, validation_split=0.2, shuffle = True, verbose=1)
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('./plot'  +'.png')
model.save_weights("./classificationmodel.h5")
