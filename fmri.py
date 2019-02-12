import numpy as np
from scipy.ndimage import zoom
import  nibabel as nib
import math
import sys
from keras.layers.normalization import BatchNormalization
from keras.layers.core import *
from keras.layers import Input, Dense, Flatten, Dropout, merge, Reshape, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Add
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

img = nib.load("./PNAS_Smith09_rsn10.nii")
output_images = np.array(img.dataobj)
input_images = np.load("/media/biometric/Data21/FM/Project/D_L/Micaai/Data/Neuro_Image_2_class/extracted/normal-20.npy")

num_classes = output_images.shape[3]
# num_subjects = 

output_images = np.moveaxis(output_images, -1, 0)

output_images = zoom(output_images, (1, input_images.shape[1]/output_images.shape[1], input_images.shape[2]/output_images.shape[2], input_images.shape[3]/output_images.shape[3]))
print(output_images.shape, input_images.shape)

ActualOverlap = np.zeros(input_images.shape[0])
for k, i in enumerate(input_images):
	minOverlap = 0
	# print("-------------------------------")
	for index, j in enumerate(output_images):
		# i = sigmoid(i)
		# j = sigmoid(j)
		min_array = np.minimum(i, j)
		sum_array = 1.0*(i+j)/2
		overlap_array = 1.0*min_array/(sum_array+1)
		overlap = np.sum(overlap_array)
		# print(overlap)
		if overlap > minOverlap:
			ActualOverlap[k] = index
			minOverlap = overlap

print("--------------------------")

for i in ActualOverlap:
	print(i)
	
ActualOverlap = keras.utils.to_categorical(ActualOverlap)
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
	fc1 = Dense(1000, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(flatten)
	fc2 = Dense(500,  activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(fc1)
	fc3 = Dense(num_classes, activation='softmax', use_bias=True, kernel_initializer='normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(fc2)
	print(type(fc3.shape))
	encoded = Model(inputs = input_image, outputs = (fc3))

	return encoded
print(ActualOverlap.shape)

input_layer = Input(shape = (input_images.shape[1], input_images.shape[2], input_images.shape[3], 1))
model = CNN(input_layer)

model.compile(loss = categorical_crossentropy, optimizer=Adam(lr=0.001), metrics=['acc'])
model.summary()

checkpoint = ModelCheckpoint("./model.h5", monitor="val_loss", verbose=1, save_best_only=True, mode = 'min')
callbacks_list = [checkpoint]
input_images = np.reshape(input_images, (input_images.shape[0], input_images.shape[1], input_images.shape[2], input_images.shape[3], 1))
model.fit(x=input_images, y=ActualOverlap, batch_size=128, epochs=50, validation_split=0.2, shuffle=True, verbose=1, callbacks = callbacks_list)
