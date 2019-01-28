import os
import cv2
from keras.layers.core import *
from keras.layers import Input, Dense, Flatten, Dropout, merge, Reshape, Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose, ZeroPadding2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop, SGD, Adam
from keras import regularizers, backend as k
import numpy as np
import scipy
import numpy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')       #Generate images without having a window appear
import matplotlib.pyplot as plt
from skimage.filters import threshold_mean

k.set_image_data_format('channels_last')

x_shape = 256
y_shape = 256
channels = 3

#Encoder##############################################################################
def Encoder(input_img):
    Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name="block1_conv1")(input_img)
    Econv1_1 = BatchNormalization()(Econv1_1)
    Econv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2')(Econv1_1)
    Econv1_2 = BatchNormalization()(Econv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same', name = "block1_pool1")(Econv1_2)

    Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv1")(pool1)
    Econv2_1 = BatchNormalization()(Econv2_1)
    Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name="block2_conv2")(Econv2_1)
    Econv2_2 = BatchNormalization()(Econv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding = 'same', name="block2_pool1")(Econv2_2)

    Econv3_1 = Conv2D(128, (3, 3),  activation='relu', padding='same', name="block3_conv1")(pool2)
    Econv3_1 = BatchNormalization()(Econv3_1)
    Econv3_2 = Conv2D(128, (3, 3),  activation='relu', padding='same', name="block3_conv2")(Econv3_1)
    Econv3_2 = BatchNormalization()(Econv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same', name="block3_pool1")(Econv3_2)

    encoded = Model(inputs = input_img, outputs = pool3)
    return encoded

#BottleNeck##############################################################################################
def neck(input_layer):
    Nconv = Conv2D(256, (3, 3), padding='same', name="neck1")(input_layer)
    Nconv = BatchNormalization()(Nconv)
    Nconv = Conv2D(128, (3, 3), padding='same', name = "neck2")(Nconv)
    Nconv = BatchNormalization()(Nconv)

    neck_model = Model(inputs=input_layer, outputs=Nconv)
    return neck_model

def Decoder(inp):
    up1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name = "upsample_1")(inp)
    up1 = BatchNormalization()(up1)

    Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_1")(up1)
    Upconv1_1 = BatchNormalization()(Upconv1_1)
    Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_2")(Upconv1_1)
    Upconv1_2 = BatchNormalization()(Upconv1_2)

    up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_2")(Upconv1_2)
    up2 = BatchNormalization()(up2)

    Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_1")(up2)
    Upconv2_1 = BatchNormalization()(Upconv2_1)
    Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_2")(Upconv2_1)
    Upconv2_2 = BatchNormalization()(Upconv2_2)

    up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_3")(Upconv2_2)
    up3 = BatchNormalization()(up3)
    # up3 = merge([up3, inp[1]], mode='concat', concat_axis=3, name = "merge_3")
    Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_1")(up3)
    Upconv3_1 = BatchNormalization()(Upconv3_1)
    Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_2")(Upconv3_1)
    Upconv3_2 = BatchNormalization()(Upconv3_2)

    decoded = Conv2D(1, (3, 3), activation="sigmoid" , padding='same', name = "Ouput_layer")(Upconv3_2)
    convnet = Model(inputs = inp, outputs = decoded)
    return convnet



input_img = Input(shape = (x_shape, y_shape,channels))
encoded = Encoder(input_img)                #Creates an instance of Encoder object which takes in input and gives output

HG_ = Input(shape =( x_shape/(2**3), y_shape/(2**3), 128))
decoded = Decoder( HG_)

Neck_input = Input(shape = (x_shape/(2**3), y_shape/(2**3),128))
Neck = neck(Neck_input)

output_img = decoded(Neck(encoded(input_img)))
model = Model(inputs = input_img, outputs = output_img)
model.summary()
model.compile(optimizer = Adam(lr = 0.0005, decay = 0.0001), loss='binary_crossentropy', metrics= ["accuracy"])

name = os.listdir("./Leaf/singleLeaf/output/")
input_images = []
output_images = []
print("loading_images")
count =0

for i in name:
    if os.path.exists("./Leaf/singleLeaf/output/"+str(i)):
        img_X = cv2.imread("./Leaf/singleLeaf/output/"+str(i), 0 )
        img_X = cv2.resize(img_X, (256, 256))
        img_X = img_X[:, :, np.newaxis]
        input_images.append(img_X)
        # cv2.imwrite("./"+str(i)+".bmp", img_X)

        _, img_y = cv2.threshold(cv2.imread("./Leaf/singleLeaf/output/"+str(i), 0), 127,255,cv2.THRESH_BINARY)
        img_y = cv2.resize(img_y, (256, 256))
        img_y = img_y[:, :, np.newaxis]
        output_images.append(img_y)
        # vis = np.concatenate((img_X, img_y), axis=1)
        cv2.imwrite("./Leaf/singleLeaf/groundTruth/"+str(i), img_y)

print(input_images[0].shape)
print("data splitting: ")
X_train,X_test,Y_train,Y_test=train_test_split(input_images, output_images,test_size=0.001)
# del input_images
# del output_images

# X_train, X_test, Y_train, Y_test = train_test_split(input_images, output_images, test_size=0.001)
del input_images, output_images

X_train = np.asarray(X_train, np.float16)/255
print("X_train.shape: ", X_train.shape)
X_test = np.asarray(X_test, np.float16)/255
print("X_test: ", X_test.shape)
Y_train = np.asarray(Y_train, np.float16)/255
print("Y_train: ", Y_train.shape)
Y_test = np.asarray(Y_test, np.float16)/255
print("Y_test: ", Y_test.shape)

saveModel = "./model.h5"

batch_size = 8
num_batches = int(len(X_train)/batch_size)
print("Number of batches: ", num_batches)

# saveDir = './'
loss=[]
val_loss=[]
acc=[]
val_acc=[]
epoch=0
best_loss=1000
r_c=0
n = 100

# checkpoint = ModelCheckpoint("./model.h5", monitor = "val_loss", verbose=1, save_best_only=True, mode='min' )
# callbacks_list = [checkpoint]
# while epoch < n:
#     history = model.fit(X_train, Y_train, batch_size, epochs=1, validation_data=(X_test, Y_test), shuffle = True, verbose=1, callbacks = callbacks_list)
#     epoch = epoch + 1
#     print("Epoch no. ", str(epoch) + "/"+ str(n))
#     loss.append(float(history.history['loss'][0]))
#     val_loss.append(float(history.history['val_loss'][0]))
#     acc.append(float(history.history['acc'][0]))
#     val_acc.append(float(history.history['val_acc'][0]))
#     loss_arr = np.asarray(loss)
#     e = range(epoch)
#     plt.plot(e, loss_arr)
#     plt.xlabel("number of epochs")
#     plt.ylabel("Training loss")
#     plt.savefig("./LossCurve/"+str(epoch)+'.png')
#     plt.close()
#
#     loss1=np.asarray(loss)
#     val_loss1=np.asarray(val_loss)
#     acc1=np.asarray(acc)
#     val_acc1=np.asarray(val_acc)
#
#     np.savetxt('./Loss.txt',loss1)
#     np.savetxt('./Val_Loss.txt',val_loss1)
#     np.savetxt('./Acc.txt',acc1)
#     np.savetxt('./Val_Acc.txt',val_acc1)
#
#     s=numpy.random.randint(len(X_test))
#     x_test=X_test[s,:,:,:]
#     x_test=x_test.reshape(1,256,256,1)
#     mask_img = model.predict(x_test)
#     x_test = x_test.reshape(256,256)
#     mask_img = mask_img.reshape((256,256))
#     temp = np.zeros([256,256*2])
#     temp[:,:256] = x_test[:,:]+mask_img[:,:]
#     temp[:,256:256*2] = x_test[:,:]
#     temp = temp*255
#     mask_img=mask_img*255
#     cv2.imwrite( "./Images/"+str(epoch+1) + ".bmp", temp)
#     cv2.imwrite( "./Images/"+ str(epoch+1) + ".png", mask_img)

print("training Done.")

# model.save_weights(saveModel)
