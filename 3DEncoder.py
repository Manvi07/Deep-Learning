
# coding: utf-8

# In[1]:


from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from matplotlib.pyplot import cm
import numpy as np
import keras
import h5py
import nibabel as nib


# In[2]:


## iterate in train and test, add the rgb dimention
def add_rgb_dimention(array):
    scaler_map = cm.ScalarMappable(cmap="Oranges")
    array = scaler_map.to_rgba(array)[:, : -1]
    return array


# In[3]:


def load_data(path):
    # with h5py.File(path) as dataset:
        #[:] : slice all the elements of array

    img = nib.load("../PNAS_Smith09_rsn10.nii")
    output_images = np.array(img.dataobj)
    input_images = np.load("../fmriData-50-components-1.npy")
    input_images = input_images.reshape(-1, input_images.shape(-3), input_images.shape(-2), input_images.shape(-1))
    # ouput_images = np.load("../PNAS_Smith09_rsn10.nii")
    X_train,X_test,Y_train,Y_test=train_test_split(input_images, output_images,test_size=0.05)
    x_train = dataset["X_train"][:]
    x_test = dataset["X_test"][:]
    y_train = dataset["y_train"][:]
    y_test = dataset["y_test"][:]

    print ("x_train shape: ", x_train.shape)
    print ("y_train shape: ", y_train.shape)

    print ("x_test shape:  ", x_test.shape)
    print ("y_test shape:  ", y_test.shape)

    # xtrain = np.ndarray((x_train.shape[0], 4096, 3))
    # xtest = np.ndarray((x_test.shape[0], 4096, 3))
    #
    # for i in range(x_train.shape[0]):
    #     xtrain[i] = add_rgb_dimention(x_train[i])
    # for i in range(x_test.shape[0]):
    #     xtest[i] = add_rgb_dimention(x_test[i])
    #
    # xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3)
    # xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)
    #
    # #One hot encoding
    # y_train = keras.utils.to_categorical(y_train)
    # y_test = keras.utils.to_categorical(y_test)

    return xtrain, y_train, xtest, y_test


# In[4]:


xtrain, y_train, xtest, y_test = load_data('./full_dataset_vectors.h5')


# In[5]:


print(xtrain.shape)
y_train.shape


# In[11]:


def Encoder(input_layer):
    Econv1_1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', name = "block1_conv1")(input_layer)
    Econv1_1 = BatchNormalization()(Econv1_1)
    Econv1_2 = Conv3D(16, (3, 3, 3), activation='relu', padding='same',  name = "block1_conv2")(Econv1_1)
    Econv1_2 = BatchNormalization()(Econv1_2)
    pool1 = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same', name = "block1_pool1")(Econv1_2)

    Econv2_1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name = "block2_conv1")(pool1)
    Econv2_1 = BatchNormalization()(Econv2_1)
    Econv2_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name = "block2_conv2")(Econv2_1)
    Econv2_2 = BatchNormalization()(Econv2_2)
    pool2= MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), padding='same', name = "block2_pool1")(Econv2_2)

    Econv3_1 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name = "block3_conv1")(pool2)
    Econv3_1 = BatchNormalization()(Econv3_1)
    Econv3_2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name = "block3_conv2")(Econv3_1)
    Econv3_2 = BatchNormalization()(Econv3_2)
    pool3 = MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2), padding='same', name = "block3_pool1")(Econv3_2)

    flatten_layer = Flatten()(pool3)
    fc_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
    #fc_layer1 = Dropout(0.4)(fc_layer1)
    fc_layer2 = Dense(units=512, activation='relu')(fc_layer1)
    #fc_layer2 = Dropout(0.4)(fc_layer2)
    output_layer = Dense(units=10, activation = 'softmax')(fc_layer2)
    encoded = Model(inputs = input_layer, outputs = output_layer)
    return encoded



# In[12]:


input_layer = Input(xtrain.shape[1:])


# In[13]:


model = Encoder(input_layer)
model


# In[14]:


model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.001), metrics=['acc'])


# In[10]:


model.fit(x=xtrain, y=y_train, batch_size=128, epochs=2, validation_split=0.2, verbose=1)
