#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from keras.utils import np_utils


# In[2]:


filter_size1 = 5
num_filters1 = 16

filter_size2 = 5
num_filters2 = 36

fc_size = 128


# In[3]:


#Load data
X_train = np.load('/home/manvi/Academics/ADL/Day1/Codes/Data/mnist/x_train.npy')
X_test = np.load('/home/manvi/Academics/ADL/Day1/Codes/Data/mnist/x_test.npy')
y_train_cls = np.load('/home/manvi/Academics/ADL/Day1/Codes/Data/mnist/y_train.npy')
y_test_cls = np.load('/home/manvi/Academics/ADL/Day1/Codes/Data/mnist/y_test.npy')


# In[4]:


img_size_flat = X_train.shape[1]*X_train.shape[2]
img_size = X_train.shape[1]
img_shape = [X_train.shape[1],X_train.shape[2]]


# In[5]:


#one-hot encoding
y_train= np_utils.to_categorical(y_train_cls)
y_test = np_utils.to_categorical(y_test_cls)
num_classes = y_test.shape[1]


# In[38]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255


# In[39]:


num_channels = 1


# In[40]:


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# In[41]:


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[42]:


def new_conv_layer(input,                #The previous layer
                   num_input_channels,   # Num. channels in prev. layer.
                   filter_size,          #Width and height of each filter
                   num_filters,
                   use_pooling = True):  #use 2X2 max-pooling
    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    #The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    #padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1],padding='SAME')
    # A bias-value is added to each filter-channel.
    layer += biases

    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    layer = tf.nn.relu(layer)
    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.
    return layer, weights


# In[43]:


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])
    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]
    return layer_flat, num_features


# In[44]:


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


# In[45]:


x = tf.placeholder(tf.float32, shape=[None, img_size, img_size], name='x')
x_image = tf.reshape(x, [-1, img_size,img_size, num_channels])


# In[46]:


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_true_cls = tf.argmax(y_true, axis=1)


# In[47]:


layer_conv1, weights_conv1= new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=False)


# In[48]:


layer_conv1


# In[49]:


layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=False)


# In[50]:


layer_conv2


# In[51]:


layer_flat, num_features = flatten_layer(layer_conv2)


# In[52]:


layer_flat   #1764 = 7 x 7 x 36.


# In[53]:


num_features


# In[54]:


#FULLY CONNECTED LAYER 1
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs= fc_size, use_relu=True)


# In[55]:


layer_fc1


# In[56]:


layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


# In[57]:


layer_fc2


# In[58]:


y_pred = tf.nn.softmax(layer_fc2)


# In[59]:


y_pred_cls = tf.argmax(y_pred, axis =1)


# In[60]:


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)


# In[61]:


cost = tf.reduce_mean(cross_entropy)


# In[62]:


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# In[63]:


correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# In[64]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[65]:


session = tf.Session()


# In[66]:


session.run(tf.global_variables_initializer())


# In[67]:


train_batch_size = 10
# X_train = X_train.reshape(X_train.shape[0], img_size_flat).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], img_size_flat).astype('float32')
print(y_train.shape)


# In[68]:
def print_accuracy():
    acc = session.run(accuracy, feed_dict = feed_dict_test)
    print("Accuracy on test-set:{0:.1%}".format(acc))

# batch_size = 64
total_iterations = 0
def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    batch_size = train_batch_size
    for i in range(total_iterations, total_iterations+num_iterations):
        for img in range(batch_size, X_train.shape[0], batch_size):
            x_batch = X_train[img-batch_size:img]
#             print(x_batch.shape)
            y_true_batch=y_train[img-batch_size:img]
            feed_dict_train = {x:x_batch, y_true:y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)

        acc = session.run(accuracy, feed_dict = feed_dict_train)
        msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
        print(msg.format(i + 1, acc))
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

# def optimize(num_iterations):
#     for j in range(num_iterations):
#         print("Iteration: " + str(j))
#         for i in range(batch_size, X_train.shape[0], batch_size):
#             x_batch = X_train[i-batch_size:i]
#             y_true_batch = y_train[i-batch_size:i]
#             feed_dict_train = {x: x_batch, y_true: y_true_batch}
#             session.run(optimizer, feed_dict=feed_dict_train)
#         print_accuracy()

# In[69]:


optimize(num_iterations = 4)


# In[ ]:
