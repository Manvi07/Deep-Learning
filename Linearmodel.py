#!/usr/bin/env python
# coding: utf-8

# In[1]:

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
# from sklearn.metrics import confusion_matrix


# In[2]:

#Load data
X_train = np.load('./Day1/Codes/Data/mnist/x_train.npy')
X_test = np.load('./Day1/Codes/Data/mnist/x_test.npy')
y_train_cls = np.load('./Day1/Codes/Data/mnist/y_train.npy')
y_test_cls = np.load('./Day1/Codes/Data/mnist/y_test.npy')


# In[3]:

img_size_flat = X_train.shape[1]*X_train.shape[2]
img_shape = [X_train.shape[1], X_train.shape[2]]


# In[4]:

#flattening images into linear array
X_train = X_train.reshape(X_train.shape[0], img_size_flat).astype('float32')
X_test = X_test.reshape(X_test.shape[0], img_size_flat).astype('float32')


# In[5]:

#Normalise
X_train = X_train/255
X_test = X_test/255


# In[6]:

#one-hot encoding
y_train= np_utils.to_categorical(y_train_cls)
y_test = np_utils.to_categorical(y_test_cls)


# In[7]:

#number of classes
num_classes = y_test.shape[1]


# In[8]:

#TENSORFLOW GRAPH

#Define placeholders
# Placeholder variables serve as the input to the graph that we may change each time we execute the graph.
# x: for storing Images
# y_true : for ground truths
# y_true_cls : for class labels
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])


# In[9]:

# VARIABLES TO BE OPTIMISED
# Initialise weights and biases
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))
W2 = tf.Variable(tf.zeros([num_classes, num_classes]))
B2 = tf.Variable(tf.zeros([num_classes]))


# In[10]:

#MODEL
#add layers(feeding the output of one as input to the other.)
logits1 = tf.nn.softmax(tf.matmul(x, weights) + biases)
#apply activation function before feeding to the next layer
logits2 = tf.matmul(logits1, W2) + B2


# In[11]:

#Y_pred = predicted values of all neurons
y_pred = tf.nn.softmax(logits2)

#argmax to find the max of the values in neurons as that will correspond to the class predicted
y_pred_cls = tf.argmax(y_pred, axis=1)


# In[12]:

#cost function to be optimised
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=y_true)


# In[13]:

#in order to use the cross-entropy to guide the optimization of the model's variables we need a single scalar value,
#so we simply take the average of the cross-entropy for all the image classifications.
cost = tf.reduce_mean(cross_entropy)


# In[14]:

#Optimiser (we just add the optimizer-object to the TensorFlow graph for later execution.)
optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(cost)      # As learning rate decreases the rate of change of gradient decreases. Thus Accuracy increases.


# In[15]:

#PERFORMANCE MEASURES

#a vector of booleans whether the predicted class equals the true class of each image.
correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# In[16]:

#This calculates the classification accuracy by first type-casting the vector of booleans to floats, so that False becomes 0 and True becomes 1,
#and then calculating the average of these numbers.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[17]:

# TensorFlow Session
session = tf.Session()


# In[18]:

#Initialise Variables
session.run(tf.global_variables_initializer())


# In[19]:


batch_size = 1000

#Put the batch into a dict with the proper names
# for placeholder variables in the TensorFlow graph.
feed_dict_test = {x:X_test,
                  y_true: y_test,
                  y_true_cls: y_test_cls}


# In[20]:


def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.2%}".format(acc))


# In[21]:

#Function for performing a number of optimization iterations so as to gradually improve the weights and biases of the model.
def optimize(num_iterations):
    for img in range(num_iterations):
        for i in range(batch_size, X_train.shape[0], batch_size):
            x_batch = X_train[i-batch_size:i]
            y_true_batch = y_train[i-batch_size:i]
            feed_dict_train = {x:x_batch, y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
        print_accuracy()


# In[22]:

#Function for plotting the weights of the model. 10 images are plotted, one for each digit that the model is trained to recognize.
#Positive weights are red and negative weights are blue. These weights can be intuitively understood as image-filters.
def plot_weights():
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        if i<10:
            image = w[: ,i].reshape(img_shape)
            ax.set_xlabel("Weights :{0}".format(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show(block=True)


# In[23]:


optimize(num_iterations=2)


# In[24]:


print_accuracy()


# In[25]:


# plot_weights()


# In[26]:


optimize(num_iterations=10)


# In[27]:


print_accuracy()


# In[28]:


# plot_weights()


# In[29]:


optimize(num_iterations=20)


# In[30]:


print_accuracy()


# In[31]:


plot_weights()


# In[ ]:
