from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers 
from keras import models

print(tf.__version__)

def imshow(img):
    plt.imshow(np.reshape(img, [28,28]))
    plt.show()
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
#regularizer_ratio = 0.0001
regularizer_ratio = 0.0

def plot_images_lables_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(np.reshape(images[idx], [28, 28]), cmap='binary')
        title = 'lable=' + str(labels[idx])
        if len(prediction) > 0:
            title +=',predict=' + str(prediction[idx])

        ax.set_title(title, fontsize = 10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()

def plot_images(images, labels, prediction):
    fig = plt.gcf()
    ax = plt.subplot(5, 5, 1)
    ax.imshow(np.reshape(images, [28, 28]), cmap='binary')
    title = 'lable=' + str(labels)
    title +=',predict=' + str(prediction)

    ax.set_title(title, fontsize = 10)
    ax.set_xticks([]);
    ax.set_yticks([])
    #plt.show()
    return plt
#tet = plot_images(mnist.train.images[0], mnist.train.labels[0], 1)
#plt.imshow(tet)
#plt.show()

#for index in range(10):
#    print(mnist.train.images[index].shape)
#    print(mnist.train.labels[index])
#    print(np.nonzero(mnist.train.labels[index][0]))
    #imshow(mnist.train.images[index])


def weight_variable(shape, name):
    if regularizer_ratio != 0.0:
        regularizer = tf.contrib.layers.l2_regularizer(regularizer_ratio)
        initial = tf.truncated_normal(shape, stddev=0.1)
        weight = tf.get_variable(name, initializer=initial, regularizer=regularizer)
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)
        weight = tf.get_variable(name, initializer=initial)
    return weight

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def simple_conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def simple_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def lenet(image_batch):
   # x_image = tf.reshape(image_batch, [-1, 28, 28, 1])
    x_image = layers.Reshape(([28, 28, 1]))(image_batch)
    h_conv1 = layers.Conv2D(filters=6, kernel_size=5, padding='same', activation='relu', use_bias=True)(x_image)
    h_pool1 = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(h_conv1)
    h_conv2 = layers.Conv2D(filters=16, kernel_size=5, padding='valid', activation='relu', use_bias=True)(h_pool1)
    h_pool2 = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(h_conv2)
    h_pool2_flat = layers.Flatten()(h_pool2)
    h_fc1 = layers.Dense(120, activation='relu')(h_pool2_flat)
    h_fc2 = layers.Dense(84, activation='relu')(h_fc1)
    _y = layers.Dense(10, activation='softmax')(h_fc2)
    return _y

#x = tf.placeholder(tf.float32, [None, 784])
#y_ = tf.placeholder(tf.float32, [None, 10])
x = layers.Input(shape=(784,))
y_ = layers.Input(shape=(10,))
y = lenet(x)
model = models.Model(x, y)
print(model.summary())
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
vyy_ = tf.argmax(y_, 1)
vyy = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
weight_loss = tf.add_n(tf.losses.get_regularization_losses()) if regularizer_ratio != 0 else 0.0
cross_entropy = weight_loss + tf.reduce_mean(tf.reduce_sum(-y_ * tf.log(y + 0.0000001), reduction_indices=[1]))
print('here')
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
print('here')
#train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cross_entropy)

print('here')
#plot_images_lables_prediction(mnist.test.images, vyy_, vyy, idx=0, num=25) 

tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('accuray', accuracy)
#tf.summary.image('input', tf.reshape(x, [-1, 28, 28, 1]), 10, collections=None, family=None) 
merged = tf.summary.merge_all()

#training_iteration = 100
training_iteration = 1000
#training_iteration = 50000
batch_size = 64
display_step = 50
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./basic_mnist/', sess.graph)
    sess.run(tf.initialize_all_variables())
    for iteration in range(training_iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, _, current_accuracy = sess.run([merged, train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(summary, iteration)
        if iteration % display_step == 0:
            print(('iteration: %.5d | accuracy: %.6f') % (iteration + 1, current_accuracy)) 
            if current_accuracy > 0.8 and 0:
                vx, yy, vy_, vy = sess.run([x, y_, vyy_, vyy], feed_dict={x: mnist.test.images, y_:  mnist.test.labels})
                plot_images_lables_prediction(mnist.test.images, vy_, vy, idx=0, num=25) 
    
    print('Test Accuracy: %.6f' % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    vx, yy, vy_, vy = sess.run([x, y_, vyy_, vyy], feed_dict={x: mnist.test.images, y_:  mnist.test.labels})
   # plot_images_lables_prediction(mnist.test.images, vy_, vy, idx=0, num=25) 
    writer.close()

import pandas as pd
import numpy as np
from pandas import Series,DataFrame

print(pd.crosstab(vy_, vy, rownames=["labels"], colnames=["predict"]))
df = pd.DataFrame({'label': vy_, 'predict': vy})
print(df[:2])
#plot_images_lables_prediction(mnist.test.images, vy_, vy, idx=50, num=1) 
