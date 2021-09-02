from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers 
from keras import models
from functools import partial

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

def _after_conv(in_tensor):
    norm = layers.BatchNormalization()(in_tensor)
    return layers.Activation('relu')(norm)

def conv3(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(in_tensor)
    return _after_conv(conv)

def conv3_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same')(in_tensor)
    return _after_conv(conv)

def conv1(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(in_tensor)
    return _after_conv(conv)

def conv1_downsample(in_tensor, filters):
    conv = layers.Conv2D(filters, kernel_size=1, strides=2, padding='same')(in_tensor)
    return _after_conv(conv)

def resnet_block(in_tensor, filters, downsample=False):
    if downsample:
        conv1_rb = conv3_downsample(in_tensor, filters)
    else:
        conv1_rb = conv3(in_tensor, filters)
    conv2_rb = conv3(conv1_rb, filters)

    if downsample:
        in_tensor = conv1_downsample(in_tensor, filters)
    result = layers.Add()([conv2_rb, in_tensor])

    return layers.Activation('relu')(result)

def block(in_tensor, filters, n_block, downsample=False):
    res = in_tensor
    for i in range(n_block):
        if i == 0:
            res = resnet_block(in_tensor, filters, downsample)
        else:
            res = resnet_block(res, filters, False)
    return res

def resnet(image_batch, netname):
    resnet_config = {
                    'ResNet18':[2, 2, 2, 2],
                    'ResNet34':[3, 4, 6, 3],
                    'ResNet50':[3, 4, 6, 3],
                    'ResNet101':[3, 4, 23, 3],
                    'ResNet152':[3, 8, 36, 3]
                    }
    layers_dims = resnet_config[netname]

    filter_block1 = [64, 64, 256]
    filter_block2 = [128, 128, 512]
    filter_block3 = [256, 256, 1024]
    filter_block4 = [512, 512, 2048]

    conv = layers.Conv2D(64, 7, strides=2, padding='same')(image_batch) 
    conv =  _after_conv(conv)
    pool1 = layers.MaxPool2D(3, 2, padding='same')(conv)
    conv1_block = block(pool1, 64, 3, False)
    conv2_block = block(conv1_block, 128, 4, True)
    conv3_block = block(conv2_block, 256, 6, True)
    conv4_block = block(conv3_block, 512, 3, True)
    pool2 = layers.GlobalAvgPool2D()(conv4_block) 
    _y = layers.Dense(1000, activation='softmax')(pool2)
    # ResNet-18 & ResNet-34 [64, 128, 256, 512]
    # [2, 2, 2, 2] & [3, 4, 6, 3]
    # ResNet-50 & ResNet-101 & ResNet-152 [256, 512, 1024, 2048]
    # [3, 4, 6, 3] & [3, 4, 23, 3] & [3, 8, 36, 3]

    return _y

x = layers.Input(shape=(224, 224, 3))
y_ = layers.Input(shape=(1000,))
y = resnet(x, netname='ResNet34')
model = models.Model(x, y)
print(model.summary())
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
vyy_ = tf.argmax(y_, 1)
vyy = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
weight_loss = tf.add_n(tf.losses.get_regularization_losses()) if regularizer_ratio != 0 else 0.0
cross_entropy = weight_loss + tf.reduce_mean(tf.reduce_sum(-y_ * tf.log(y + 0.0000001), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cross_entropy)

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
