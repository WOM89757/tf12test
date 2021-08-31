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



simple_conv2d = partial(layers.Conv2D,
                        kernel_size=3, 
                        strides=1, 
                        padding='same', 
                        activation='relu')

simple_maxpool = layers.MaxPooling2D()

def block(in_tensor, filters, n_conv):
    conv_block = in_tensor
    for _ in range(n_conv):
        conv_block = simple_conv2d(filters=filters)(conv_block)
    return simple_maxpool(conv_block)

def vggnet(image_batch):
    block1 = block(image_batch, 64, 2)
    block2 = block(block1, 128, 2)
    block3 = block(block2, 256, 3)
    block4 = block(block3, 512, 3)
    block5 = block(block4, 512, 3)
    flat = layers.Flatten()(block5)
    h_fc1 = layers.Dense(4096, activation='relu')(flat)
    h_fc2 = layers.Dense(4096, activation='relu')(h_fc1)
    _y = layers.Dense(1000, activation='softmax')(h_fc2)
    return _y

x = layers.Input(shape=(224, 224, 3))
y_ = layers.Input(shape=(1000,))
y = vggnet(x)
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
