from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def fcn(image_batch):
    W_fc1 = weight_variable([784,200], name='weight_fc1')
    b_fc1 = bias_variable([200], name='bias_fc1')
    W_fc2 = weight_variable([200,200], name='weight_fc2')
    b_fc2 = bias_variable([200], name='bias_fc2')
    W_out = weight_variable([200,10], name='weight_out')
    b_out = bias_variable([10], name='bias_out')

    hidden_1 = tf.nn.sigmoid(tf.matmul(image_batch, W_fc1) + b_fc1)
    #hidden_2 = tf.nn.sigmoid(tf.matmul(hidden_1, W_fc2) + b_fc2)
    hidden_2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(hidden_1, W_fc2) + b_fc2), 0.5)
    _y = tf.nn.softmax(tf.matmul(hidden_2, W_out) + b_out)
    return _y

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
y = fcn(x)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
vyy_ = tf.argmax(y_, 1)
vyy = tf.argmax(y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
weight_loss = tf.add_n(tf.losses.get_regularization_losses()) if regularizer_ratio != 0 else 0.0
cross_entropy = weight_loss + tf.reduce_mean(tf.reduce_sum(-y_ * tf.log(y + 0.0000001), reduction_indices=[1]))
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cross_entropy)

#plot_images_lables_prediction(mnist.test.images, vyy_, vyy, idx=0, num=25) 

tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('accuray', accuracy)
#tf.summary.image('input', tf.reshape(x, [-1, 28, 28, 1]), 10, collections=None, family=None) 
merged = tf.summary.merge_all()

#training_iteration = 100
#training_iteration = 10000
training_iteration = 50000
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
