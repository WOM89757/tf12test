from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def imshow(img):
    plt.imshow(np.reshape(img, [28,28]))
    plt.show()

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

for index in range(10):
    print(mnist.train.images[index].shape)
    print(mnist.train.labels[index])
    print(np.nonzero(mnist.train.labels[index][0]))
    #imshow(mnist.train.images[index])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fcnn(image_batch):
    W_fc1 = weight_variable([784,200])
    b_fc1 = bias_variable([200])
    W_fc2 = weight_variable([200,200])
    b_fc2 = bias_variable([200])
    W_out = weight_variable([200,10])
    b_out = bias_variable([10])

    hidden_1 = tf.nn.sigmoid(tf.matmul(image_batch, W_fc1) + b_fc1)
    hidden_2 = tf.nn.sigmoid(tf.matmul(hidden_1, W_fc2) + b_fc2)
    _y = tf.nn.softmax(tf.matmul(hidden_2, W_out) + b_out)
    return _y

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
y = fcnn(x)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_ * tf.log(y + 0.0000001), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

training_iteration = 50000
batch_size = 64
display_step = 50
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for iteration in range(training_iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, current_accuracy = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        if iteration % display_step == 0:
            print(('iteration: %.5d | accuracy: %.6f') % (iteration + 1, current_accuracy)) 
    print('Test Accuracy: %.6f' % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
