from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
# pip install keras==2.2.4

def imshow(img, shape=[28, 28]):
    plt.imshow(np.reshape(img, shape))
    plt.show()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#W_conv = weight_variable([3, 3, 1, 32])
#b_conv = bias_variable([32])

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
one_image = mnist.train.images[0]
#imshow(one_image)

input_image = tf.convert_to_tensor(one_image)
image = tf.reshape(input_image, [-1, 28, 28, 1])
#features = tf.nn.conv2d(image, W_conv, strides=[1, 1, 1, 1], padding='SAME')
features = layers.Conv2D(filters=32, kernel_size=3, padding='same')(image)
print(features)

with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    output_features = sess.run(features)[0]
    print(output_features.shape)
for _ in range(output_features.shape[2]):
    imshow(output_features[:, :, _])


