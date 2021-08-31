import tensorflow as tf
print(tf.__version__)


#z = tf.add(3, 4)
#print(z)
#
#with tf.Session() as sess:
#    print(sess)
#    result = sess.run(z)
#
#print(result)
#
#a = tf.constant(3)
#b = tf.constant(4)
#
#x = a + b
#print(x)
#
#with tf.Session() as sess:
#    writer = tf.summary.FileWriter('./introduction', sess.graph)
#    print(sess.run(x))
#    writer.close()


import numpy as np
import matplotlib.pyplot as plt

train_X = np.asarray([30.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0])
train_Y = np.asarray([320.0, 360.0, 480.0, 490.0, 546.0, 588.0, 680.0])
train_X /= 100.0
train_Y /= 100.0

def plot_points(x, y, title_name):
    plt.title(title_name);
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)
    plt.show()

def plot_line(w, b, title_name):
    plt.title(title_name)
    plt.xlabel('x')
    plt.ylabel('y')
    x = np.linspace(0.0, 2.0, num=100)
    y = w * x + b;
    plt.plot(x, y)
    plt.show()

#plot_points(train_X, train_Y, title_name='Training Points')

n_samples = len(train_X)
X = tf.placeholder('float')
y = tf.placeholder('float')
W = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')
y_pred = tf.add(tf.multiply(X, W), b)

loss = tf.reduce_sum(tf.pow((y_pred - y), 2)) / (2 * n_samples)
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

training_epochs = 1000
display_step = 50

with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for (x_train, y_train) in zip(train_X, train_Y):
            _, cost, current_W, current_b = sess.run([optimizer, loss, W, b], feed_dict={X: x_train, y: y_train});

        if epoch % display_step == 0:
             print(('Epoch: %04d | Loss: %.6f | W: %.6f | b: %.6f') % (epoch + 1, cost, current_W, current_b))
             #plot_line(current_W, current_b, 'Model Parameter')
    print(('Training Loss: %.6f | W: %.6f | b: %.6f') % (cost, current_W, current_b))

