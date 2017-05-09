import tensorflow as tf
import numpy as np
import sys
import time


def countdown(secs, buffer=sys.stdout):
    for i in xrange(secs, 0, -1):
        buffer.write(str(i) + '...')
        buffer.flush()
        time.sleep(1)

checkpoint_file = './checkpoints/iris.ckpt'

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code


num_test = int(0.15 * x_data.shape[0])

x_test  = x_data[:num_test]
x_valid = x_data[-num_test:]
x_data = x_data[num_test:-num_test]

y_test  = y_data[:num_test]
y_valid = y_data[-num_test:]
y_data  = y_data[num_test:-num_test]

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x  = tf.placeholder("float", [None, x_data.shape[1]])  # samples
y_ = tf.placeholder("float", [None, y_data.shape[1]])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.relu(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.AdamOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

#saver.restore(sess, checkpoint_file)

try:

    print "----------------------"
    print "   Start training...  "
    print "    Ctrl+C to skip    "
    print "----------------------"
    print ""
    countdown(3)

    batch_size = 20

    for epoch in xrange(400):
        for jj in xrange(len(x_data) / batch_size):
            batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
            batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
            sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        if epoch % 50 == 0:
            saver.save(sess, checkpoint_file)

        print "Epoch #:", epoch, "\tLoss: ", sess.run(loss, feed_dict={x: x_valid, y_: y_valid})

except KeyboardInterrupt:
    print "-----------------"
    print "Skipping training"
    print "-----------------"

saver.restore(sess, checkpoint_file)

result = sess.run(y, feed_dict={x: x_test})
errors = 0.
for b, r in zip(y_test, result):
    b = np.argmax(b)
    r = np.argmax(r)

    if b != r:
        print b, '-->', r , '<<<<<<<<<<<<<<< Err'
        errors += 1
    else:
        print b, '-->', r

print 'Errors: ', errors, '/', len(y_test), ' Accuracy: ', 100. * (1. - errors/len(y_test)), '%'
