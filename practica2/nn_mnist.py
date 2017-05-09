import tensorflow as tf
import cPickle
import gzip
import numpy as np
import sys
import time


def countdown(secs, buffer=sys.stdout):
    for i in xrange(secs, 0, -1):
        buffer.write(str(i) + '...')
        buffer.flush()
        time.sleep(1)

checkpoint_file = './checkpoints/mnist.ckpt'

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

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x,  test_y  = test_set


train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)


x  = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(100, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

saver.restore(sess, checkpoint_file)

try:

    print "----------------------"
    print "   Start training...  "
    print "    Ctrl+C to skip    "
    print "----------------------"
    print ""
    countdown(5)

    batch_size = 20

    for epoch in xrange(400):
        for jj in xrange(len(train_x) / batch_size):
            batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
            batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
            sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

        if epoch % 50 == 0:
            saver.save(sess, checkpoint_file)

        print "Epoch #:", epoch, "\tLoss: ", sess.run(loss, feed_dict={x: valid_x, y_: valid_y})

except KeyboardInterrupt:
    print ""
    print "-----------------"
    print "Skipping training"
    print "-----------------"

saver.restore(sess, checkpoint_file)

result = sess.run(y, feed_dict={x: test_x})
errors = 0.
for b, r in zip(test_y, result):
    b = np.argmax(b)
    r = np.argmax(r)

    if b != r:
        print b, '-->', r , '<<<<<<<<<<<<<<< Err'
        errors += 1
    else:
        print b, '-->', r

print 'Errors: ', errors, '/', len(test_y), ' Accuracy: ', 100. * (1. - errors/len(test_y)), '%'
