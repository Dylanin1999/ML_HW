import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def Kernel(shape):
    weight = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name="conv_kernel")
    return weight


def biases(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias


def CNN_Net(X_train, Y_train, X_test, Y_test, train_batch):

    X = tf.placeholder(tf.float32, [None, 48*48])
    Y = tf.placeholder(tf.float32, [None, 7])

    image = tf.reshape(X, [-1, 48, 48, 1])
    # conv_layer_1
    with tf.name_scope('conv1_lay') as scope:
        Kernel1 = Kernel([5, 5, 1, 64])
        bias1 = biases([64])
        conv1 = tf.nn.conv2d(image, Kernel1, strides=[1, 1, 1, 1], padding='SAME', name=scope)
        activation1 = tf.nn.relu(conv1 + bias1, name='acv1')
        print_activations(activation1)

        pool1 = tf.nn.max_pool(activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)
    # conv_layer_2
    with tf.name_scope('conv2_lay') as scope:
        Kernel2 = Kernel([3, 3, 64, 128])
        bias2 = biases([128])
        conv2 = tf.nn.conv2d(pool1, Kernel2, strides=[1, 1, 1, 1], padding='SAME', name=scope)
        activation2 = tf.nn.relu(conv2 + bias2, name='acv2')

        pool2 = tf.nn.max_pool(activation2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    with tf.name_scope('conv3_lay') as scope:
        Kernel3 = Kernel([3, 3, 128, 512])
        bias3 = biases([512])
        conv3 = tf.nn.conv2d(pool2, Kernel3, strides=[1, 1, 1, 1], padding='SAME', name=scope)
        activation2 = tf.nn.relu(conv3 + bias3, name='acv2')

        pool3 = tf.nn.max_pool(activation2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')

    print_activations(pool3)
    flatten = tf.reshape(pool2, [-1, 6*6*512])

    with tf.name_scope('FCL1') as scope:
        weights_FCL1 = Kernel([6*6*512, 512])
        bias_FCL1 = biases([512])
        FCL1 = tf.matmul(flatten, weights_FCL1) + bias_FCL1
        act_FCL1 = tf.nn.relu(FCL1)
        print_activations(act_FCL1)

    with tf.name_scope('FCL2') as scope:
        weights_FCL2 = Kernel([512, 512])
        bias_FCL2 = biases([512])
        FCL2 = tf.matmul(act_FCL1, weights_FCL2) + bias_FCL2
        act_FCL2 = tf.nn.relu(FCL2)

    with tf.name_scope('FCL3') as scope:
        weights_FCL3 = Kernel([512, 7])
        bias_FCL3 = biases([7])
        FCL3 = tf.matmul(act_FCL2, weights_FCL3) + bias_FCL3

    prediction = tf.nn.sigmoid(FCL3)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=FCL3))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    prediction_2 = tf.nn.softmax(prediction)
    correct_prediction = tf.equal(tf.argmax(prediction_2, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.90
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    for epoch in range(40):
        for batch in range(train_batch-1):
            sess.run(train_step, feed_dict={X:X_train[batch], Y:Y_train[batch]})
        acc = sess.run(accuracy, feed_dict={X:X_test, Y:Y_test})
        print(print("Iter: " + str(epoch) + ", acc: " + str(acc)))
     #   print(print("Iter: " + str(epoch) + ", loss: " + str(cross_entropy)))
