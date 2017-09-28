#!/usr/bin/env python3


'''
linear model fit
'''

__author__ = 'smdsbz'


import pandas as pd
import numpy as np
import tensorflow as tf
from random import randint

# from my_utils import *
from my_utils.read_team_data import *
from my_utils.read_match_data import *


TEAM_DATA_PATH = './teamData.csv'
MATCH_RESULT_PATH = './matchDataTrain.csv'
TEST_DATA_PATH = './matchDataTest_Self.csv'


######## get data from .csv files ########

team_features = read_team_data(path=TEAM_DATA_PATH)

match_results = read_match_data(path=MATCH_RESULT_PATH)

test_set = read_match_data(path=TEST_DATA_PATH)
# print(len(test_set))

######## build input vector x and y ########

def my_input_x(match_result):
    '''
    read 1 record of match_result,
    gives diffs of team_features
    '''
    guest_feature = team_features[match_result[0]]
    host_feature = team_features[match_result[1]]
    feature_diff = []
    for x, y in zip(guest_feature, host_feature):
        feature_diff.append(x - y)
    return feature_diff


def my_input_y(match_result):
    return match_result[2]


# x_train = [
#         lambda x: my_input_x(x) \
#         for x in match_results
# ]
#
# y_train = [
#         lambda y: my_input_y(y) \
#         for y in match_results
# ]


def next_batch(size=100):
    batch_xs = []
    batch_ys = []
    rand_range = len(match_results)
    for _ in range(size):
        idx = randint(0, rand_range-1)
        batch_xs.append(
            my_input_x(match_results[idx])
        )
        batch_ys.append(
            my_input_y(match_results[idx])
        )
    return (
            np.array(batch_xs, dtype=np.float32),
            np.array(batch_ys, dtype=np.float32)
    )


def test_batch():
    batch_xs = []
    batch_ys = []
    for idx in range(len(test_set)):
        batch_xs.append(
                my_input_x(test_set[idx])
        )
        batch_ys.append(
                my_input_y(test_set[idx])
        )
    # for x in batch_xs: print(x[-1], end=',\n')  # NOTE: check 1-fit here!
    return (
            np.array(batch_xs, dtype=np.float32),
            np.array(batch_ys, dtype=np.float32)
    )





if __name__ == '__main__':

    ######## build computable graph ########

    # x     - diff between two team features
    # y     - predicted value
    # y_    - true value (diff of match_results)
    # W, b  - param to be trained
    x = tf.placeholder(tf.float32, shape=[None, len(team_features['0'])])
    W = tf.Variable(tf.zeros([len(team_features['0']), 2], tf.float32))
    # W = tf.Print(W, [W], message='W: ')
    b = tf.Variable(tf.zeros([2], tf.float32))

    # NOTE: linear model definition
    y = tf.matmul(x, W) + b
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    y_ = tf.placeholder(tf.float32, [None, 2])

    # print(x.dtype, W.dtype, b.dtype, y.dtype, y_.dtype)

    loss = tf.reduce_sum(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(3e-7).minimize(loss)


    ######## RUN ########

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for step in range(20000):
        batch_xs, batch_ys = next_batch(size=100)
        _, loss_val = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
        if step % 100 == 0:
            print('loss = %.3f, at step %d' % (loss_val, step))


    ######## evaluate ########

    # zero_valve = tf.constant(0, dtype=tf.float32)

    # TODO: TensorFlow Comparison Operators
    correct_prediction = tf.equal(
            tf.argmax(y, 1),
            tf.argmax(y_, 1)
    )

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_xs, test_ys = test_batch()

    accuracy, W_list = sess.run([accuracy, W], feed_dict={x: test_xs, y_: test_ys})

    print('final accuracy = %.4f' % (accuracy))
    print('W is...\n%s' % (W_list))
