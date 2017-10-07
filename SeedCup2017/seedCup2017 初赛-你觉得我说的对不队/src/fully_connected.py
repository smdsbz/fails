#!/usr/bin/env python3


'''
fully-connected version
'''

__author__ = 'smdsbz'



import pandas as pd
import tensorflow as tf
import math
from random import randint

from my_utils.read_match_data import *
from my_utils.read_team_data import *
from linear_model import *


######## get data ########

# from import
FEAT_LEN = len(team_features['0'])

max_steps = 150000

BATCH_SIZE = 100

SAMPLE_SIZE = len(match_results)

hidden1_unit = 128

hidden2_unit = 32

learning_rate = 1e-2

logdir = './'

######## build network ########

def inference(feature, hidden1_unit, hidden2_unit):

    with tf.name_scope('hidden1'):
        W = tf.Variable(
                tf.truncated_normal([FEAT_LEN, hidden1_unit],
                                    stddev=1.0 / math.sqrt(float(FEAT_LEN)) ),
                name='Weight'
        )
        b = tf.Variable(tf.zeros([hidden1_unit]),
                        name='bias')
        hidden1 = tf.nn.relu(tf.matmul(feature, W) + b)

    with tf.name_scope('hidden2'):
        W = tf.Variable(
                tf.truncated_normal([hidden1_unit, hidden2_unit],
                                    stddev=1.0 / math.sqrt(float(hidden1_unit)) ),
                name='Weight'
        )
        b = tf.Variable(tf.zeros([hidden2_unit]),
                        name='bias')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, W) + b)

    with tf.name_scope('softmax'):
        W = tf.Variable(
                tf.truncated_normal([hidden2_unit, 2],
                                    stddev=1.0 / math.sqrt(float(hidden2_unit)) ),
                name='Weight'
        )
        b = tf.Variable(tf.zeros([2]),
                        name='bias')
        logits = tf.matmul(hidden2, W) + b

    return logits



def loss(logits, truth):

    truth = tf.to_int64(truth)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=truth, logits=logits, name='xentropy'
    )
    return tf.reduce_mean(xentropy, name='xentropy_mean')



def training(loss, learning_rate):

    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op



def evaluation(logits, truth):

    bingo = tf.nn.in_top_k(logits, truth, 1)

    return tf.reduce_sum(tf.cast(bingo, tf.int32))




######## training net details ########

def placeholder_inputs(batch_size=BATCH_SIZE):

    feature_placeholder = tf.placeholder(tf.float32,
                                         shape=[batch_size, FEAT_LEN])
    predict_placeholder = tf.placeholder(tf.int32,
                                         shape=[batch_size])
    return feature_placeholder, predict_placeholder



def fill_feed_dict(feat_hold, pred_hold, size=BATCH_SIZE, eval_flag=False):

    def next_batch(size=size):
        batch_xs = []
        batch_ys = []
        rand_range = len(match_results)
        for _ in range(size):
            idx = randint(0, rand_range-1)
            batch_xs.append(
                my_input_x(match_results[idx])
            )
            batch_ys.append(
                my_input_y(match_results[idx]).index(1)
            )
        return (
                np.array(batch_xs, dtype=np.float32),
                np.array(batch_ys, dtype=np.float32)
        )

    def test_batch():
        batch_xs = []
        batch_ys = []
        for _ in range(100):
            idx = randint(0, 99)
            batch_xs.append(
                    my_input_x(test_set[idx])
            )
            batch_ys.append(
                    my_input_y(test_set[idx]).index(1)
            )
        return (
                np.array(batch_xs, dtype=np.float32),
                np.array(batch_ys, dtype=np.float32)
        )


    feat_batch, pred_batch = next_batch(size=size) if not eval_flag else test_batch()

    feed_dict = {
            feat_hold: feat_batch,
            pred_hold: pred_batch
    }
    return feed_dict



def do_eval(sess, eval_correct,
            feat_hold, pred_hold,
            eval_flag=False):

    true_cnt = 0
    # 301 - test set size
    steps_per_epoch = (SAMPLE_SIZE // BATCH_SIZE) if not eval_flag else (301 // BATCH_SIZE)
    sample_total = steps_per_epoch * BATCH_SIZE
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(feat_hold, pred_hold, eval_flag=eval_flag)
        true_cnt += sess.run(eval_correct, feed_dict=feed_dict)
    accuracy = float(true_cnt) / sample_total
    print('got %d out of %d, accuracy %.3f' %
          (true_cnt, sample_total, accuracy) )



def run_training():

    with tf.Graph().as_default():

        feat_hold, pred_hold = placeholder_inputs(BATCH_SIZE)

        logits = inference(feat_hold, hidden1_unit, hidden2_unit)

        loss_loc = loss(logits, pred_hold)

        train_op = training(loss_loc, learning_rate)

        eval_correct = evaluation(logits, pred_hold)

        # summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        # summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        sess.run(init)

        for step in range(max_steps):

            feed_dict = fill_feed_dict(feat_hold, pred_hold)

            _, loss_val = sess.run([train_op, loss_loc],
                                   feed_dict=feed_dict)

            if step % 100 == 0:

                print('loss = %.3f at step %d' % (loss_val, step))

                # summary_str = sess.run(summary, feed_dict=feed_dict)
                # summary_writer.add_summary(summary_str, step)
                # summary_writer.flush()

            if (step + 1) == max_steps:

                final_file = './trained_save/model.ckpt'
                saver.save(sess, final_file)

                print('Training data eval:')
                do_eval(sess, eval_correct, feat_hold, pred_hold)

                print('Test data eval:')
                do_eval(sess, eval_correct, feat_hold, pred_hold,
                        eval_flag=True)



        def moment_of_truth(path='./matchDataTest.csv'):

            question_mark = pd.read_csv(path)
            output = []

            # test_x, test_y = placeholder_inputs(batch_size=1)
            #
            # logits = inference(test_x, hidden1_unit, hidden2_unit)

            for line in question_mark.iterrows():
                guest_team = line[1]['客场队名']
                host_team = line[1]['主场队名']

                feed_dict = {
                    feat_hold: [my_input_x([str(guest_team), str(host_team)]) for _ in range(BATCH_SIZE)],
                    pred_hold: [0 for _ in range(BATCH_SIZE)]
                }
                # print(feed_dict)

                pred_val = sess.run(logits, feed_dict=feed_dict)[0]
                # print(pred_val)
                output.append(1 if pred_val[0] == max(pred_val) else 0)

            pd.Series(output).to_csv(
                    'predictPro.csv',
                    header=['主场赢得比赛的置信度'],
                    index=False
            )

            return True


        moment_of_truth()



if __name__ == '__main__':
    run_training()
