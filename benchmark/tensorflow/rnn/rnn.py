#!/usr/bin/env python
from six.moves import xrange  # pylint: disable=redefined-builtin
import math
import time
import numpy as np
from datetime import datetime

import reader
import tensorflow as tf
from tensorflow.python.ops import rnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_layers', 1, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_len', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('forward_only', False,
                            """Only run the forward pass.""")
tf.app.flags.DEFINE_boolean('forward_backward_only', False,
                            """Only run the forward-forward pass.""")
tf.app.flags.DEFINE_integer('hidden_size', 128, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('emb_size', 128, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

VOCAB_SIZE = 30000
NUM_CLASS = 2


def get_feed_dict(x_data, y_data=None):
    feed_dict = {}

    if y_data is not None:
        feed_dict[y_input] = y_data

    for i in xrange(x_data.shape[0]):
        feed_dict[x_input[i]] = x_data[i, :, :]

    return feed_dict


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


# Note input * W is done in LSTMCell, 
# which is different from PaddlePaddle
def single_lstm(name,
                incoming,
                n_units,
                use_peepholes=True,
                return_seq=False,
                return_state=False):
    with tf.name_scope(name) as scope:
        cell = tf.nn.rnn_cell.LSTMCell(n_units, use_peepholes=use_peepholes)
        output, _cell_state = rnn.rnn(cell, incoming, dtype=tf.float32)
        out = output if return_seq else output[-1]
        return (out, _cell_state) if return_state else out


def lstm(name,
         incoming,
         n_units,
         use_peepholes=True,
         return_seq=False,
         return_state=False,
         num_layers=1):
    with tf.name_scope(name) as scope:
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            n_units, use_peepholes=use_peepholes)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
        initial_state = cell.zero_state(FLAGS.batch_size, dtype=tf.float32)
        if not isinstance(incoming, list):
            # if the input is embeding, the Tensor shape : [None, time_step, emb_size]
            incoming = [
                tf.squeeze(input_, [1])
                for input_ in tf.split(1, FLAGS.max_len, incoming)
            ]
        outputs, state = tf.nn.rnn(cell,
                                   incoming,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
        out = outputs if return_seq else outputs[-1]
        return (out, _cell_state) if return_state else out


def embedding(name, incoming, vocab_size, emb_size):
    with tf.name_scope(name) as scope:
        #with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            name + '_emb', [vocab_size, emb_size], dtype=tf.float32)
        out = tf.nn.embedding_lookup(embedding, incoming)
        return out


def fc(name, inpOp, nIn, nOut, act=True):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            name + '_w', [nIn, nOut],
            initializer=tf.truncated_normal_initializer(
                stddev=0.01, dtype=tf.float32),
            dtype=tf.float32)

        biases = tf.get_variable(
            name + '_b', [nOut],
            initializer=tf.constant_initializer(
                value=0.0, dtype=tf.float32),
            dtype=tf.float32,
            trainable=True)

        net = tf.nn.relu_layer(inpOp, kernel, biases, name=name) if act else \
                  tf.matmul(inpOp, kernel) + biases

        return net


def inference(seq):
    net = embedding('emb', seq, VOCAB_SIZE, FLAGS.emb_size)
    print "emb:", get_incoming_shape(net)
    net = lstm('lstm', net, FLAGS.hidden_size, num_layers=FLAGS.num_layers)
    print "lstm:", get_incoming_shape(net)
    net = fc('fc1', net, FLAGS.hidden_size, 2)
    return net


def loss(logits, labels):
    # one label index for one sample
    labels = tf.cast(labels, tf.float32)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def time_tensorflow_run(session, target, x_input, y_input, info_string):
    num_steps_burn_in = 50
    total_duration = 0.0
    total_duration_squared = 0.0
    if not isinstance(target, list):
        target = [target]
    target_op = tf.group(*target)
    train_dataset = reader.create_datasets("imdb.pkl", VOCAB_SIZE)
    for i in xrange(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        data, label = train_dataset.next_batch(FLAGS.batch_size)
        _ = session.run(target_op, feed_dict={x_input: data, y_input: label})
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        global_step = 0
        with tf.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False)
        with tf.device('/gpu:0'):
            #x_input = tf.placeholder(tf.int32, [None, FLAGS.max_len], name="x_input")
            #y_input = tf.placeholder(tf.int32, [None, NUM_CLASS], name="y_input")
            x_input = tf.placeholder(
                tf.int32, [FLAGS.batch_size, FLAGS.max_len], name="x_input")
            y_input = tf.placeholder(
                tf.int32, [FLAGS.batch_size, NUM_CLASS], name="y_input")
            # Generate some dummy sequnce.

            last_layer = inference(x_input)

            objective = loss(last_layer, y_input)
            opt = tf.train.AdamOptimizer(0.001)
            grads = opt.compute_gradients(objective)
            apply_gradient_op = opt.apply_gradients(
                grads, global_step=global_step)

            init = tf.initialize_all_variables()
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
            sess.run(init)

            run_forward = True
            run_forward_backward = True
            if FLAGS.forward_only and FLAGS.forward_backward_only:
                raise ValueError("Cannot specify --forward_only and "
                                 "--forward_backward_only at the same time.")
            if FLAGS.forward_only:
                run_forward_backward = False
            elif FLAGS.forward_backward_only:
                run_forward = False

            if run_forward:
                time_tensorflow_run(sess, last_layer, x_input, y_input,
                                    "Forward")

            if run_forward_backward:
                with tf.control_dependencies([apply_gradient_op]):
                    train_op = tf.no_op(name='train')
                time_tensorflow_run(sess, [train_op, objective], x_input,
                                    y_input, "Forward-backward")


def main(_):
    run_benchmark()


if __name__ == '__main__':
    tf.app.run()
