#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
import math
import time

import tensorflow.python.platform
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('forward_only', False,
                            """Only run the forward pass.""")
tf.app.flags.DEFINE_boolean('forward_backward_only', False,
                            """Only run the forward-forward pass.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

parameters = []

conv_counter = 1
pool_counter = 1
affine_counter = 1


def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType, wd=0.005, act=True):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                [kH, kW, nIn, nOut], dtype=tf.float32, stddev=1e-1),
            name='weights')

        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        if FLAGS.data_format == 'NCHW':
            strides = [1, 1, dH, dW]
        else:
            strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(
            inpOp,
            kernel,
            strides,
            padding=padType,
            data_format=FLAGS.data_format)
        biases = tf.Variable(
            tf.constant(
                0.0, shape=[nOut], dtype=tf.float32),
            trainable=True,
            name='biases')
        bias = tf.reshape(
            tf.nn.bias_add(
                conv, biases, data_format=FLAGS.data_format),
            conv.get_shape())

        conv1 = tf.nn.relu(bias, name=scope) if act else bias

        parameters += [kernel, biases]

        return conv1


def _affine(inpOp, nIn, nOut, wd=None, act=True):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                [nIn, nOut], dtype=tf.float32, stddev=1e-1),
            name='weights')

        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        biases = tf.Variable(
            tf.constant(
                0.0, shape=[nOut], dtype=tf.float32),
            trainable=True,
            name='biases')

        affine1 = tf.nn.relu_layer(
            inpOp, kernel, biases,
            name=name) if act else tf.matmul(inpOp, kernel) + biases

        parameters += [kernel, biases]

        return affine1


def _mpool(inpOp, kH, kW, dH, dW, padding):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    if FLAGS.data_format == 'NCHW':
        ksize = [1, 1, kH, kW]
        strides = [1, 1, dH, dW]
    else:
        ksize = [1, kH, kW, 1]
        strides = [1, dH, dW, 1]
    return tf.nn.max_pool(
        inpOp,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=FLAGS.data_format,
        name=name)


def _apool(inpOp, kH, kW, dH, dW, padding):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    if FLAGS.data_format == 'NCHW':
        ksize = [1, 1, kH, kW]
        strides = [1, 1, dH, dW]
    else:
        ksize = [1, kH, kW, 1]
        strides = [1, dH, dW, 1]
    return tf.nn.avg_pool(
        inpOp,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=FLAGS.data_format,
        name=name)


def _norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input,
                     lsize,
                     bias=1.0,
                     alpha=0.001 / 9.0,
                     beta=0.75,
                     name=name)


def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated,
                                       tf.pack([batch_size, 10]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def inference(images):
    conv1 = _conv(images, 3, 32, 5, 5, 1, 1, 'SAME')
    pool1 = _mpool(conv1, 3, 3, 2, 2, 'SAME')
    conv2 = _conv(pool1, 32, 32, 5, 5, 1, 1, 'SAME')
    pool2 = _apool(conv2, 3, 3, 2, 2, 'SAME')
    conv3 = _conv(pool2, 32, 64, 5, 5, 1, 1, 'SAME')
    pool3 = _apool(conv3, 3, 3, 2, 2, 'SAME')
    resh1 = tf.reshape(pool3, [-1, 64 * 4 * 4])
    affn1 = _affine(resh1, 64 * 4 * 4, 64)
    affn2 = _affine(affn1, 64, 10, act=False)

    print('conv1:', get_incoming_shape(conv1))
    print('pool1:', get_incoming_shape(pool1))
    print('conv2:', get_incoming_shape(conv2))
    print('pool2:', get_incoming_shape(pool2))
    print('conv3:', get_incoming_shape(conv3))
    print('pool3:', get_incoming_shape(pool3))

    return affn2


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    if not isinstance(target, list):
        target = [target]
    target_op = tf.group(*target)
    for i in xrange(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target_op)
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
    global parameters
    with tf.Graph().as_default():
        # Generate some dummy images.
        image_size = 32
        # Note that our padding definition is slightly different the cuda-convnet.
        # In order to force the model to start with the same activations sizes,
        # we add 3 to the image_size and employ VALID padding above.
        if FLAGS.data_format == 'NCHW':
            image_shape = [FLAGS.batch_size, 3, image_size, image_size]
        else:
            image_shape = [FLAGS.batch_size, image_size, image_size, 3]

        images = tf.get_variable(
            'image',
            image_shape,
            initializer=tf.truncated_normal_initializer(
                stddev=0.1, dtype=tf.float32),
            dtype=tf.float32,
            trainable=False)

        labels = tf.get_variable(
            'label', [FLAGS.batch_size],
            initializer=tf.constant_initializer(1),
            dtype=tf.int32,
            trainable=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        last_layer = inference(images)

        objective = loss(last_layer, labels)

        # Compute gradients.
        opt = tf.train.MomentumOptimizer(0.001, 0.9)
        grads = opt.compute_gradients(objective)
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(
                0.0, dtype=tf.float32),
            trainable=False,
            dtype=tf.float32)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables(
        ))

        # Build an initialization operation.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
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
            # Run the forward benchmark.
            time_tensorflow_run(sess, last_layer, "Forward")

        if run_forward_backward:
            with tf.control_dependencies(
                [apply_gradient_op, variables_averages_op]):
                train_op = tf.no_op(name='train')
            time_tensorflow_run(sess, [train_op, objective], "Forward-backward")


def main(_):
    run_benchmark()


if __name__ == '__main__':
    tf.app.run()
