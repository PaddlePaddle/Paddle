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


def _conv(name, inpOp, nIn, nOut, kH, kW, dH, dW, padType, wd=0.0005):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            name + '_w', [kH, kW, nIn, nOut],
            initializer=tf.truncated_normal_initializer(
                stddev=0.01, dtype=tf.float32),
            dtype=tf.float32)

        if wd is not None and wd > 0:
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

        biases = tf.get_variable(
            name=name + '_b',
            shape=[nOut],
            initializer=tf.constant_initializer(
                value=0.0, dtype=tf.float32),
            dtype=tf.float32)

        bias = tf.reshape(
            tf.nn.bias_add(
                conv, biases, data_format=FLAGS.data_format),
            conv.get_shape())

        conv1 = tf.nn.relu(bias, name=scope)
        return conv1


def _affine(name, inpOp, nIn, nOut, wd=0.0005, act=True, drop=None):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            name + '_w', [nIn, nOut],
            initializer=tf.truncated_normal_initializer(
                stddev=0.01, dtype=tf.float32),
            dtype=tf.float32)

        if wd is not None and wd > 0:
            weight_decay = tf.mul(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        biases = tf.get_variable(
            name + '_b', [nOut],
            initializer=tf.constant_initializer(
                value=0.0, dtype=tf.float32),
            dtype=tf.float32,
            trainable=True)

        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name) if act else \
                  tf.matmul(inpOp, kernel) + biases

        output = tf.nn.dropout(affine1, drop) if drop else affine1

        return output


def _mpool(name, inpOp, kH, kW, dH, dW):
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
        padding='VALID',
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
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def inference(images):
    conv1 = _conv('conv1', images, 3, 96, 11, 11, 4, 4, 'VALID')
    pool1 = _mpool('pool1', conv1, 3, 3, 2, 2)
    norm1 = _norm('norm1', pool1, lsize=5)
    conv2 = _conv('conv2', norm1, 96, 256, 5, 5, 1, 1, 'SAME')
    pool2 = _mpool('pool2', conv2, 3, 3, 2, 2)
    norm2 = _norm('norm2', pool2, lsize=5)
    conv3 = _conv('conv3', norm2, 256, 384, 3, 3, 1, 1, 'SAME')
    conv4 = _conv('conv4', conv3, 384, 384, 3, 3, 1, 1, 'SAME')
    conv5 = _conv('conv5', conv4, 384, 256, 3, 3, 1, 1, 'SAME')
    pool5 = _mpool('pool5', conv5, 3, 3, 2, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
    affn1 = _affine('fc6', resh1, 256 * 6 * 6, 4096, 0.5)
    affn2 = _affine('fc7', affn1, 4096, 4096, 0.5)
    affn3 = _affine('fc8', affn2, 4096, 1000, wd=None, act=False)  # last fc

    return affn3


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


def _add_loss_summaries(total_loss):
    """
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def run_benchmark():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            # Generate some dummy images.
            image_size = 224
            # Note that our padding definition is slightly different the cuda-convnet.
            # In order to force the model to start with the same activations sizes,
            # we add 3 to the image_size and employ VALID padding above.
            if FLAGS.data_format == 'NCHW':
                image_shape = [
                    FLAGS.batch_size, 3, image_size + 3, image_size + 3
                ]
            else:
                image_shape = [
                    FLAGS.batch_size, image_size + 3, image_size + 3, 3
                ]
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
            # Compute the gradient with respect to all the parameters.

            # Compute gradients.
            # opt = tf.train.GradientDescentOptimizer(0.001)
            opt = tf.train.MomentumOptimizer(0.001, 0.9)
            grads = opt.compute_gradients(objective)
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32),
                trainable=False,
                dtype=tf.float32)
            apply_gradient_op = opt.apply_gradients(
                grads, global_step=global_step)

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(0.9,
                                                                  global_step)
            variables_averages_op = variable_averages.apply(
                tf.trainable_variables())

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
                time_tensorflow_run(sess, last_layer, "Forward")

            if run_forward_backward:
                with tf.control_dependencies(
                    [apply_gradient_op, variables_averages_op]):
                    train_op = tf.no_op(name='train')
                time_tensorflow_run(sess, [train_op, objective],
                                    "Forward-backward")


def main(_):
    run_benchmark()


if __name__ == '__main__':
    tf.app.run()
