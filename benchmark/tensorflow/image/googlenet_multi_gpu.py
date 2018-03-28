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
import re
import time

import tensorflow.python.platform
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64, """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)

tf.app.flags.DEFINE_string('train_dir', '/train_model',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_gpus', 4, """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 50
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
TOWER_NAME = 'tower'


def _conv(name, inpOp, nIn, nOut, kH, kW, dH, dW, padType, wd=0.005):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            name + '_w', [kH, kW, nIn, nOut],
            initializer=tf.truncated_normal_initializer(
                stddev=0.01, dtype=tf.float32),
            dtype=tf.float32)

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


def _affine(name, inpOp, nIn, nOut, wd=0.005, act=True):
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            name + '_w', [nIn, nOut],
            initializer=tf.truncated_normal_initializer(
                stddev=0.01, dtype=tf.float32),
            dtype=tf.float32)

        if wd is not None:
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

        return affine1


def _mpool(name, inpOp, kH, kW, dH, dW, padding):
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


def _apool(name, inpOp, kH, kW, dH, dW, padding):
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


def _inception(name, inp, inSize, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2):
    conv1 = _conv(name + '_1', inp, inSize, o1s, 1, 1, 1, 1, 'VALID')

    conv3_ = _conv(name + '_3r', inp, inSize, o2s1, 1, 1, 1, 1, 'VALID')
    conv3 = _conv(name + '_3', conv3_, o2s1, o2s2, 3, 3, 1, 1, 'SAME')

    conv5_ = _conv(name + '_5r', inp, inSize, o3s1, 1, 1, 1, 1, 'VALID')
    conv5 = _conv(name + '5', conv5_, o3s1, o3s2, 5, 5, 1, 1, 'SAME')

    pool_ = _mpool(name + 'pool', inp, o4s1, o4s1, 1, 1, 'SAME')
    pool = _conv(name + 'proj', pool_, inSize, o4s2, 1, 1, 1, 1, 'VALID')

    if FLAGS.data_format == 'NCHW':
        channel_dim = 1
    else:
        channel_dim = 3
    incept = tf.concat(channel_dim, [conv1, conv3, conv5, pool])
    return incept


def inference(images):
    # stage 1
    conv1 = _conv('conv1', images, 3, 64, 7, 7, 2, 2, 'SAME')
    pool1 = _mpool('pool1', conv1, 3, 3, 2, 2, 'SAME')

    # stage 2
    conv2 = _conv('conv2', pool1, 64, 64, 1, 1, 1, 1, 'VALID')
    conv3 = _conv('conv3', conv2, 64, 192, 3, 3, 1, 1, 'SAME')
    pool3 = _mpool('pool3', conv3, 3, 3, 2, 2, 'SAME')

    # stage 3
    incept3a = _inception('ince3a', pool3, 192, 64, 96, 128, 16, 32, 3, 32)
    incept3b = _inception('ince3b', incept3a, 256, 128, 128, 192, 32, 96, 3, 64)
    pool4 = _mpool('pool4', incept3b, 3, 3, 2, 2, 'SAME')

    # stage 4
    incept4a = _inception('ince4a', pool4, 480, 192, 96, 208, 16, 48, 3, 64)
    incept4b = _inception('ince4b', incept4a, 512, 160, 112, 224, 24, 64, 3, 64)
    incept4c = _inception('ince4c', incept4b, 512, 128, 128, 256, 24, 64, 3, 64)
    incept4d = _inception('ince4d', incept4c, 512, 112, 144, 288, 32, 64, 3, 64)
    incept4e = _inception('ince4e', incept4d, 528, 256, 160, 320, 32, 128, 3,
                          128)
    pool5 = _mpool('pool5', incept4e, 3, 3, 2, 2, 'SAME')

    # stage 5
    incept5a = _inception('ince5a', pool5, 832, 256, 160, 320, 32, 128, 3, 128)
    incept5b = _inception('ince5b', incept5a, 832, 384, 192, 384, 48, 128, 3,
                          128)
    pool6 = _apool('pool6', incept5b, 7, 7, 1, 1, 'VALID')

    # output 1
    resh1 = tf.reshape(pool6, [-1, 1024])
    drop = tf.nn.dropout(resh1, 0.4)
    affn1 = _affine('fc_out', resh1, 1024, 1000, act=False)

    return affn1


def tower_loss(scope):
    """Calculate the total loss on a single tower running the model.
    Args:
        scope: unique prefix string identifying the tower, e.g. 'tower_0'
    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """
    image_size = 224
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

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(last_layer, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(loss_name + ' (raw)', l)
        tf.scalar_summary(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def time_tensorflow_run(session, target):
    num_steps_burn_in = 50
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in xrange(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        _, loss_value = session.run(target)
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration

                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch batch_size = %d)')
                print(format_str %
                      (datetime.now(), i - num_steps_burn_in, loss_value,
                       duration, sec_per_batch, num_examples_per_step))

            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: FwdBwd across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), FLAGS.num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            decay_steps,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.MomentumOptimizer(lr, 0.9)

        # Calculate the gradients for each model tower.
        tower_grads = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    # Calculate the loss for one tower of the model. This function
                    # constructs the entire model but shares the variables across
                    # all towers.
                    loss = tower_loss(scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op)

        # Build an initialization operation.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        time_tensorflow_run(sess, [train_op, loss])


def main(_):
    run_benchmark()


if __name__ == '__main__':
    tf.app.run()
