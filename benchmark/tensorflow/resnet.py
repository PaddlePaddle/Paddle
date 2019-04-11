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
"""
based on https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

Get help: python resnet.py --help
See performance on flowers: python resnet.py
Train on cifar10: python resnet.py --data=cifar10 --with_test
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import tensorflow as tf

DTYPE = tf.float32


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet'],
        default='resnet',
        help='The model architecture.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--use_fake_data',
        action='store_true',
        help='use real data or fake data')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=5,
        help='The first num of minibatch num to skip, for better performance test'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=105,
        help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=300, help='The number of passes.')
    parser.add_argument(
        '--order',
        type=str,
        default='NHWC',
        choices=['NCHW', 'NHWC'],
        help='The data order, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--data',
        type=str,
        default='flowers102',
        choices=['flowers102', 'cifar10'],
        help='The kinds of data.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--with_test',
        action='store_true',
        help='If set, test the testset during training.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    vars(args)['iterations'] = vars(args)['pass_num'] * 1000 if vars(args)[
        'with_test'] else vars(args)['iterations']
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    # This is consistent with PaddlePaddle.
    # In addition, the calculation for output size in TensorFlow can refer: 
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def conv_bn(inputs,
            filters,
            kernel_size,
            strides,
            is_training,
            data_format,
            act=True):
    # def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    # set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        data_format=data_format)
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == 'channels_first' else 3,
        momentum=0.9,
        epsilon=1e-05,
        center=True,
        scale=True,
        training=is_training,
        fused=True)
    if act:
        inputs = tf.nn.relu(inputs)
    return inputs


def basicblock(inputs, filters, is_training, projection_shortcut, strides,
               data_format):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    inputs = conv_bn(inputs, filters, 3, strides, is_training, data_format)
    inputs = conv_bn(inputs, filters, 3, 1, is_training, data_format, act=False)
    inputs = inputs + shortcut
    inputs = tf.nn.relu(inputs)
    return inputs


def bottleneck(inputs, filters, is_training, projection_shortcut, strides,
               data_format):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    inputs = conv_bn(inputs, filters, 1, strides, is_training, data_format)
    inputs = conv_bn(inputs, filters, 3, 1, is_training, data_format, act=False)
    inputs = conv_bn(
        inputs, filters * 4, 1, 1, is_training, data_format, act=False)
    inputs = inputs + shortcut
    inputs = tf.nn.relu(inputs)
    return inputs


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs,
            filters=filters_out,
            kernel_size=1,
            strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut,
                      strides, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

    return tf.identity(inputs, name)


def resnet_imagenet(depth, class_dim, data_format):
    """Returns the ResNet model for a given size and number of output classes."""

    def resnet_generator(block_fn,
                         layers,
                         num_classes,
                         data_format='channels_last'):
        if data_format is None:
            data_format = ('channels_first'
                           if tf.test.is_built_with_cuda() else 'channels_last')

        def model(inputs, is_training):
            """Constructs the ResNet model given the inputs."""
            if data_format == 'channels_first':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = conv_bn(inputs, 64, 7, 2, is_training, data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            inputs = tf.layers.max_pooling2d(
                inputs=inputs,
                pool_size=3,
                strides=2,
                padding='SAME',
                data_format=data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')
            inputs = block_layer(inputs, 64, block_fn, layers[0], 1,
                                 is_training, 'block_layer1', data_format)
            inputs = block_layer(inputs, 128, block_fn, layers[1], 2,
                                 is_training, 'block_layer2', data_format)
            inputs = block_layer(inputs, 256, block_fn, layers[2], 2,
                                 is_training, 'block_layer3', data_format)
            inputs = block_layer(inputs, 512, block_fn, layers[3], 2,
                                 is_training, 'block_layer4', data_format)
            inputs = tf.layers.average_pooling2d(
                inputs=inputs,
                pool_size=7,
                strides=1,
                padding='VALID',
                data_format=data_format)
            inputs = tf.identity(inputs, 'final_avg_pool')
            inputs = tf.reshape(inputs,
                                [-1, 512 if block_fn is basicblock else 2048])
            inputs = tf.layers.dense(inputs=inputs, units=num_classes)
            inputs = tf.identity(inputs, 'final_dense')
            return inputs

        return model

    model_params = {
        18: {
            'block': basicblock,
            'layers': [2, 2, 2, 2]
        },
        34: {
            'block': basicblock,
            'layers': [3, 4, 6, 3]
        },
        50: {
            'block': bottleneck,
            'layers': [3, 4, 6, 3]
        },
        101: {
            'block': bottleneck,
            'layers': [3, 4, 23, 3]
        },
        152: {
            'block': bottleneck,
            'layers': [3, 8, 36, 3]
        },
        200: {
            'block': bottleneck,
            'layers': [3, 24, 36, 3]
        }
    }
    if depth not in model_params:
        raise ValueError('Not a valid depth:', depth)
    params = model_params[depth]
    return resnet_generator(params['block'], params['layers'], class_dim,
                            data_format)


def resnet_cifar10(depth, num_classes, data_format):
    if depth % 6 != 2:
        raise ValueError('depth must be 6n + 2:', depth)

    num_blocks = (depth - 2) // 6

    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        inputs = conv_bn(inputs, 16, 3, 1, is_training, data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = block_layer(inputs, 16, basicblock, num_blocks, 1, is_training,
                             'block_layer1', data_format)
        inputs = block_layer(inputs, 32, basicblock, num_blocks, 2, is_training,
                             'block_layer2', data_format)
        inputs = block_layer(inputs, 64, basicblock, num_blocks, 2, is_training,
                             'block_layer3', data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs,
            pool_size=8,
            strides=1,
            padding='VALID',
            data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs, [-1, 64])
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs

    return model


def run_benchmark(args, data_format='channels_last', device='/cpu:0'):
    """Our model_fn for ResNet to be used with our Estimator."""

    class_dim = 1000
    dshape = (None, 224, 224, 3)

    pdshape = (3, 224, 224)
    if args.data == 'flowers102':
        class_dim = 102
        dshape = (None, 224, 224, 3)
        pdshape = (3, 224, 224)
    elif args.data == 'cifar10':
        class_dim = 10
        dshape = (None, 32, 32, 3)
        pdshape = (3, 32, 32)

    with tf.device(device):
        images = tf.placeholder(DTYPE, shape=dshape)
        labels = tf.placeholder(tf.int64, shape=(None, ))
        is_training = tf.placeholder('bool')
        onehot_labels = tf.one_hot(labels, depth=class_dim)

        network = resnet_cifar10(
            32, class_dim,
            data_format) if args.data == 'cifar10' else resnet_imagenet(
                50, class_dim, data_format)

        logits = network(inputs=images, is_training=is_training)

        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=onehot_labels)
        avg_cost = tf.reduce_mean(cross_entropy)

        correct = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        lr = 0.1 if args.data == 'cifar10' else 0.01
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10()
            if args.data == 'cifar10' else paddle.dataset.flowers.train(),
            buf_size=5120),
        batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if args.data == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=100)

    def test():
        test_accs = []
        for batch_id, data in enumerate(test_reader()):
            test_images = np.array(
                map(lambda x: np.transpose(x[0].reshape(pdshape),
                axes=[1, 2, 0]), data)).astype("float32")
            test_labels = np.array(map(lambda x: x[1], data)).astype('int64')
            test_accs.append(
                accuracy.eval(feed_dict={
                    images: test_images,
                    labels: test_labels,
                    is_training: False
                }))
        print("Pass = %d, Train performance = %f imgs/s, Test accuracy = %f\n" %
              (pass_id, num_samples / train_elapsed, np.mean(test_accs)))

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)

        if args.use_fake_data:
            data = train_reader().next()
            images_data = np.array(
                    map(lambda x: np.transpose(x[0].reshape(pdshape),
                    axes=[1, 2, 0]), data)).astype("float32")
            labels_data = np.array(map(lambda x: x[1], data)).astype('int64')
        iters, num_samples, start_time = 0, 0, 0.0
        for pass_id in range(args.pass_num):
            if iters == args.iterations:
                break
            train_accs = []
            train_losses = []
            for batch_id, data in enumerate(train_reader()):
                if iters == args.skip_batch_num:
                    start_time = time.time()
                    num_samples = 0
                if iters == args.iterations:
                    break
                if not args.use_fake_data:
                    images_data = np.array(
                        map(lambda x: np.transpose(x[0].reshape(pdshape),
                        axes=[1, 2, 0]), data)).astype("float32")
                    labels_data = np.array(map(lambda x: x[1], data)).astype(
                        'int64')
                _, loss, acc = sess.run([train_op, avg_cost, accuracy],
                                        feed_dict={
                                            images: images_data,
                                            labels: labels_data,
                                            is_training: True
                                        })
                iters += 1
                train_accs.append(acc)
                train_losses.append(loss)
                num_samples += len(data)
                print("Pass=%d, Iter=%d, Loss=%f, Accuray=%f\n" %
                      (pass_id, iters, loss, acc))

            train_elapsed = time.time() - start_time
            print("Pass=%d, Loss=%f, Accuray=%f\n" %
                  (pass_id, np.mean(train_losses), np.mean(train_accs)))

            # evaluation
            if args.with_test:
                test()

        if not args.with_test:
            duration = time.time() - start_time
            examples_per_sec = num_samples / duration
            sec_per_batch = duration / (iters - args.skip_batch_num)

            print('Total examples: %d, total time: %.5f' %
                  (num_samples, duration))
            print('%.5f examples/sec, %.5f sec/batch' %
                  (examples_per_sec, sec_per_batch))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if tf.test.is_built_with_cuda():
        device = '/device:GPU:0'
        if args.order == 'NHWC':
            data_format = 'channels_last'
        else:
            data_format = 'channels_first'
    else:
        device = '/cpu:0'
        if args.order == 'NHWC':
            data_format = 'channels_last'
        else:
            raise ValueError('Only support NHWC order in CPU mode')

    run_benchmark(args, data_format, device)
