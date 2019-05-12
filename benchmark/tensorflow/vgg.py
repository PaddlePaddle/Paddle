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
"""VGG16 benchmark in TensorFlow"""
import tensorflow as tf
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--batch_size', type=int, default=128, help="Batch size for training.")
parser.add_argument(
    '--skip_batch_num',
    type=int,
    default=5,
    help='The first num of minibatch num to skip, for better performance test')
parser.add_argument(
    '--iterations', type=int, default=80, help='The number of minibatches.')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help="Learning rate for training.")
parser.add_argument('--num_passes', type=int, default=50, help="No. of passes.")
parser.add_argument(
    '--device',
    type=str,
    default='GPU',
    choices=['CPU', 'GPU'],
    help="The device type.")
parser.add_argument(
    '--data_format',
    type=str,
    default='NHWC',
    choices=['NCHW', 'NHWC'],
    help='The data order, NCHW=[batch, channels, height, width].'
    'Only support NHWC right now.')
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    choices=['cifar10', 'flowers'],
    help='Optional dataset for benchmark.')
args = parser.parse_args()


class VGG16Model(object):
    def __init__(self):
        self.parameters = []

    def batch_norm_relu(self, inputs, is_training):
        """Performs a batch normalization followed by a ReLU."""
        # We set fused=True for a significant speed boost. See
        # https://www.tensorflow.org/speed/speed_guide#common_fused_ops
        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=1 if args.data_format == 'NCHW' else -1,
            momentum=0.9,
            epsilon=1e-05,
            center=True,
            scale=True,
            training=is_training,
            fused=True)
        inputs = tf.nn.relu(inputs)
        return inputs

    def conv_bn_layer(self,
                      name,
                      images,
                      kernel_shape,
                      is_training,
                      drop_rate=0.0):
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    kernel_shape, dtype=tf.float32, stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(
                images,
                kernel, [1, 1, 1, 1],
                data_format=args.data_format,
                padding='SAME')
            biases = tf.Variable(
                tf.constant(
                    0.0, shape=[kernel_shape[-1]], dtype=tf.float32),
                trainable=True,
                name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = self.batch_norm_relu(out, is_training)
            out = tf.layers.dropout(out, rate=drop_rate, training=is_training)
            return out

    def fc_layer(self, name, inputs, shape):
        with tf.name_scope(name) as scope:
            fc_w = tf.Variable(
                tf.truncated_normal(
                    shape, dtype=tf.float32, stddev=1e-1),
                name='weights')
            fc_b = tf.Variable(
                tf.constant(
                    0.0, shape=[shape[-1]], dtype=tf.float32),
                trainable=True,
                name='biases')
            out = tf.nn.bias_add(tf.matmul(inputs, fc_w), fc_b)
            return out

    def network(self, images, class_dim, is_training):
        """ VGG16 model structure.

            TODO(kuke): enable this network to support the 'NCHW' data format
        """

        # conv1
        conv1_1 = self.conv_bn_layer(
            'conv1_1', images, [3, 3, 3, 64], is_training, drop_rate=0.3)
        conv1_2 = self.conv_bn_layer(
            'conv1_2', conv1_1, [3, 3, 64, 64], is_training, drop_rate=0.0)
        # pool1
        pool1 = tf.nn.max_pool(
            conv1_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool1')
        # conv2
        conv2_1 = self.conv_bn_layer(
            'conv2_1', pool1, [3, 3, 64, 128], is_training, drop_rate=0.4)
        conv2_2 = self.conv_bn_layer(
            'conv2_2', conv2_1, [3, 3, 128, 128], is_training, drop_rate=0.0)
        # pool2
        pool2 = tf.nn.max_pool(
            conv2_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool2')
        # conv3
        conv3_1 = self.conv_bn_layer(
            'conv3_1', pool2, [3, 3, 128, 256], is_training, drop_rate=0.4)
        conv3_2 = self.conv_bn_layer(
            'conv3_2', conv3_1, [3, 3, 256, 256], is_training, drop_rate=0.4)
        conv3_3 = self.conv_bn_layer(
            'conv3_3', conv3_2, [3, 3, 256, 256], is_training, drop_rate=0.0)
        # pool3
        pool3 = tf.nn.max_pool(
            conv3_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool3')
        # conv4
        conv4_1 = self.conv_bn_layer(
            'conv4_1', pool3, [3, 3, 256, 512], is_training, drop_rate=0.4)
        conv4_2 = self.conv_bn_layer(
            'conv4_2', conv4_1, [3, 3, 512, 512], is_training, drop_rate=0.4)
        conv4_3 = self.conv_bn_layer(
            'conv4_3', conv4_2, [3, 3, 512, 512], is_training, drop_rate=0.0)
        # pool4
        pool4 = tf.nn.max_pool(
            conv4_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4')
        # conv5
        conv5_1 = self.conv_bn_layer(
            'conv5_1', pool4, [3, 3, 512, 512], is_training, drop_rate=0.4)
        conv5_2 = self.conv_bn_layer(
            'conv5_2', conv5_1, [3, 3, 512, 512], is_training, drop_rate=0.4)
        conv5_3 = self.conv_bn_layer(
            'conv5_3', conv5_2, [3, 3, 512, 512], is_training, drop_rate=0.0)
        # pool5
        pool5 = tf.nn.max_pool(
            conv5_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4')
        # flatten
        shape = int(np.prod(pool5.get_shape()[1:]))
        pool5_flat = tf.reshape(pool5, [-1, shape])
        # fc1
        drop = tf.layers.dropout(pool5_flat, rate=0.5, training=is_training)
        fc1 = self.fc_layer('fc1', drop, [shape, 512])
        # fc2
        bn = self.batch_norm_relu(fc1, is_training)
        drop = tf.layers.dropout(bn, rate=0.5, training=is_training)
        fc2 = self.fc_layer('fc2', drop, [512, 512])

        fc3 = self.fc_layer('fc3', fc2, [512, class_dim])

        return fc3


def run_benchmark():
    """Run benchmark on cifar10 or flowers."""

    if args.data_set == "cifar10":
        class_dim = 10
        raw_shape = (3, 32, 32)
        dat_shape = (None, 32, 32, 3) if args.data_format == 'NHWC' else (
            None, 3, 32, 32)
    else:
        class_dim = 102
        raw_shape = (3, 224, 224)
        dat_shape = (None, 224, 224, 3) if args.data_format == 'NHWC' else (
            None, 3, 224, 224)

    device = '/cpu:0' if args.device == 'CPU' else '/device:GPU:0'

    with tf.device(device):
        images = tf.placeholder(tf.float32, shape=dat_shape)
        labels = tf.placeholder(tf.int64, shape=(None, ))
        is_training = tf.placeholder('bool')
        onehot_labels = tf.one_hot(labels, depth=class_dim)

        vgg16 = VGG16Model()
        logits = vgg16.network(images, class_dim, is_training)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        avg_loss = tf.reduce_mean(loss)

        correct = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(avg_loss)

    # data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10()
            if args.data_set == 'cifar10' else paddle.dataset.flowers.train(),
            buf_size=5120),
        batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.test10()
            if args.data_set == 'cifar10' else paddle.dataset.flowers.test(),
            buf_size=5120),
        batch_size=args.batch_size)

    # test
    def test():
        test_accs = []
        for batch_id, data in enumerate(test_reader()):
            test_images = np.array(
         map(lambda x: np.transpose(x[0].reshape(raw_shape),
         axes=[1, 2, 0]) if args.data_format == 'NHWC' else x[0], data)).astype("float32")
            test_labels = np.array(map(lambda x: x[1], data)).astype('int64')
            test_accs.append(
                accuracy.eval(feed_dict={
                    images: test_images,
                    labels: test_labels,
                    is_training: False
                }))
        return np.mean(test_accs)

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        iters, num_samples, start_time = 0, 0, time.time()
        for pass_id in range(args.num_passes):
            # train
            num_samples = 0
            start_time = time.time()
            for batch_id, data in enumerate(train_reader()):
                if iters == args.skip_batch_num:
                    start_time = time.time()
                    num_samples = 0
                if iters == args.iterations:
                    break
                train_images = np.array(
                    map(lambda x: np.transpose(x[0].reshape(raw_shape),
                    axes=[1, 2, 0]) if args.data_format == 'NHWC' else x[0], data)).astype("float32")
                train_labels = np.array(map(lambda x: x[1], data)).astype(
                    'int64')
                _, loss, acc = sess.run([train_op, avg_loss, accuracy],
                                        feed_dict={
                                            images: train_images,
                                            labels: train_labels,
                                            is_training: True
                                        })
                iters += 1
                num_samples += len(data)
                print("Pass = %d, Iters = %d, Loss = %f, Accuracy = %f" %
                      (pass_id, iters, loss, acc))
            train_elapsed = time.time() - start_time
            # test
            pass_test_acc = test()
            print("Pass = %d, Train speed = %f imgs/s, Test accuracy = %f\n" %
                  (pass_id, num_samples / train_elapsed, pass_test_acc))


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    print_arguments()
    run_benchmark()
