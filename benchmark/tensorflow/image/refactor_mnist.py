import tensorflow.python.platform
import tensorflow as tf
import paddle.v2 as paddle
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.initializer as initializer
import paddle.v2.fluid.core as core
from paddle.v2.fluid.executor import Executor
import numpy as np
import time

BATCH_SIZE = 128
PASS_NUM = 5
SEED = 1
DTYPE = tf.float32


def normal_scale(size, channels):
    scale = (2.0 / (size**2 * channels))**0.5
    return scale


# NOTE(dzhwinter) : tensorflow use Phliox random algorithm
# as normal generator, fetch out paddle random for comparization
def paddle_random_normal(shape, loc=.0, scale=1., seed=1, dtype="float32"):
    program = framework.Program()
    block = program.global_block()
    w = block.create_var(
        dtype="float32",
        shape=shape,
        lod_level=0,
        name="param",
        initializer=initializer.NormalInitializer(
            loc=.0, scale=scale, seed=seed))
    place = core.CPUPlace()
    exe = Executor(place)
    out = exe.run(program, fetch_list=[w])
    return np.array(out[0])


train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
images = tf.placeholder(DTYPE, shape=(None, 28, 28, 1))
labels = tf.placeholder(tf.int64, shape=(None, ))

# conv layer
arg = tf.convert_to_tensor(
    np.transpose(
        paddle_random_normal(
            [20, 1, 5, 5], scale=normal_scale(5, 1), seed=SEED, dtype=DTYPE),
        axes=[2, 3, 1, 0]))
conv1_weights = tf.Variable(arg)
conv1_bias = tf.Variable(tf.zeros([20]), dtype=DTYPE)
conv1 = tf.nn.conv2d(
    images, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
pool1 = tf.nn.max_pool(
    relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

arg = tf.convert_to_tensor(
    np.transpose(
        paddle_random_normal(
            [50, 20, 5, 5], scale=normal_scale(5, 20), seed=SEED, dtype=DTYPE),
        axes=[2, 3, 1, 0]))
conv2_weights = tf.Variable(arg)
conv2_bias = tf.Variable(tf.zeros([50]), dtype=DTYPE)
conv2 = tf.nn.conv2d(
    pool1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
pool2 = tf.nn.max_pool(
    relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

pool_shape = pool2.get_shape().as_list()
hidden_dim = reduce(lambda a, b: a * b, pool_shape[1:], 1)
reshape = tf.reshape(pool2, shape=(tf.shape(pool2)[0], hidden_dim))

# fc layer
# NOTE(dzhwinter) : paddle has a NCHW data format, tensorflow has a NHWC data format
# need to convert the fc weight
paddle_weight = paddle_random_normal(
    [hidden_dim, 10],
    scale=normal_scale(hidden_dim, 10),
    seed=SEED,
    dtype=DTYPE)
new_shape = pool_shape[-1:] + pool_shape[1:-1] + [10]
paddle_weight = np.reshape(paddle_weight, new_shape)
paddle_weight = np.transpose(paddle_weight, [1, 2, 0, 3])

arg = tf.convert_to_tensor(np.reshape(paddle_weight, [hidden_dim, 10]))
fc_weights = tf.Variable(arg, dtype=DTYPE)
fc_bias = tf.Variable(tf.zeros([10]), dtype=DTYPE)
logits = tf.matmul(reshape, fc_weights) + fc_bias

# cross entropy

prediction = tf.nn.softmax(logits)

one_hot_labels = tf.one_hot(labels, depth=10)
cost = -tf.reduce_sum(tf.log(prediction) * one_hot_labels, [1])
avg_cost = tf.reduce_mean(cost)

correct = tf.equal(tf.argmax(prediction, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
g_accuracy = tf.metrics.accuracy(labels, tf.argmax(prediction, axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
train_op = optimizer.minimize(avg_cost)

with tf.Session() as sess:
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run(init_g)
    sess.run(init_l)
    pass_start = time.clock()
    for pass_id in range(PASS_NUM):
        for batch_id, data in enumerate(train_reader()):
            images_data = np.array(
                map(lambda x: np.transpose(x[0].reshape([1, 28, 28]), axes=[1,2,0]), data)).astype("float32")
            labels_data = np.array(map(lambda x: x[1], data)).astype("int64")
            start = time.clock()
            _, loss, acc, g_acc = sess.run(
                [train_op, avg_cost, accuracy, g_accuracy],
                feed_dict={images: images_data,
                           labels: labels_data})
            end = time.clock()
            # print g_acc

            print "pass=%d, batch=%d, loss=%f, error=%f, elapse=%f" % (
                pass_id, batch_id, loss, 1 - acc, (end - start) / 1000)
        print "pass=%d, accuracy=%f, elapse=%f" % (pass_id, g_acc[0], (
            time.clock() - pass_start) / 1000)
