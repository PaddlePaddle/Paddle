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

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 128, "Training batch size")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    print "cluster:", cluster
    print "ps:", ps_hosts
    print "worker_host:", worker_hosts

    # Create and start a server for the local task.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):

            # Variables of the hidden layer
            hid_w = tf.Variable(
                tf.truncated_normal(
                    [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                    stddev=1.0 / IMAGE_PIXELS),
                name="hid_w")
            hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

            # Variables of the softmax layer
            sm_w = tf.Variable(
                tf.truncated_normal(
                    [FLAGS.hidden_units, 10],
                    stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
                name="sm_w")
            sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

            x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
            y_ = tf.placeholder(tf.float32, [None, 10])

            hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
            hid = tf.nn.relu(hid_lin)

            y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
            loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

            global_step = tf.Variable(0)

            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                loss, global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.initialize_all_variables()

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(
            is_chief=(FLAGS.task_index == 0),
            logdir="/tmp/train_logs",
            init_op=init_op,
            summary_op=summary_op,
            saver=saver,
            global_step=global_step,
            save_model_secs=600)

        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            while not sv.should_stop() and step < 1000000:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                train_feed = {x: batch_xs, y_: batch_ys}

                _, step = sess.run([train_op, global_step],
                                   feed_dict=train_feed)
                if step % 100 == 0:
                    print "Done step %d" % step

        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
