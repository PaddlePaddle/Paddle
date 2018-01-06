"""
This attack was originally implemented by Goodfellow et al. (2015) with the
infinity norm (and is known as the "Fast Gradient Sign Method"). This is therefore called
the Fast Gradient Method.
Paper link: https://arxiv.org/abs/1412.6572
"""

import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

BATCH_SIZE = 50
PASS_NUM = 1
EPS = 0.3
CLIP_MIN = -1
CLIP_MAX = 1
PASS_NUM = 1

def mnist_cnn_model(img):
    """
    Mnist cnn model

    Args:
        img(Varaible): the input image to be recognized

    Returns:
        Variable: the label prediction
    """
    #conv1 = fluid.nets.conv2d()
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        num_filters=20,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu')

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        num_filters=50,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu')

    logits = fluid.layers.fc(
        input=conv_pool_2,
        size=10,
        act='softmax')
    return logits


def main():
    """
    Generate adverserial example and evaluate accuracy on mnist using FGSM
    """

    images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype='float32')
    # The gradient should flow
    images.stop_gradient = False
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    predict = mnist_cnn_model(images)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Cal gradient of input
    params_grads = fluid.backward.append_backward_ops(avg_cost, parameter_list=['pixel'])
    # data batch
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    accuracy = fluid.evaluator.Accuracy(input=predict, label=label)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    accuracy.reset(exe)
    #exe.run(fluid.default_startup_program())
    feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
    for pass_id in range(PASS_NUM):
        fluid.io.load_params(exe, "./mnist/", main_program=fluid.default_main_program())
        for data in train_reader():
            # cal gradient and eval accuracy
            ps, acc = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[params_grads[0][1]]+accuracy.metrics)
            labels = []
            for idx, _ in enumerate(data):
                labels.append(data[idx][1])
            # generate adversarial example
            batch_num = ps.shape[0]
            new_data = []
            for i in range(batch_num):
                adv_img = np.reshape(data[0][0], (1, 28, 28)) + EPS * np.sign(ps[i])
                adv_img = np.clip(adv_img, CLIP_MIN, CLIP_MAX)
                #adv_imgs.append(adv_img)
                t = (adv_img, data[0][1])
                new_data.append(t)

            # predict label
            predict_label, = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(new_data),
                fetch_list=[predict])
            adv_labels = np.argmax(predict_label, axis=1)
            batch_accuracy = np.mean(np.equal(labels, adv_labels))
            print "pass_id=" + str(pass_id) + " acc=" + str(acc)+ " adv_acc=" + str(batch_accuracy)


if __name__ == "__main__":
    main()
