"""
CNN on mnist data using fluid api of paddlepaddle
"""
import paddle.v2 as paddle
import paddle.v2.fluid as fluid


def mnist_cnn_model(img):
    """
    Mnist cnn model

    Args:
        img(Varaible): the input image to be recognized

    Returns:
        Variable: the label prediction
    """
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

    logits = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return logits


def main():
    """
    Train the cnn model on mnist datasets
    """
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    logits = mnist_cnn_model(img)
    cost = fluid.layers.cross_entropy(input=logits, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    accuracy = fluid.evaluator.Accuracy(input=logits, label=label)

    BATCH_SIZE = 50
    PASS_NUM = 3
    ACC_THRESHOLD = 0.98
    LOSS_THRESHOLD = 10.0
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(fluid.default_startup_program())

    for pass_id in range(PASS_NUM):
        accuracy.reset(exe)
        for data in train_reader():
            loss, acc = exe.run(fluid.default_main_program(),
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost] + accuracy.metrics)
            pass_acc = accuracy.eval(exe)
            print("pass_id=" + str(pass_id) + " acc=" + str(acc) + " pass_acc="
                  + str(pass_acc))
            if loss < LOSS_THRESHOLD and pass_acc > ACC_THRESHOLD:
                break

        pass_acc = accuracy.eval(exe)
        print("pass_id=" + str(pass_id) + " pass_acc=" + str(pass_acc))
    fluid.io.save_params(
        exe, dirname='./mnist', main_program=fluid.default_main_program())
    print('train mnist done')


if __name__ == '__main__':
    main()
