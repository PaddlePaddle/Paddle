import os
from PIL import Image
import numpy as np
import paddle
import paddle.fluid as fluid


def softmax_regression():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    return predict


def multilayer_perceptron():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # first fully-connected layer, using ReLu as its activation function
    hidden = fluid.layers.fc(input=img, size=128, act='relu')
    # second fully-connected layer, using ReLu as its activation function
    hidden = fluid.layers.fc(input=hidden, size=64, act='relu')
    # The thrid fully-connected layer, note that the hidden size should be 10,
    # which is the number of unique digits
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction


def convolutional_neural_network():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # first conv pool
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # second conv pool
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # output layer with softmax activation function. size = 10 since there are only 10 possible digits.
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Here we can build the prediction network in different ways. Please
    # predict = softmax_regression() # uncomment for Softmax
    # predict = multilayer_perceptron() # uncomment for MLP
    predict = convolutional_neural_network()  # uncomment for LeNet5

    # Calculate the cost from the prediction and label.
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, acc]


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


def main():
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
        batch_size=64)

    test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=64)

    use_cuda = False  # set to True if training with GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_program)

    # Save the parameter into a directory. The Inferencer can load the parameters from it to do infer
    params_dirname = "recognize_digits_network.inference.model"

    lists = []

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            if event.step % 100 == 0:
                # event.metrics maps with train program return arguments.
                # event.metrics[0] will yeild avg_cost and event.metrics[1] will yeild acc in this example.
                print "Pass %d, Batch %d, Cost %f" % (event.step, event.epoch,
                                                      event.metrics[0])

        if isinstance(event, fluid.EndEpochEvent):
            avg_cost, acc = trainer.test(
                reader=test_reader, feed_order=['img', 'label'])

            print("Test with Epoch %d, avg_cost: %s, acc: %s" %
                  (event.epoch, avg_cost, acc))

            # save parameters
            trainer.save_params(params_dirname)
            lists.append((event.epoch, avg_cost, acc))

    # Train the model now
    trainer.train(
        num_epochs=5,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=['img', 'label'])

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % (float(best[2]) * 100)

    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img = load_image(cur_dir + '/image/infer_3.png')
    inferencer = fluid.Inferencer(
        # infer_func=softmax_regression, # uncomment for softmax regression
        # infer_func=multilayer_perceptron, # uncomment for MLP
        infer_func=convolutional_neural_network,  # uncomment for LeNet5
        param_path=params_dirname,
        place=place)

    results = inferencer.infer({'img': img})
    lab = np.argsort(results)  # probs and lab are the results of one batch data
    print "Label of image/infer_3.png is: %d" % lab[0][0][-1]


if __name__ == '__main__':
    main()
