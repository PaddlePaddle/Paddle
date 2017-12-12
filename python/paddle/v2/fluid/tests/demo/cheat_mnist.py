import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import random
import numpy

import math
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(gen_data):
    gen_data.resize(gen_data.shape[0], 28, 28)
    n = int(math.ceil(math.sqrt(gen_data.shape[0])))
    fig = plt.figure(figsize=(n, n))
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(gen_data):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def train_net(x, label):
    hidden = fluid.layers.fc(input=x,
                             size=200,
                             act='tanh',
                             param_attr='classification.fc1.w',
                             bias_attr='classification.fc1.b')
    prediction = fluid.layers.fc(input=hidden,
                                 size=10,
                                 act='softmax',
                                 param_attr='classification.fc2.w',
                                 bias_attr='classification.fc2.b')
    return fluid.layers.mean(x=fluid.layers.cross_entropy(
        input=prediction, label=label)), prediction


def train_classification():
    train_program = fluid.Program()
    train_startup_program = fluid.Program()
    with fluid.program_guard(train_program, train_startup_program):
        x = fluid.layers.data(name='img', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        loss, prediction = train_net(x, label)
        adam = fluid.optimizer.Adam()
        adam.minimize(loss=loss)
        acc = fluid.evaluator.Accuracy(input=prediction, label=label)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=8192),
        batch_size=1024)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(train_startup_program)
    feeder = fluid.DataFeeder(feed_list=[x, label], place=place)

    for pass_id in range(100):
        acc.reset(exe)
        for data in train_reader():
            exe.run(train_program, feed=feeder.feed(data))
        pass_acc = acc.eval(exe)
        if pass_acc[0] > 0.95:
            break

    fluid.io.save_params(exe, dirname='./mnist', main_program=train_program)
    print 'train mnist done'


def train_cheat_net():
    cheat_init_program = fluid.Program()
    data_program = fluid.Program()
    with fluid.program_guard(data_program):
        x = fluid.layers.data(
            name='img',
            shape=[1, 784],
            dtype='float32',
            append_batch_size=False)
        x.stop_gradient = False
        x.persistable = True
        label = fluid.layers.data(
            name='label', shape=[1, 1], dtype='int64', append_batch_size=False)
        label.persistable = True

    cheat_program = data_program.clone()

    with fluid.program_guard(cheat_program, cheat_init_program):
        loss, prediction = train_net(
            fluid.layers.clip(
                x=x, min=-1.0, max=1.0), label)
        adam = fluid.optimizer.Adam()
        adam.minimize(loss=loss, parameter_list=[x.name])  # only optimize x
        acc = fluid.layers.accuracy(input=prediction, label=label)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    train_reader = paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192)
    counter = 0
    for data, label in train_reader():
        new_lbl = random.randint(0, 9)
        while new_lbl == label:
            new_lbl = random.randint(0, 9)
        fluid.io.load_params(
            executor=exe, dirname='./mnist', main_program=cheat_program)
        exe.run(program=data_program,
                feed={
                    'img': numpy.array(
                        data, dtype='float32').reshape(1, 784),
                    'label': numpy.array(
                        [new_lbl], dtype='int64').reshape(1, 1)
                })  # feed image
        exe.run(cheat_init_program)  # reset train parameters
        acc_np = [0.]
        while acc_np[0] < 0.5:  # acc should be 0 or 1 since batch_size == 1.
            loss_np, acc_np = exe.run(program=cheat_program,
                                      fetch_list=[loss, acc])

        generated_img = exe.run(program=data_program, fetch_list=[x])[0]
        fig = plot(generated_img)
        fig.savefig(
            '{0}_{1}.png'.format(str(counter).zfill(6), str(new_lbl)),
            bbox_inches='tight')
        plt.close(fig)
        counter += 1
        print 'generate a fake image for label %d' % new_lbl


if __name__ == '__main__':
    # the following line can be commented, if the model has been trained before
    train_classification()
    train_cheat_net()
