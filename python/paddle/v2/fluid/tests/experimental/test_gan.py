import math

import matplotlib
import numpy
import paddle.v2.fluid.core as core

import paddle.v2 as paddle
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.io import save_params
from paddle.v2.fluid.optimizer import AdamOptimizer

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_WIDTH = 10


class Counter(object):
    def __init__(self):
        self.counter = 0

    def __call__(self, *args, **kwargs):
        tmp = self.counter
        self.counter += 1
        return tmp


def D(x, main_program, startup_program):
    c = Counter()
    kwargs = {'main_program': main_program, 'startup_program': startup_program}

    hidden = layers.fc(input=x,
                       size=200,
                       act='tanh',
                       param_attr={"name": "D_%d" % c()},
                       bias_attr={"name": "D_%d" % c()},
                       **kwargs)
    prob = layers.fc(input=hidden,
                     size=1,
                     act=None,
                     param_attr={'name': 'D_%d' % c()},
                     bias_attr={'name': 'D_%d' % c()},
                     **kwargs)
    return prob


def G(x, main_program, startup_program):
    c = Counter()
    kwargs = {'main_program': main_program, 'startup_program': startup_program}
    hidden = layers.fc(input=x,
                       size=100,
                       act='tanh',
                       param_attr={"name": "G_%d" % c()},
                       bias_attr={"name": "G_%d" % c()},
                       **kwargs)
    hidden = layers.fc(input=hidden,
                       size=200,
                       act='tanh',
                       param_attr={"name": "G_%d" % c()},
                       bias_attr={"name": "G_%d" % c()},
                       **kwargs)
    out = layers.fc(input=hidden,
                    size=784,
                    act=None,
                    param_attr={"name": "G_%d" % c()},
                    bias_attr={"name": "G_%d" % c()},
                    **kwargs)
    return out


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


def main():
    startup_program = framework.default_startup_program()
    d_program = framework.Program()
    d_kwargs = {'main_program': d_program, 'startup_program': startup_program}
    g_program = framework.Program()
    g_kwargs = {'main_program': g_program, 'startup_program': startup_program}

    img = layers.data(
        name='image', shape=[784], data_type='float32', **d_kwargs)
    label = layers.data(
        name='label', shape=[1], data_type='float32', **d_kwargs)
    d_prob = D(img, **d_kwargs)

    d_loss = layers.sigmoid_cross_entropy_with_logits(
        x=d_prob, labels=label, **d_kwargs)
    d_loss = layers.mean(x=d_loss, **d_kwargs)

    noise = layers.data(
        name='noise', shape=[NOISE_WIDTH], data_type='float32', **g_kwargs)
    g_data = G(noise, **g_kwargs)

    dg_program = g_program.clone()
    dg_kwargs = {'main_program': dg_program, 'startup_program': startup_program}

    dg_data = dg_program.global_block().var(g_data.name)
    dg_prob = layers.sigmoid(x=D(dg_data, **dg_kwargs), **dg_kwargs)
    # dg_loss = -log(prob)
    dg_loss = layers.scale(
        x=layers.log(x=dg_prob, **dg_kwargs), scale=-1.0, **dg_kwargs)
    dg_loss = layers.mean(x=dg_loss, **dg_kwargs)
    opt = AdamOptimizer(learning_rate=1e-5)

    opt.minimize(loss=d_loss, startup_program=startup_program)
    # only optimize G when optimize dg_loss
    opt.minimize(
        loss=dg_loss,
        startup_program=startup_program,
        parameter_list=[
            p.name for p in g_program.global_block().all_parameters()
        ])

    cpu = core.CPUPlace()
    exe = Executor(cpu)
    exe.run(startup_program)

    num_true = 36
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=60000),
        batch_size=num_true)

    for pass_id in xrange(1000):
        save_params(
            main_program=d_program,
            dirname='model_{0}'.format(str(pass_id).zfill(3)),
            executor=exe)
        save_params(
            main_program=g_program,
            dirname='model_{0}'.format(str(pass_id).zfill(3)),
            executor=exe)
        for batch_id, data in enumerate(train_reader()):
            # Generate Fake Data
            num_true = len(data)

            n = numpy.random.uniform(
                low=-1.0, high=1.0,
                size=[num_true * NOISE_WIDTH]).astype('float32').reshape(
                    [num_true, NOISE_WIDTH])
            n_tensor = core.LoDTensor()
            n_tensor.set(n, cpu)
            gen_data = numpy.array(
                exe.run(g_program,
                        feed={'noise': n_tensor},
                        fetch_list=[g_data])[0])

            # Train D
            real_data = numpy.array(map(lambda x: x[0], data)).astype('float32')
            total_data = numpy.concatenate([real_data, gen_data])
            total_label = numpy.concatenate([
                numpy.ones(
                    shape=[real_data.shape[0], 1], dtype='float32'),
                numpy.zeros(
                    shape=[real_data.shape[0], 1], dtype='float32')
            ])

            d_tensor = core.LoDTensor()
            d_tensor.set(total_data, cpu)

            l_tensor = core.LoDTensor()
            l_tensor.set(total_label, cpu)

            d_loss_np = numpy.array(
                exe.run(d_program,
                        feed={'image': d_tensor,
                              'label': l_tensor},
                        fetch_list=[d_loss])[0])

            # Train D(G(x))
            # Train D(G) twice, D once.
            for _ in xrange(2):
                n = numpy.random.uniform(
                    low=-1.0, high=1.0,
                    size=[num_true * NOISE_WIDTH]).astype('float32').reshape(
                        [num_true, NOISE_WIDTH])
                n_tensor = core.LoDTensor()
                n_tensor.set(n, cpu)

                dg_loss_np = numpy.array(
                    exe.run(dg_program,
                            feed={'noise': n_tensor},
                            fetch_list=[dg_loss])[0])

            print "PassID {0}, BatchID {1}, D-Loss {2}, D(G)-Loss {3}, " \
                  "total-loss {4}".format(
                str(pass_id), str(batch_id), str(d_loss_np), str(dg_loss_np),
                str(d_loss_np + dg_loss_np))

        fig = plot(gen_data)
        plt.savefig(
            'out/{0}.png'.format(str(pass_id).zfill(3)), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
