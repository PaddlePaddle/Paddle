import paddle.v2.fluid as fluid
import paddle.v2 as paddle
import numpy
import os


class Counter(object):
    def __init__(self):
        self.counter = 0

    def __call__(self, *args, **kwargs):
        tmp = self.counter
        self.counter += 1
        return str(tmp)

    def __str__(self):
        return self.__call__()


def D(x):
    c = Counter()
    hidden = fluid.layers.fc(input=x,
                             size=200,
                             act='relu',
                             param_attr='D_{0}'.format(c),
                             bias_attr='D_{0}'.format(c))
    logits = fluid.layers.fc(input=hidden,
                             size=1,
                             act=None,
                             param_attr='D_{0}'.format(c),
                             bias_attr='D_{0}'.format(c))
    return logits


def G(x):
    c = Counter()
    hidden = fluid.layers.fc(input=x,
                             size=200,
                             act='relu',
                             param_attr='G_{0}'.format(c),
                             bias_attr='G_{0}'.format(c))
    hidden = fluid.layers.fc(input=hidden,
                             size=784,
                             act='tanh',
                             param_attr='G_{0}'.format(c),
                             bias_attr='G_{0}'.format(c))
    return hidden


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


NOISE_SIZE = 100
NUM_PASS = 10


def main():
    startup_program = fluid.Program()
    d_program = fluid.Program()
    dg_program = fluid.Program()
    with fluid.program_guard(d_program, startup_program):
        img = fluid.layers.data(name='img', shape=[784], dtype='float32')
        logit = D(img)
        d_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=logit,
            labels=fluid.layers.data(
                name='label', shape=[1], dtype='float32'))
        d_loss = fluid.layers.mean(x=d_loss)

    with fluid.program_guard(dg_program, startup_program):
        noise = fluid.layers.data(
            name='noise', shape=[NOISE_SIZE], dtype='float32')
        g_img = G(x=noise)
        g_program = dg_program.clone()
        logit = D(g_img)
        dg_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=logit,
            labels=fluid.layers.fill_constant_batch_size_like(
                input=noise, dtype='float32', shape=[-1, 1], value=1.0))
        dg_loss = fluid.layers.mean(x=dg_loss)

    opt = fluid.optimizer.Adam(learning_rate=1e-5)

    opt.minimize(loss=d_loss, startup_program=startup_program)
    opt.minimize(
        loss=dg_loss,
        startup_program=startup_program,
        parameter_list=[
            p.name for p in g_program.global_block().all_parameters()
        ])
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(startup_program)

    num_true = 121
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=60000),
        batch_size=num_true)

    for pass_id in range(NUM_PASS):
        for batch_id, data in enumerate(train_reader()):
            num_true = len(data)
            n = numpy.random.uniform(
                low=-1.0, high=1.0,
                size=[num_true * NOISE_SIZE]).astype('float32').reshape(
                    [num_true, NOISE_SIZE])
            generated_img = exe.run(g_program,
                                    feed={'noise': n},
                                    fetch_list={g_img})[0]

            real_data = numpy.array(map(lambda x: x[0], data)).astype('float32')
            real_data = real_data.reshape(num_true, 784)
            total_data = numpy.concatenate([real_data, generated_img])
            total_label = numpy.concatenate([
                numpy.ones(
                    shape=[real_data.shape[0], 1], dtype='float32'),
                numpy.zeros(
                    shape=[real_data.shape[0], 1], dtype='float32')
            ])
            d_loss_np = exe.run(d_program,
                                feed={'img': total_data,
                                      'label': total_label},
                                fetch_list={d_loss})[0]

            n = numpy.random.uniform(
                low=-1.0, high=1.0,
                size=[2 * num_true * NOISE_SIZE]).astype('float32').reshape(
                    [2 * num_true, NOISE_SIZE, 1, 1])
            dg_loss_np = exe.run(dg_program,
                                 feed={'noise': n},
                                 fetch_list={dg_loss})[0]
            print("Pass ID={0}, Batch ID={1}, D-Loss={2}, DG-Loss={3}".format(
                pass_id, batch_id, d_loss_np, dg_loss_np))
        os.makedirs("./out/")
        fig = plot(generated_img)
        plt.savefig(
            'out/{0}.png'.format(str(pass_id).zfill(3)), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
