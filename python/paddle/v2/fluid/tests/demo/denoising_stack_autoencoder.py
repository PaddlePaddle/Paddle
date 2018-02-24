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

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import numpy
import contextlib
import argparse


def network(data, layer_sizes, noise_max, noise_min, intermedia_act='tanh'):
    assert isinstance(data, fluid.Variable)
    with fluid.unique_name.guard("encoder_"):
        hidden = data
        for sz in layer_sizes[:-1]:
            hidden = fluid.layers.fc(input=hidden, size=sz, act=intermedia_act)

        with fluid.unique_name.guard("noise_"):
            noise = fluid.layers.uniform_random(
                shape=hidden.shape,
                min=noise_min,
                max=noise_max,
                dtype=hidden.dtype)
            noise.stop_gradient = True
            hidden += noise

        hidden = fluid.layers.fc(input=hidden,
                                 size=layer_sizes[-1],
                                 act=intermedia_act)

    with fluid.unique_name.guard("decoder_"):
        reconstructed_data = fluid.layers.fc(input=hidden,
                                             size=data.shape[1],
                                             act=intermedia_act)

    loss = fluid.layers.square_error_cost(input=reconstructed_data, label=data)
    return fluid.layers.mean(x=loss)


class EncoderSaveLoadPredicate(object):
    def __init__(self):
        self.saved_var_names = set()

    def save_predicate(self, var):
        if var.name in self.saved_var_names:
            return True
        if isinstance(
                var,
                fluid.framework.Parameter) and var.name.startswith("encoder_"):
            self.saved_var_names.add(var.name)
            return True
        else:
            return False

    def load_predicate(self, var):
        return var.name in self.saved_var_names


@contextlib.contextmanager
def new_scope_prog():
    new_scope = fluid.core.Scope()
    main = fluid.Program()
    startup = fluid.Program()

    with fluid.scope_guard(new_scope):
        with fluid.program_guard(main, startup):
            yield


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", nargs="+", help='Layer sizes', required=True, type=int)
    parser.add_argument(
        '--noise_min', help='the min value of noise', default=-0.1, type=float)
    parser.add_argument(
        '--noise_max', help='the max value of noise', default=0.1, type=float)
    parser.add_argument(
        '--use_cuda', help='use cuda or not', action='store_true')
    parser.add_argument(
        '--batch_size', help='the batch size', default=32, type=int)
    parser.add_argument(
        '--stop_loss',
        help='the loss to stop iteration',
        nargs="+",
        default=[0.05],
        type=float)
    parser.add_argument(
        '--avg_window_size',
        help='the window size to calculate loss',
        default=5,
        type=int)
    return parser.parse_args()


def train(layer_sizes, noise_max, noise_min, batch_size, place, stop_loss,
          avg_window_size):
    if len(stop_loss) == 1:
        stop_loss = stop_loss * len(layer_sizes)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=60000),
        batch_size=batch_size)

    io_predicate = EncoderSaveLoadPredicate()

    for depth in range(len(layer_sizes)):
        depth += 1

        with new_scope_prog():
            data = fluid.layers.data(
                name='image', shape=[batch_size, 784], append_batch_size=False)
            avg_loss = network(
                data,
                layer_sizes[:depth],
                noise_max=noise_max,
                noise_min=noise_min)
            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(avg_loss)

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            fluid.io.load_vars(
                exe, dirname='dae_model', predicate=io_predicate.load_predicate)

            feeder = fluid.DataFeeder(
                feed_list=[data], place=place, batch_size_dim=0)

            try:
                for epoch in range(100):
                    moving_avg = []
                    for batch_id, batch_data in enumerate(train_reader()):
                        if len(batch_data) != batch_size:
                            break
                        batch_data = map(lambda x: [x[0]], batch_data)

                        should_print = (batch_id + 1) % 100 == 0
                        avg_loss_np = \
                            exe.run(feed=feeder.feed(batch_data), fetch_list=[avg_loss] if should_print else [])
                        if len(avg_loss_np) != 0:
                            avg_loss_np = avg_loss_np[0]
                            print 'Stack Depth {3}, Epoch {0}, Batch {1}, Avg Loss {2}'.format(
                                str(epoch),
                                str(batch_id), str(avg_loss_np), str(depth))
                            moving_avg.append(avg_loss_np[0])

                            if len(moving_avg) == avg_window_size:
                                moving_avg_result = numpy.array(
                                    moving_avg).mean()
                                if moving_avg_result < stop_loss[depth - 1]:
                                    raise StopIteration()

                                moving_avg = moving_avg[1:]
            except StopIteration:
                print 'Early stop depth={0}, moving average loss={1:.4}'.format(
                    str(depth), moving_avg_result)

            fluid.io.save_vars(
                exe, dirname='dae_model', predicate=io_predicate.save_predicate)

            print 'Done training depth={0}'.format(str(depth))


def main():
    args = parse_arg()
    print "Training a stacked autoencoder with args: {0}".format(str(args))
    train(args.size, args.noise_max, args.noise_min, args.batch_size,
          fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace(),
          args.stop_loss, args.avg_window_size)


if __name__ == '__main__':
    main()
