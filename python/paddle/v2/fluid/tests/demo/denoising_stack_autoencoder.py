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


def network(data, layer_sizes, noise_max, noise_min):
    assert isinstance(data, fluid.Variable)
    with fluid.unique_name.guard("noise_"):
        noise = fluid.layers.uniform_random(
            shape=data.shape, min=noise_min, max=noise_max, dtype=data.dtype)
        noise.stop_gradient = True
        noised_data = data + noise

    with fluid.unique_name.guard("encoder_"):
        hidden = noised_data
        for sz in layer_sizes:
            hidden = fluid.layers.fc(input=hidden, size=sz, act='tanh')

    with fluid.unique_name.guard("decoder_"):
        for sz in reversed(layer_sizes):
            hidden = fluid.layers.fc(input=hidden, size=sz, act='tanh')

        reconstructed_data = fluid.layers.fc(input=hidden,
                                             size=data.shape[1],
                                             act='tanh')

    loss = fluid.layers.square_error_cost(input=reconstructed_data, label=data)
    return fluid.layers.mean(x=loss)


class EncoderSaveLoadPredicate(object):
    def __init__(self):
        self.saved_var_names = set()

    def save_predicate(self, var):
        if isinstance(
                var,
                fluid.framework.Parameter) and var.name.startswith("encoder_"):
            self.saved_var_names.add(var.name)
            return True
        else:
            return False

    def load_predicate(self, var):
        return var.name in self.saved_var_names

    def reset(self):
        self.saved_var_names = set()


def main(layer_sizes=[100, 10, 10],
         noise_max=0.1,
         noise_min=-0.1,
         batch_size=32):
    place = fluid.CUDAPlace(0)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=60000),
        batch_size=batch_size)

    io_predicate = EncoderSaveLoadPredicate()

    for depth in range(len(layer_sizes)):
        depth += 1

        new_scope = fluid.core.Scope()
        main = fluid.Program()
        startup = fluid.Program()

        with fluid.scope_guard(new_scope):
            with fluid.program_guard(main, startup):
                data = fluid.layers.data(
                    name='image',
                    shape=[batch_size, 784],
                    append_batch_size=False)
                avg_loss = network(
                    data,
                    layer_sizes[:depth],
                    noise_max=noise_max,
                    noise_min=noise_min)
                adam = fluid.optimizer.Adam()
                adam.minimize(avg_loss)

                exe = fluid.Executor(place)
                exe.run(startup)
                fluid.io.load_vars(
                    exe,
                    dirname='dae_model',
                    predicate=io_predicate.load_predicate)

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
                                print 'Epoch {0}, Batch {1}, Avg Loss {2}'.format(
                                    str(epoch), str(batch_id), str(avg_loss_np))
                                moving_avg.append(avg_loss_np[0])

                                if len(moving_avg) > 5:
                                    moving_avg_result = numpy.array(
                                        moving_avg).mean()
                                    if moving_avg_result < 0.05:
                                        raise StopIteration()

                                    moving_avg = moving_avg[1:]
                except StopIteration:
                    print 'Early stop depth={0}, moving average loss={1:.4}'.format(
                        str(depth), moving_avg_result)

                fluid.io.save_vars(
                    exe,
                    dirname='dae_model',
                    predicate=io_predicate.save_predicate)

                print 'Done training depth={0}'.format(str(depth))


if __name__ == '__main__':
    main()
