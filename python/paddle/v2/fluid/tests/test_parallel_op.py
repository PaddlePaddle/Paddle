import unittest
import paddle.v2.fluid as fluid
import numpy


class BaseParallelForTest(unittest.TestCase):
    def main(self, callback, feed, fetch):
        cpu = fluid.CPUPlace()
        result_cpu = self._main_impl_(
            callback=callback,
            feed=feed,
            fetch=fetch,
            place=cpu,
            use_parallel=False)
        print result_cpu

    def _main_impl_(self, callback, feed, fetch, place, use_parallel=False):
        main = fluid.Program()
        startup = fluid.Program()
        # Fix seed
        main.random_seed = 10
        startup.random_seed = 10

        with fluid.program_guard(main, startup):
            generator = callback()
            # Automatically insert parallel do if use_parallel = True
            if use_parallel:
                places = fluid.layers.get_places()
                pd = fluid.layers.ParallelDo(places)
                data = next(generator)

                if isinstance(data, fluid.Variable):
                    data = [data]
                with pd.do():
                    ins = map(pd.read_input, data)
                    if len(ins) == 1:
                        ins = ins[0]
                    generator.send(ins)  # patch input
                    loss = next(generator)
                    pd.write_output(loss)

                loss = pd()
            else:
                data = next(generator)
                generator.send(data)
                loss = next(generator)

            avg_loss = fluid.layers.mean(x=loss)
            fluid.backward.append_backward(loss=avg_loss)

        exe = fluid.Executor(place)
        exe.run(startup)
        return exe.run(main, feed=feed, fetch_list=fetch)


class ParallelOpTest(BaseParallelForTest):
    def test_simple_fc(self):
        def __network__():
            x = fluid.layers.data(shape=[784], dtype='float32', name='img')
            x = yield x
            hidden = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
            loss = fluid.layers.mean(x=hidden)
            yield loss

        self.main(
            callback=__network__,
            feed={
                'img': numpy.random.random(size=(128, 784)).astype('float32')
            },
            fetch='fc1.w@GRAD')


if __name__ == '__main__':
    unittest.main()
