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

import unittest

import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import numpy


class BaseParallelForTest(unittest.TestCase):
    def run_test(self, callback, feed, fetch):
        """
        Run the unittest for parallel.for
        Args:
            callback(callable): A callable function returns a generator. There 
                are two yields in the generator function. The first yield 
                returns the data layers, and the second yield returns the loss. 
                The modified data variables will be sent back during the first 
                yield.

            feed(dict): The executor feeding dictionary.
            fetch(list|basestr): The fetch name lists. 

        Returns:
            None

        Raises:
            AssertionError when the computation of cpu, parallel.for in cpu, 
                gpu, parallel.for in gpu are different.

        """
        cpu = fluid.CPUPlace()
        result_cpu = self._run_test_impl_(
            callback=callback,
            feed=feed,
            fetch=fetch,
            place=cpu,
            use_parallel=False)
        result_cpu_parallel = self._run_test_impl_(
            callback=callback,
            feed=feed,
            fetch=fetch,
            place=cpu,
            use_parallel=True)
        if fluid.core.is_compiled_with_cuda():
            gpu = fluid.CUDAPlace(0)
            result_gpu = self._run_test_impl_(
                callback=callback,
                feed=feed,
                fetch=fetch,
                place=gpu,
                use_parallel=False,
                use_gpu=True)
            result_gpu_parallel = self._run_test_impl_(
                callback=callback,
                feed=feed,
                fetch=fetch,
                place=gpu,
                use_parallel=True,
                use_gpu=True)
            result_gpu_nccl = self._run_test_impl_(
                callback=callback,
                feed=feed,
                fetch=fetch,
                place=gpu,
                use_parallel=True,
                use_nccl=True,
                use_gpu=True)
            self._assert_same_(fetch, result_cpu, result_cpu_parallel,
                               result_gpu, result_gpu_parallel, result_gpu_nccl)
        else:
            self._assert_same_(fetch, result_cpu, result_cpu_parallel)

    def _run_test_impl_(self,
                        callback,
                        feed,
                        fetch,
                        place,
                        use_parallel=False,
                        use_nccl=False,
                        use_gpu=False):
        """
        Run a single test, returns the fetch values
        Args:
            place(Place): the computation place. 
            use_parallel(bool): Whether use parallel.for or not. 

        Returns:
            Fetched numpy arrays.

        """
        if isinstance(fetch, basestring):
            fetch = [fetch]
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
                pd = fluid.layers.ParallelDo(places, use_nccl=use_nccl)
                data = next(generator)

                if isinstance(data, fluid.Variable):
                    data = [data]

                with pd.do():
                    ins = map(pd.read_input, data)
                    if len(ins) == 1:
                        ins = ins[0]
                    loss = generator.send(ins)  # patch input
                    pd.write_output(loss)

                loss = pd()
            else:
                data = next(generator)
                loss = generator.send(data)
            self.assertIsNotNone(loss)
            avg_loss = fluid.layers.mean(loss)
            fluid.backward.append_backward(loss=avg_loss)

        exe = fluid.Executor(place)
        exe.run(startup)
        if use_gpu:
            profile_type = 'GPU'
        else:
            profile_type = 'CPU'
        with profiler.profiler(profile_type, 'total', '/tmp/profiler'):
            return exe.run(main, feed=feed, fetch_list=fetch)

    def _assert_same_(self, fetch, *args):
        """
        Assert the return values of `run_test` are same.
        Args:
            fetch: Fetch list. Used for print error message
            *args: The fetch result lists of each situations.

        Returns:
            None
            
        Raises:
            AssertionError

        """

        def _impl_(a, b, fetch_id, item_id):
            item_str = [
                'CPU', 'ParallelCPU', 'GPU', 'ParallelGPU', 'ParallelGPUNCCL'
            ]
            flag = numpy.allclose(a, b, rtol=0.1, atol=1e-3)
            self.assertTrue(flag,
                            "The {0} are different in {1}, {2} vs {3}".format(
                                fetch[fetch_id], item_str[item_id], a, b))

        for i, items in enumerate(zip(*args)):
            self.assertGreater(len(items), 0)
            for j in range(1, len(items)):
                _impl_(items[0], items[j], fetch_id=i, item_id=j)


class ParallelOpTest(BaseParallelForTest):
    @staticmethod
    def __network__():
        x = fluid.layers.data(shape=[784], dtype='float32', name='img')
        x = yield x
        hidden = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
        hidden = fluid.layers.batch_norm(input=hidden)
        loss = fluid.layers.mean(hidden)
        yield loss

    def test_simple_fc(self):
        self.run_test(
            callback=self.__network__,
            feed={
                'img': numpy.random.random(size=(51, 784)).astype('float32')
            },
            fetch=['fc1.w@GRAD'])

    def test_fc_with_tiny_data(self):
        self.run_test(
            callback=self.__network__,
            feed={'img': numpy.random.random(size=(1, 784)).astype('float32')},
            fetch=['fc1.w@GRAD'])


class ParallelOpTestMultipleInput(BaseParallelForTest):
    @staticmethod
    def __network__():
        x = fluid.layers.data(
            shape=[784], dtype='float32', name='img1', stop_gradient=False)
        y = fluid.layers.data(
            shape=[784], dtype='float32', name='img2', stop_gradient=False)
        yield [x, y]
        x = x + y
        hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
        hidden2 = fluid.layers.fc(input=hidden1, size=200, param_attr='fc2.w')
        hidden3 = fluid.layers.fc(input=hidden2, size=200, param_attr='fc3.w')
        loss = fluid.layers.mean(hidden3)
        yield loss

    def test_simple_fc(self):
        self.run_test(
            callback=self.__network__,
            feed={
                'img1': numpy.random.random(size=(51, 784)).astype('float32'),
                'img2': numpy.random.random(size=(51, 784)).astype('float32')
            },
            fetch=['fc1.w@GRAD', 'fc2.w@GRAD', 'fc3.w@GRAD'])


if __name__ == '__main__':
    unittest.main()
