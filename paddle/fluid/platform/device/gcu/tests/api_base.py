# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging
from inspect import isfunction

import numpy as np

import paddle

is_tuple = lambda var: isinstance(var, (tuple))
is_tuple_list = lambda var: isinstance(var, (tuple, list))


class ApiBase:
    def __init__(
        self,
        func,
        feed_names,
        feed_shapes,
        feed_dtypes=None,
        input_is_list=False,
        is_train=True,
        threshold=1.0e-5,
        rel_tol=1.0e-6,
        equal_nan=False,
    ):
        self.func = func
        self.feed_list = feed_names
        self.feed_shapes = feed_shapes
        self.threshold = threshold
        self.rel_tol = rel_tol
        self.equal_nan = equal_nan
        if feed_dtypes:
            self.feed_dtypes = feed_dtypes
        else:
            self.feed_dtypes = ['float32'] * len(self.feed_list)
        assert len(self.feed_list) == len(self.feed_shapes) and len(
            self.feed_list
        ) == len(self.feed_dtypes)
        self.is_train = is_train
        self.input_is_list = input_is_list
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def compare(self, cpu_res, gcu_res):
        assert len(cpu_res) == len(gcu_res)
        for i in range(len(cpu_res)):
            out = gcu_res[i]
            exp = cpu_res[i]
            assert (
                out.shape == exp.shape
            ), f"out shape: {out.shape}, expect: {exp.shape}"
            if exp.dtype in [np.float16, np.float32, np.float64]:
                np.testing.assert_allclose(
                    out,
                    exp,
                    rtol=self.rel_tol,
                    atol=self.threshold,
                    equal_nan=self.equal_nan,
                )
            elif exp.dtype in [
                bool,
                np.int8,
                np.uint8,
                np.int16,
                np.uint16,
                np.int32,
                np.uint32,
                np.int64,
                np.uint64,
            ]:
                assert np.all(out == exp)
            else:
                assert logging.info('unsupport data type')
                assert 0

    def get_legal_cpu_fetch_list(self, out):
        if is_tuple(out):
            cpu_fetch_list = []
            for i in range(len(out)):
                if is_tuple(out[i]):
                    cpu_fetch_list.append(list(out[i]))
                else:
                    cpu_fetch_list.append(out[i])
            return cpu_fetch_list
        else:
            return [out]

    def run(self, feed, **kwargs):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        main_program.random_seed = 33
        startup_program.random_seed = 33
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(
                main_program=main_program, startup_program=startup_program
            ):
                inputs = []

                def input_data(name, shape, dtype):
                    data = paddle.static.data(
                        name=name, shape=shape, dtype=dtype
                    )
                    data.stop_gradient = False
                    return data

                for i in range(len(self.feed_list)):
                    # if (is_tuple(self.feed_shapes[i]) and not is_tuple(self.feed_dtypes[i])):
                    #     len_tuple = len(self.feed_shapes[i])
                    #     data_tuple = []
                    #     for k in range(len_tuple):
                    #         data = input_data(self.feed_list[i][k], self.feed_shapes[i][k], self.feed_dtypes[i])
                    #         data_tuple.append(data)
                    #     inputs.append()
                    #     # data_tuple = input_data(self.feed_list[i], list(self.feed_shapes[i]), self.feed_dtypes[i])
                    #     # inputs.append(data_tuple)
                    #     # data_tuple = []
                    #     # for feed_shape in self.feed_shapes[i]:
                    #     #     data = input_data(self.feed_list[i], feed_shape, self.feed_dtypes[i])
                    #     #     data_tuple.append(data)
                    #     # inputs.append(tuple(data_tuple))
                    # else:
                    data = input_data(
                        self.feed_list[i],
                        self.feed_shapes[i],
                        self.feed_dtypes[i],
                    )
                    inputs.append(data)
                out = None
                if isfunction(self.func):
                    if self.input_is_list:
                        out = self.func(inputs, **kwargs)
                    else:
                        out = self.func(*inputs, **kwargs)
                else:
                    obj = self.func(**kwargs)
                    if self.input_is_list:
                        out = obj(inputs)
                    else:
                        out = obj(*inputs)

                fetch_list = []
                if is_tuple_list(out):
                    for item in out:
                        if is_tuple_list(item):
                            fetch_list.append(
                                [mini_item.name for mini_item in item]
                            )
                        else:
                            fetch_list.append(item.name)
                else:
                    fetch_list.append(out.name)
                print(">>>>>> fetch_list:\n", fetch_list)

                # fetch_list = [item.name for item in out]  else [out.name]
                if self.is_train:
                    if isinstance(out, (list, tuple)):
                        loss = paddle.mean(out[0])
                    else:
                        loss = paddle.mean(out)
                    g = paddle.static.gradients(loss, inputs)
                    for grad in g:
                        fetch_list.append(grad.name)
                feed_dict = {}
                for i in range(len(self.feed_list)):
                    feed_dict.update({self.feed_list[i]: feed[i]})
                print(">>>>>> feed_dict:\n", feed_dict)

                # get golden data from cpu result
                cpu_place = paddle.CPUPlace()
                cpu_exe = paddle.static.Executor(cpu_place)
                cpu_exe.run(startup_program)
                cpu_fetch_list = self.get_legal_cpu_fetch_list(out)
                print(">>>>>> cpu_fetch_list:\n", cpu_fetch_list)
                if self.is_train:
                    cpu_fetch_list.append(g)
                cpu_res = cpu_exe.run(
                    main_program,
                    feed=feed_dict,
                    fetch_list=cpu_fetch_list,
                    return_numpy=True,
                )

                # get gcu result
                gcu_exe = paddle.static.Executor('gcu')
                gcu_res = gcu_exe.run(
                    main_program,
                    feed=feed_dict,
                    fetch_list=fetch_list,
                    return_numpy=True,
                )
                logging.info(
                    'result number: '
                    + str(len(fetch_list))
                    + ", result names: "
                    + str(fetch_list)
                )
                logging.info('[cpu result]')
                logging.info(cpu_res)
                logging.info('[gcu result]')
                logging.info(gcu_res)
                # print("gcu result length:", len(gcu_res))
                # print("cpu result length:", len(cpu_res))
                print("gcu result shape:")
                for x in gcu_res:
                    print(x.shape)
                    # print(x)
                print("cpu result shape:")
                for x in cpu_res:
                    print(x.shape)
                    # print(x)

                # verify the result
                self.compare(cpu_res, gcu_res)

                logging.info(f'cpu_fetch_list: {cpu_fetch_list}')
                logging.info(f'gcu_fetch_list: {fetch_list}')


# relu = TestBase(func=paddle.nn.ReLU, feed_dict={'name':["data"], 'shape':[[3]], 'dtype':['float32']}, is_train=False)
#
# data = np.array([-1, 0, 1], dtype=np.float32)
# relu.run(feed=[data])
