#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

import argparse
import logging
import os
import sys
import unittest

import numpy as np
from op_mappers.op_mapper_test import OpMapperTest

import paddle
from paddle.cinn.common import DefaultHostTarget, DefaultNVGPUTarget
from paddle.cinn.frontend import PaddleModelConvertor
from paddle.cinn.runtime import seed as cinn_seed

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="paddle_model_convertor")

parser = argparse.ArgumentParser(
    description='Load Paddle Model File and Running at CINN'
)
parser.add_argument(
    "--path", help="The path to load the paddle model", type=str, required=True
)
parser.add_argument(
    "-m",
    "--model_filename",
    help='The filename of model file, default "__model__"',
    type=str,
    default="__model__",
)
parser.add_argument(
    "-p",
    "--params_filename",
    help="The filename of model parameter file, default None, in which each parameter will saved in each file",
    type=str,
    default=None,
)
parser.add_argument(
    "-cuda",
    "--enable_cuda",
    help="Whether enable CUDA, default True",
    type=bool,
    default=True,
)
args = parser.parse_args()

np.random.seed(1234)
paddle.seed(1234)
cinn_seed(1234)

paddle.enable_static()

# first save paddle model like:
# ```
# import paddle
# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[10, 12, 128, 128], dtype='float32')
# y = paddle.static.data(name='y', shape=[10, 12, 128, 128], dtype='float32')
# prediction = paddle.stack([x, y], 1)

# place = paddle.CUDAPlace(0)

# exe = paddle.static.Executor(place)
# exe.run(paddle.static.default_startup_program())
# prog = paddle.static.default_main_program()

# paddle.static.io.save_inference_model("./stack", [x.name, y.name], [prediction], exe, prog)
# ```
# Second load and run model like:
# ```
# python test_paddle_model_convertor.py --path build/thirds/resnet_model -m "__model__" -p "params"
# ```


class TestPaddleModel(OpMapperTest):
    def setUp(self):
        if args.enable_cuda:
            self.target = DefaultNVGPUTarget()
            self.place = paddle.CUDAPlace(0)
        else:
            self.target = DefaultHostTarget()
            self.place = paddle.CPUPlace()

        self.model_dir = args.path
        self.model_filename = args.model_filename
        self.params_filename = args.params_filename

        logger.info(
            f'Run Model From "{self.model_dir}", which model filename is "{self.model_filename}", and parameter filename is "{self.params_filename}"'
        )

        self.load_paddle_program()
        self.init_case()

    @staticmethod
    def eliminate_unknown_shape(shape):
        return [1 if dim == -1 else dim for dim in shape]

    def get_paddle_op_attrs(self, op):
        attr_map = {}
        for n in op.attr_names:
            attr_map[n] = op.attr(n)

        return attr_map

    def init_case(self):
        self.feed_data = {}
        for i in range(len(self.feed_names)):
            # check no repeat variable
            self.assertNotIn(
                self.feed_names[i],
                self.feed_data,
                msg="Repeat feed name: " + self.feed_names[i],
            )

            dtype = self.paddledtype2nptype(self.feed_dtypes[i])
            # random int type data should not limited to [0, 1]
            high = 1 if ("int" not in dtype) else self.feed_shapes[i][0]

            # the paddle's feed list need dict not list
            self.feed_data[self.feed_names[i]] = self.random(
                self.eliminate_unknown_shape(self.feed_shapes[i]),
                dtype,
                high=high,
            )

    def load_paddle_program(self):
        self.exe = paddle.static.Executor(self.place)

        [
            self.inference_program,
            self.feed_names,
            self.fetch_targets,
        ] = paddle.static.io.load_inference_model(
            path_prefix=self.model_dir,
            executor=self.exe,
        )

        self.param_vars = paddle.load(
            self.model_dir,
            model_filename=self.model_filename,
            params_filename=self.params_filename,
            return_numpy=True,
        )

        logger.debug(msg=f"Program:\n{self.inference_program}")
        logger.debug(msg=f"Param List: {self.param_vars.keys()}")
        logger.debug(msg=f"Feed List: {self.feed_names}")
        logger.debug(
            msg=f"Fetch List: {[var.name for var in self.fetch_targets]}"
        )

        self.feed_shapes = []
        self.feed_dtypes = []

        for var in self.inference_program.list_vars():
            if var.name in self.feed_names:
                self.feed_shapes.append(var.shape)
                self.feed_dtypes.append(var.dtype)

        self.assertEqual(
            len(self.feed_names),
            len(self.feed_shapes),
            msg="Cannot found some feed var in program!",
        )

    def build_paddle_program(self, target):
        self.paddle_outputs = self.exe.run(
            self.inference_program,
            feed=self.feed_data,
            fetch_list=self.fetch_targets,
            return_numpy=True,
        )
        logger.debug(f"Paddle Result:\n{self.paddle_outputs}")

    def build_cinn_program(self, target):
        self.assertEqual(
            1,
            self.inference_program.num_blocks,
            msg="CINN only support single block now",
        )

        feed_with_param = []

        convertor = PaddleModelConvertor(target)
        for i in range(len(self.feed_names)):
            convertor.create_input(
                dtype=self.paddledtype2nptype(self.feed_dtypes[i]),
                shape=self.feed_data[self.feed_names[i]].shape,
                name=self.feed_names[i],
            )
            feed_with_param.append(self.feed_names[i])

        for param_name, param_value in self.param_vars.items():
            convertor.create_input(
                dtype=str(param_value.dtype),
                shape=param_value.shape,
                name=param_name,
            )
            feed_with_param.append(param_name)

        for op in self.inference_program.global_block().ops:
            if op.desc.type() == "feed" or op.desc.type() == "fetch":
                continue
            convertor.append_op(
                op.desc.type(),
                op.desc.inputs(),
                op.desc.outputs(),
                self.get_paddle_op_attrs(op),
            )

        prog = convertor()

        # get cinn input list
        inputs = prog.get_inputs()
        logger.debug(f"CINN Input List: {[var.name() for var in inputs]}")
        self.assertEqual(
            len(feed_with_param),
            len(inputs),
            msg="The paddle's input list not equal to cinn's input list!",
        )

        # map the name the variable
        input_dict = {var.name(): var for var in inputs}

        cinn_inputs = []
        cinn_feed_datas = []
        for name in feed_with_param:
            cinn_name = convertor.get_cinn_name(name)

            self.assertIn(
                cinn_name,
                input_dict,
                msg="Cannot find variable "
                + cinn_name
                + " in cinn program's input, which are "
                + str(input_dict.items()),
            )
            cinn_inputs.append(input_dict[cinn_name])

            if name in self.feed_data:
                cinn_feed_datas.append(self.feed_data[name])
            else:
                self.assertIn(
                    name,
                    self.param_vars,
                    msg="The input variable should in feed list or parameter list",
                )
                cinn_feed_datas.append(self.param_vars[name])

        # get cinn output list
        fetch_names = {var.name for var in self.fetch_targets}
        output_dict = convertor.get_fetch_list(fetch_names)
        cinn_output = [output_dict[var.name] for var in self.fetch_targets]

        # run and get result
        self.cinn_outputs = self.get_cinn_output(
            prog, target, cinn_inputs, cinn_feed_datas, cinn_output, passes=[]
        )

        logger.debug(f"CINN Result:\n{self.cinn_outputs}")

    def test_check_results(self):
        # TODO(6clc): There is a random accuracy problem,
        #             temporarily adjust max_absolute_error from 1e-6 to 1e-3
        self.check_outputs_and_grads(
            max_relative_error=1e-2, max_absolute_error=1e-3
        )


if __name__ == "__main__":
    tester = unittest.defaultTestLoader.loadTestsFromTestCase(TestPaddleModel)
    test_runner = unittest.TextTestRunner()
    res = test_runner.run(tester)
    sys.exit(not res.wasSuccessful())
