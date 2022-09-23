#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import random
import unittest
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np
import paddle
import paddle.static

map_np_dtype_to_fluid_dtype = {
    'bool': "bool",
    'int8': "int8",
    'uint8': "uint8",
    "int32": "int32",
    "int64": "int64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


def np_dtype_to_fluid_str(dtype: np.dtype) -> str:
    return map_np_dtype_to_fluid_dtype[dtype.name]


class ExecutionModeFull(IntEnum):
    # Run fp32 model on cpu
    CPU_FP32 = 1
    # Run fp32 model on ipu
    IPU_FP32 = 2
    # Convert model to fp16 using mixed-precision approch
    # All parameters will be converted to fp16
    IPU_FP16 = 3


class ExecutionMode(IntEnum):
    CPU_FP32 = ExecutionModeFull.CPU_FP32
    IPU_FP32 = ExecutionModeFull.IPU_FP32
    IPU_FP16 = ExecutionModeFull.IPU_FP16


class IPUTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get random seeds
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()

        cls.SEED = 2021
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        paddle.seed(cls.SEED)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    # Check if ipumodel mode is enabled
    @classmethod
    def use_ipumodel(cls):
        if 'POPLAR_IPUMODEL' not in os.environ:
            return False
        else:
            flag = os.environ['POPLAR_IPUMODEL']
            if flag.upper() in ['1', "TRUE"]:
                return True


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class IPUD2STest(IPUTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Disable paddle static graph mode
        paddle.disable_static()

    def tearDown(self):
        # Manual reset when using ipumodel
        if self.use_ipumodel():
            paddle.framework.core.IpuBackend.get_instance().reset()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class IPUOpTest(IPUTest):
    """Base Class for single op unit tests using static graph on IPU.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Enable paddle static graph mode
        paddle.enable_static()

        # Items that a op_tester needs
        cls.main_prog: paddle.static.Program = None
        cls.startup_prog: paddle.static.Program = None
        cls.scope: paddle.static.Scope = None
        cls.feed_list: List[str] = None
        cls.fetch_list: List[str] = None
        cls.output_dict: Optional[Dict] = {}

    def tearDown(self):
        # Manual reset when using ipumodel
        if self.use_ipumodel():
            paddle.framework.core.IpuBackend.get_instance().reset()

    @property
    def fp16_enabled(self):
        return True

    def skip_mode(self, exec_mode):
        if exec_mode > ExecutionMode.IPU_FP32 and not self.fp16_enabled:
            return True
        else:
            return False

    def is_ipu_mode(self, exec_mode):
        if exec_mode == ExecutionMode.CPU_FP32:
            return False
        return True

    def is_fp16_mode(self, exec_mode):
        if exec_mode != ExecutionMode.IPU_FP16:
            return False
        return True

    def set_atol(self):
        self.atol = 1e-10
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_training(self):
        self.is_training = False
        self.epoch = 1

    # Decorator for static graph building
    def static_graph(builder):

        def wrapper(self, *args, **kwargs):
            self.scope = paddle.static.Scope()
            self.main_prog = paddle.static.Program()
            self.startup_prog = paddle.static.Program()
            self.main_prog.random_seed = self.SEED
            self.startup_prog.random_seed = self.SEED
            with paddle.static.scope_guard(self.scope):
                with paddle.utils.unique_name.guard(
                        paddle.utils.unique_name.generate('')):
                    with paddle.static.program_guard(self.main_prog,
                                                     self.startup_prog):
                        builder(self, *args, **kwargs)

        return wrapper

    # Cast a fp32 model to a full-fp16 model
    @classmethod
    def cast_model_to_fp16(cls, main_program):
        amp_list = paddle.static.amp.CustomOpLists()
        amp_list.unsupported_list = {'scale'}
        to_fp16_var_names = paddle.static.amp.cast_model_to_fp16(
            main_program, amp_list, use_fp16_guard=False)
        paddle.static.amp.cast_parameters_to_fp16(
            paddle.CPUPlace(),
            main_program,
            to_fp16_var_names=to_fp16_var_names)

    def run_op_test(self, exec_mode, ipu_strategy=None):
        # NOTE: some op has no inputs
        # if len(self.feed_list) == 0 or len(self.fetch_list) == 0:
        #     raise ValueError('feed_list or fetch_list is empty')
        if self.is_ipu_mode(exec_mode):
            place = paddle.IPUPlace()
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(self.startup_prog)
        if self.is_ipu_mode(exec_mode):
            if ipu_strategy is None:
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=self.is_training)
            if self.is_fp16_mode(exec_mode):
                ipu_strategy.set_precision_config(enable_fp16=True)
                IPUOpTest.cast_model_to_fp16(self.main_prog)

            # TODO(ipu) remove in the future version of popart
            # keep the log clean, no side effects for tests without profiling
            ipu_strategy.set_options(
                {'engine_options': {
                    'debug.retainDebugInformation': 'false'
                }})

            program = paddle.static.IpuCompiledProgram(
                self.main_prog,
                ipu_strategy=ipu_strategy).compile(self.feed_list,
                                                   self.fetch_list)
        else:
            program = self.main_prog

        feed = self.feed_fp32
        if self.is_fp16_mode(exec_mode):
            feed = self.feed_fp16

        if self.is_training:
            result = []
            for _ in range(self.epoch):
                loss_res = exe.run(program,
                                   feed=feed,
                                   fetch_list=self.fetch_list)
                result.append(loss_res)
        else:
            result = exe.run(program, feed=feed, fetch_list=self.fetch_list)

        if isinstance(result, list) and len(result) == 1:
            self.output_dict[exec_mode] = result[0]
        else:
            self.output_dict[exec_mode] = result

    def check(self, check_shape=False, output_dict=None):
        if output_dict is None:
            output_dict = self.output_dict
        if len(output_dict) == 0:
            raise ValueError("output_dict is empty")
        cpu_fp32 = output_dict[ExecutionMode.CPU_FP32]
        ipu_fp32 = output_dict[ExecutionMode.IPU_FP32]
        if len(cpu_fp32) != len(ipu_fp32):
            raise ValueError("different outputs number between ipu and cpu.")
        for cpu_fp32_res, ipu_fp32_res in zip(cpu_fp32, ipu_fp32):
            cpu_fp32_res = np.asarray(cpu_fp32_res).astype(np.float32).flatten()
            ipu_fp32_res = np.asarray(ipu_fp32_res).astype(np.float32).flatten()
            pass_check = np.allclose(ipu_fp32_res,
                                     cpu_fp32_res,
                                     rtol=self.rtol,
                                     atol=self.atol)
            if not pass_check:
                max_atol = np.abs(ipu_fp32_res - cpu_fp32_res).max()
                cpu_fp32_abs = np.abs(cpu_fp32_res)
                cpu_fp32_abs[cpu_fp32_abs == 0.0] = 1e-20
                max_rtol = (np.abs(ipu_fp32_res - cpu_fp32_res) /
                            cpu_fp32_abs).max()
                raise AssertionError(
                    f"ipu_fp32 check failed. max_atol is {max_atol}, max_rtol is {max_rtol}"
                )

            if check_shape:
                self.assertTrue(cpu_fp32_res.shape == ipu_fp32_res.shape)

        if ExecutionMode.IPU_FP16 in output_dict.keys():
            ipu_fp16 = output_dict[ExecutionMode.IPU_FP16]
            if len(cpu_fp32) != len(ipu_fp16):
                raise ValueError(
                    "different outputs number between ipu and cpu.")
            for cpu_fp32_res, ipu_fp16_res in zip(cpu_fp32, ipu_fp16):
                cpu_fp32_res = np.asarray(cpu_fp32_res).astype(
                    np.float32).flatten()
                ipu_fp16_res = np.asarray(ipu_fp16_res).astype(
                    np.float32).flatten()
                pass_check = np.allclose(ipu_fp16_res,
                                         cpu_fp32_res,
                                         rtol=self.rtol_fp16,
                                         atol=self.atol_fp16)
                if not pass_check:
                    max_atol = np.abs(ipu_fp16_res - cpu_fp32_res).max()
                    cpu_fp32_abs = np.abs(cpu_fp32_res)
                    cpu_fp32_abs[cpu_fp32_abs == 0.0] = 1e-20
                    max_rtol = (np.abs(ipu_fp16_res - cpu_fp32_res) /
                                cpu_fp32_abs).max()
                    raise AssertionError(
                        f"ipu_fp16 check failed. max_atol is {max_atol}, max_rtol is {max_rtol}"
                    )

                if check_shape:
                    self.assertTrue(ipu_fp16_res.shape == cpu_fp32_res.shape)

    # Execution Mode
    class ExecutionMode(IntEnum):
        CPU_FP32 = ExecutionModeFull.CPU_FP32
        IPU_FP32 = ExecutionModeFull.IPU_FP32
        IPU_FP16 = ExecutionModeFull.IPU_FP16
