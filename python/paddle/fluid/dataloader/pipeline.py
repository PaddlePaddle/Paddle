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

from __future__ import print_function

import paddle
import paddle.fluid as fluid

from paddle import _C_ops
from paddle.fluid import core, framework
from paddle.fluid.layers.utils import _hash_with_id
from ..multiprocess_utils import CleanupFuncRegistrar

from collections.abc import Sequence, Mapping

__all__ = ["Pipeline"]


CleanupFuncRegistrar.register(core._shutdown_all_dataloaders)


AVAILABLE_OP_TYPES = ['data_reader', 'map']


class Pipeline:
    """
    Data pipeline

    Args:
        queue_depth(int): queue depth for caching data between OPs
    """

    def __init__(self, queue_depth=2):
        assert isinstance(queue_depth, int), \
                "queue_depth should be an integer"
        self._queue_depth = queue_depth
        self._init_programs()

        self.is_shutdown = False

    def _init_programs(self):
        self._main_program = fluid.Program()
        self._startup_program = fluid.Program()
        self._out_vars = []
        self._out_names = []
        self._is_built = False

    def __enter__(self):
        # switch main and startup program
        paddle.enable_static()
        self._main_program = framework.switch_main_program(self._main_program)
        self._startup_program = framework.switch_startup_program(
            self._startup_program)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._main_program = framework.switch_main_program(self._main_program)
        self._startup_program = framework.switch_startup_program(
            self._startup_program)
            
        local_rank = paddle.distributed.get_rank()
        paddle.disable_static("gpu:" + str(local_rank))

        self._check_op_type()

    def _check_op_type(self):
        for op in self._main_program.block(0).ops:
            if op.type not in ['data_reader', 'map']:
                raise RuntimeError(
                    "pipeline given to DataLoader.from_pipeline should be "
                    "composed of reader OPs and map OP, other OPs(e.g. "
                    "decoder OPs or Paddle OPs) should be run under "
                    "paddle.io.map")

    def set_outputs(self, outputs):
        if isinstance(outputs, Sequence):
            for var in outputs:
                self._out_vars.append(output)
        elif isinstance(outputs, Mapping):
            for name, var in outputs.items():
                self._out_vars.append(var)
                self._out_names.append(name)
        else:
            assert isinstance(outputs, fluid.Variable), \
                    "outputs should be list, dict or Variable"

    def build(self):
        global_block = self._main_program.desc.block(0)
        program_id = _hash_with_id(self._main_program, self)

        self._attrs = ('global_block', global_block, 'start_op_index', 0,
                       'end_op_index', global_block.op_size(), 'program_id',
                       program_id)
        self._is_built = True

    def _prepare_output_vars(self):
        output_vars = []
        for var in self._out_vars:
            if isinstance(var, (list, tuple)):
                var = var[0]
            assert isinstance(var, framework.Variable), \
                    "output of DataLoader program should be Variable"
            var_desc = var.desc
            output_var = core.VarBase(var_desc.dtype(),
                                      var_desc.shape(),
                                      var_desc.name(), var_desc.type(), False)
            output_vars.append(output_var)

        return output_vars

    def __iter__(self):
        return self

    def __next__(self):
        assert self._is_built, \
                "Pipeline not built, please call build() firstly"
        self._output_vars = self._prepare_output_vars()

        try:
            import sys
            import time
            tic = time.time()
            _C_ops.dataloader(self._output_vars, *self._attrs)
            toc = time.time()
            print("_C_ops calling cost {}ms".format((toc - tic) * 1000.))
            sys.stdout.flush()
        except:
            raise StopIteration

        return {k: v for k, v in zip(self._out_names, self._output_vars)}

    # Python 2 compatable
    def next(self):
        return self.__next__()

    def shutdown(self):
        if not self.is_shutdown:
            try:
                program_id = _hash_with_id(self._main_program)
                core._shutdown_readers_and_decoders(program_id)

                map_program_ids = []
                for op in self._main_program.block(0).ops:
                    if op.type == "map":
                        map_program_ids.append(op.attrs['program_id'])
                core._shutdown_maps(program_id)

                core._shutdown_pipeline(program_id)
            finally:
                self.is_shutdown = True

    def __del__(self):
        self.shutdown()
