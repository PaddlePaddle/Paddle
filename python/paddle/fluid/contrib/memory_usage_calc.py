#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
"""
This module privides a memory usage calculate function for user.
The purpose of this API is to allow users to estimate memory usage of
a program under a special batch size, then user can set appropriate 
batch size to fully utilize a GPU. 

This API is still under active development and may change drastically.
"""

from .. import core
from ..framework import Program, Variable

__all__ = ['MemoryInfo']

DEBUG = False

dtype_to_size = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    core.VarDesc.VarType.UINT8: 1,
}


class MemoryInfo(object):
    def __init__(self, program):
        if not isinstance(program, Program):
            raise TypeError(
                "Calculating Memory Usage requires Program as its Parameter."
                "But you passed in %s" % (type(prgram)))
        self._program = program

    def _has_var(self, block, var_name):
        return block.has_var(str(var_name))

    def _find_var(self, block, var_name):
        return block.var(str(var_name))

    def get_memory_usage(self, batch_size, with_details=False):

        # get the first block of program
        first_block = self._program.global_block()

        # get the var_name list of first block
        # TODO(chenweihang): not find the API get block's var list directly
        total_memory = 0.0
        for var in self._program.list_vars():
            if DEBUG:
                print "All Block's Var: %s" % (var.name)
            # TODO(chenweihang): why not used program.list_vars() 
            #                       calculate all variable's memory directly?
            if self._has_var(first_block, var.name):
                if DEBUG:
                    print "First Block's Var: %s" % (var.name)
                    print "Var's shape: ", var.shape
                    print "Var's dtype: ", var.dtype
                data_count = 1
                for x in var.shape:
                    if x == -1:
                        data_count *= batch_size
                    else:
                        data_count *= x
                var_memory = data_count * dtype_to_size[var.dtype]
                if DEBUG:
                    print "Var's memory: %d" % (var_memory)
                total_memory += var_memory

        # Convert unit and make result string
        result_str = "- With current batch size, memory usage is about "
        unit_str = " B."
        if total_memory > 1024:
            total_memory /= 1024
            unit_str = " KB."
            if total_memory > 1024:
                total_memory /= 1024
                unit_str = " MB."

        # Append extra memory consumption (5% - 10%)
        result_str += str(round(total_memory * 1.05, 3)) + " - " \
                    + str(round(total_memory * 1.10, 3)) + unit_str

        return result_str
