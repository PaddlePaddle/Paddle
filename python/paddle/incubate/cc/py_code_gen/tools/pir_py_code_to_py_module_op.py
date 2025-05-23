# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import glob as glob
import hashlib
import itertools
from collections import defaultdict

import paddle
from paddle.incubate.cc.py_code_gen.generators.module_op_py_code_generator import (
    ModuleOpPyCodeGenerator,
)
from paddle.incubate.cc.py_code_gen.ir import ir_op, ir_type
from paddle.incubate.cc.py_code_gen.util.load_pir_py_classes import (
    GetProgramClasses,
)
from paddle.incubate.cc.py_code_gen.util.primitive_op_extractor import (
    PrimitiveOpExtractor,
)


def TranslatePirPyCodeToPyModuleOp(pir_py_code_file_path):
    seg_counter = defaultdict(lambda: itertools.count())
    for py_code in GetOutputOpPyCode(pir_py_code_file_path):
        return py_code
    return None


def GetSha256sum(content):
    m = hashlib.sha256()
    m.update(paddle.__git_commit__.encode())
    m.update(content.encode())
    return m.hexdigest()


def IsBackwardProgram(ir_program):
    for name, op in vars(ir_program).items():
        if not isinstance(op, ir_op.Op):
            continue
        if op.name != "builtin.module":
            continue
        keyword_arg_names = op.block_keyword_arg_names[0][0]
        if len(keyword_arg_names) > 0:
            return True
    return False


def GetOutputOpPyCode(original_programs_file):

    yield from (
        py_code
        for cls in GetProgramClasses(original_programs_file)
        for ir_program in [cls()]
        if not IsBackwardProgram(ir_program)
        if AllInputOutputTypesSupported(ir_program)
        for generator in [ModuleOpPyCodeGenerator(ir_program)]
        for op_names in [GetOpNames(ir_program)]
        for py_code in [generator.Generate()]
    )


def GetOpNames(ir_program):
    primitive_op_extractor = PrimitiveOpExtractor()
    return [op.name for op in primitive_op_extractor.Extract(ir_program)]


def AllInputOutputTypesSupported(ir_program):
    supported_operand_types = (
        ir_type.DenseTensorType,
        ir_type.NullType,
        ir_type.VectorType,
    )
    primitive_op_extractor = PrimitiveOpExtractor()
    return all(
        isinstance(in_out_type, supported_operand_types)
        for op in primitive_op_extractor.Extract(ir_program)
        for in_out_type in op.input_types + op.output_types
    )
