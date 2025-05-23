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

import os
from collections import namedtuple

from jinja2 import Template

from paddle.incubate.cc.py_code_gen.generators.ap_variadic_op_call_generator import (
    ApVariadicOpCallGenerator,
)
from paddle.incubate.cc.py_code_gen.generators.block_name_generator import (
    BlockNameGenerator,
)
from paddle.incubate.cc.py_code_gen.generators.blocks_generator import (
    BlocksGenerator,
)
from paddle.incubate.cc.py_code_gen.generators.paddle_block_stmts_generator import (
    PaddleBlockStmtsGenerator,
)
from paddle.incubate.cc.py_code_gen.ir_converters.paddle_tensor_converter import (
    ConvertToPaddleTensor,
)

BlockDescriptor = namedtuple(
    "BlockDescriptor",
    [
        "is_entry_block",
        "block_name",
        "input_arg_names",
        "stmts",
        "output_arg_names",
    ],
)


class ModuleOpPyCodeGenerator:

    def __init__(self, ir_program):
        self.name = type(ir_program).__name__
        self.program_id = int(self.name[len("PirProgram_") :])
        self.blocks_generator = BlocksGenerator(ir_program)
        self.block_name_gen = BlockNameGenerator(use_local_name=True)
        self.literal_str2local_id = {}
        self.stmts_gen = PaddleBlockStmtsGenerator(
            MakeOpName2CustomOpCallGenerator(self.literal_str2local_id),
            self.block_name_gen,
        )

    def Generate(self):

        def MakeBlockDescriptor(block):
            (
                input_local_tensors,
                stmts,
                output_local_tensors,
            ) = self.stmts_gen.Generate(block)
            input_local_tensors = [
                ConvertToPaddleTensor(t) for t in input_local_tensors
            ]
            return BlockDescriptor(
                is_entry_block=block.is_entry_block,
                block_name=self.block_name_gen.Generate(
                    block.owner_op, block.region_idx, block.block_idx
                ),
                input_arg_names=[tensor.name for tensor in input_local_tensors],
                stmts=stmts,
                output_arg_names=[
                    tensor.name for tensor in output_local_tensors
                ],
            )

        blocks = [
            MakeBlockDescriptor(block)
            for block in self.blocks_generator.Generate()
        ]
        return self._RenderTemplate(blocks=blocks)

    def _RenderTemplate(self, blocks):
        template = self._GetTemplate("template_module_op_py_code.jinja")
        ap_workspace_dir = os.environ["AP_WORKSPACE_DIR"]
        return template.render(
            blocks=blocks,
            tensor_name_converter=lambda x: x,
            lambda_str_attr_getter_name=kLambdaStrAttrGetterName,
            axpr_lambda_json_str_and_local_ids=list(
                self.literal_str2local_id.items()
            ),
            ap_workspace_dir=repr(ap_workspace_dir),
        )

    def _GetTemplate(self, template_name):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(f"{dir_path}/{template_name}", "r") as f:
            return Template(f.read())


def MakeOpName2CustomOpCallGenerator(literal_str2local_id):
    return {
        "pd_op.ap_variadic": ApVariadicOpCallGenerator(
            literal_str2local_id=literal_str2local_id,
            lambda_str_attr_getter_name=kLambdaStrAttrGetterName,
        )
    }


kLambdaStrAttrGetterName = 'get_lambda_str_registry_key'
