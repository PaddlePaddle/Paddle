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
import os

from paddle import pir
from paddle.base import core
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.base.framework import get_flags
from paddle.static.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

__all__ = []

MAX_FILE_PATH_LEN = 100


class SubGraphRole:
    Infer = 0
    Forward = 1
    Backward = 2


def get_saving_dir():
    flag = "FLAGS_pir_subgraph_saving_dir"
    value = get_flags(flag)[flag]
    return value


class BaseExporter:
    def __init__(self, partial_program_layer, program, role):
        self.pp_layer = partial_program_layer
        self.program = program
        self.role = role
        self.root_dir = get_saving_dir()
        self.fetch_col = 0

    def save(self):
        # step 1: Create subgraph saving path.
        saving_path = self.generate_saving_path()

        # step 2: Translate into pir program.
        pir_program = self.translate_into_pir()

        # step 3: save into local disk.
        self._save(pir_program, saving_path)

    def _save(self, pir_program, path):
        content = str(pir_program)
        with open(path, 'w') as f:
            f.write(content)
        _logger.info(f"Successfully save subgraph into {path}")

    def parse_inout(self):
        """
        Return feed/fetch/intermediate var name list.
        """
        raise NotImplementedError("Need to implement parse_inout method")

    def translate_into_pir(self):
        # step 1: Insert data op for inputs/params
        feed_list, fetch_list, inter_outs = self.parse_inout()
        self.insert_feed_op(feed_list, "pt_input_")
        # step 2: Insert fetch op for outputs
        self.insert_fetch_op(fetch_list, "pt_output_")
        self.insert_fetch_op(inter_outs, "pt_intermediate_")
        # step 3: translate into pir
        pir_program = pir.translate_to_pir(self.program.desc)
        return pir_program

    def generate_saving_path(self):
        layer_name = self.pp_layer._debug_name
        assert layer_name is not None
        ops_name = [
            op.type for op in self.program.block(0).ops[:MAX_FILE_PATH_LEN]
        ]
        prefix = ["infer_", "forward_", "backward_"][self.role]
        file_name = prefix + "_".join(ops_name)[:MAX_FILE_PATH_LEN] + '.txt'
        saving_dir = os.path.join(self.root_dir, layer_name)
        self.verify_saving_dir(saving_dir)
        return os.path.join(self.root_dir, layer_name, file_name)

    def verify_saving_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def insert_feed_op(self, inputs, rename_prefix):
        global_block = self.program.block(0)
        inputs.sort()
        for i, old_name in enumerate(inputs):
            new_name = rename_prefix + str(i)
            global_block._rename_var(old_name, new_name)
            out = global_block.var(new_name)
            global_block._prepend_op(
                type='data',
                inputs={},
                outputs={'out': out},
                attrs={
                    'shape': out.shape,
                    'dtype': out.dtype,
                    'place': 0,
                    'name': out.name,
                },
            )
        global_block._sync_with_cpp()

    def insert_fetch_op(self, outputs, rename_prefix):
        global_block = self.program.block(0)
        fetch_var = global_block.create_var(
            name="fetch_outputs",
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=False,
        )
        outputs.sort()
        for i, old_name in enumerate(outputs):
            new_name = rename_prefix + str(i)
            global_block._rename_var(old_name, new_name)
            new_var = global_block.var(new_name)
            global_block.append_op(
                type="fetch",
                inputs={'X': [new_var]},
                outputs={'Out': [fetch_var]},
                attrs={'col': self.fetch_col},
            )
            self.fetch_col += 1
        global_block._sync_with_cpp()


class InferExporter(BaseExporter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_inout(self):
        inputs, outputs = [], []
        global_block = self.program.block(0)
        raw_inputs = self.pp_layer._inputs.tolist() + self.pp_layer._params
        raw_outputs = self.pp_layer._outputs.tolist()
        for var in raw_inputs:
            inputs.append(var.name)

        for var in raw_outputs:
            outputs.append(var.name)

        return inputs, outputs, []


class TrainFwdExporter(BaseExporter):
    def __init__(self, pp_layer, copy_program, role, raw_inter_outs):
        super().__init__(pp_layer, copy_program, role)
        self.raw_inter_outs = raw_inter_outs

    def parse_inout(self):
        inputs, outputs = [], []
        global_block = self.program.block(0)
        raw_inputs = self.pp_layer._inputs.tolist() + self.pp_layer._params
        raw_outputs = self.pp_layer._outputs.tolist()

        inter_outs = {
            name for name in self.raw_inter_outs if global_block.has_var(name)
        }
        for var in raw_inputs:
            inputs.append(var.name)
            if var.name in inter_outs:
                inter_outs.remove(var.name)

        for var in raw_outputs:
            outputs.append(var.name)
            if var.name in inter_outs:
                inter_outs.remove(var.name)

        return inputs, outputs, list(inter_outs)


class TrainBwdExporter(BaseExporter):
    def __init__(self, pp_layer, copy_program, role, raw_inputs, raw_outputs):
        super().__init__(pp_layer, copy_program, role)
        self.raw_inputs = raw_inputs
        self.raw_outputs = raw_outputs

    def parse_inout(self):
        inputs, outputs = [], []
        global_block = self.program.block(0)

        for var_name in self.raw_inputs:
            if global_block.has_var(var_name):
                inputs.append(var_name)

        # add fill_constant grad_var as input
        for var in self.pp_layer._outputs.tolist():
            init_grad_name = var.name + "@GRAD"
            if init_grad_name not in self.raw_inputs and global_block.has_var(
                init_grad_name
            ):
                inputs.append(init_grad_name)

        for var_name in self.raw_outputs:
            if (
                global_block.has_var(var_name)
                and var_name not in self.raw_inputs
            ):
                outputs.append(var_name)

        return inputs, outputs, []


@switch_to_static_graph
def pir_exporter(pp_layer, program, role, shared_inputs=None, inter_outs=None):
    # skip it if not specify root_saving_dir by FLAGS.
    root_saving_dir = get_saving_dir()
    if not root_saving_dir:
        return
    try:
        copy_program = program.clone()
        if role == SubGraphRole.Infer:
            InferExporter(pp_layer, copy_program, role).save()
        elif role == SubGraphRole.Forward:
            TrainFwdExporter(pp_layer, copy_program, role, inter_outs).save()
        elif role == SubGraphRole.Backward:
            TrainBwdExporter(
                pp_layer, copy_program, role, shared_inputs, inter_outs
            ).save()
        else:
            raise RuntimeError(
                f"role only support Infer/Forward/Backward, but got: {role}"
            )
    except Exception as e:
        _logger.error(
            f"Export subgraph failed: {e}\n. Received original program: {str(program)}"
        )
