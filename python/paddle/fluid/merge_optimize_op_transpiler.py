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


def fuse_optimize_op(input_program):
    global_block = input_program.global_block()
    op_size = len(global_block.ops)
    sgd_op_idxs = list()
    for op_idx in range(op_size):
        if global_block.ops[op_idx].type == "sgd":
            sgd_op_idxs.append(op_idx)

    param_inputs = list()
    grad_inputs = list()
    lr_inputs = list()
    param_outputs = list()

    # TODO should check if these optimize op have no dependency with each other
    for op_idx in sgd_op_idxs:
        sgd_op = global_block.ops[op_idx]
        param_inputs.append(sgd_op.input("Param")[0])
        grad_inputs.append(sgd_op.input("Grad")[0])
        lr_inputs.append(sgd_op.input("LearningRate")[0])
        param_outputs.append(sgd_op.output("ParamOut")[0])

    global_block.append_op(
        type="sgd_group",
        inputs={
            "Params": param_inputs,
            "Grads": grad_inputs,
            "LearningRates": lr_inputs
        },
        outputs={"ParamOuts": param_inputs})

    for op_idx in sgd_op_idxs[::-1]:
        global_block.delete_op(op_idx)
