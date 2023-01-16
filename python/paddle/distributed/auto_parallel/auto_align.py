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

import os
import pickle

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.utils import (
    is_backward_op,
    is_forward_op,
    is_loss_op,
)
from paddle.fluid import core
from paddle.fluid.framework import Program

_valid_types = [
    core.VarDesc.VarType.LOD_TENSOR,
    core.VarDesc.VarType.SELECTED_ROWS,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY,
]


class AutoAlign:
    def __init__(self, program: Program, step=1, fetch_list=None):
        assert isinstance(program, Program)
        self._program = program
        self._block = program.global_block()
        self._step = step
        self._fetch_list = fetch_list
        assert self._block is not None

    def set_step(self, step):
        self._step = step

    def get_var(self, level, step):
        if step < self._step:
            return self._fetch_list, None
        if level == 0:
            return self.get_loss_lr_var()
        elif level == 1:
            return self.get_data_var()
        elif level == 2:
            return self.get_param_var()
        elif level == 3:
            return self.get_param_grad_var()
        elif level == 4:
            return self.get_forward_tmp_var()
        elif level == 5:
            return self.get_backward_tmp_var()
        else:
            raise ValueError()

    def set_program(self, program: Program):
        assert isinstance(program, Program)
        self._program = program
        self._block = program.global_block()
        assert self._block is not None

    def get_data_var(self):
        fetch_set = set()
        for varname in self._block.vars:
            var = self._block._find_var_recursive(varname)

            if var is None or var.type not in _valid_types:
                continue

            if var.is_data:
                fetch_set.add(var.name)
        return list(fetch_set)

    def get_loss_lr_var(self):
        fetch_set = set()
        loss_ops = []
        for op in self._block.ops:
            if is_loss_op(op):
                assert (
                    len(op.desc.output_arg_names()) == 1
                ), "loss op should only output loss var"
                loss_ops.append(op)

        for varname in self._block.vars:
            var = self._block._find_var_recursive(varname)

            if var is None or var.type not in _valid_types:
                continue

            if "learning_rate" in var.name:
                fetch_set.add(var.name)

        assert len(loss_ops) == 1, "num of loss op is not equal to one"

        fetch_set.add(loss_ops[0].output_arg_names[0])

        return list(fetch_set)

    def save(self, step, save_dir, vars, varname_op_list):
        if step != self._step:
            return
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)

        assert os.path.exists(save_dir)
        if dist.get_world_size() == 1:
            vars_path = os.path.join(save_dir, "vars.pkl")
            varname_op_path = os.path.join(save_dir, "varname_op_list.pkl")
        else:
            vars_path = os.path.join(
                save_dir, "vars_rank{}.pkl".format(dist.get_rank())
            )
            varname_op_path = os.path.join(
                save_dir, "varname_op_list_rank{}.pkl".format(dist.get_rank())
            )

        if vars is not None:
            pickle.dump(vars, open(vars_path, "wb"))
        if varname_op_list is not None:
            pickle.dump(varname_op_list, open(varname_op_path, "wb"))

    def get_param_var(self):
        fetch_set = set()
        for op in self._block.ops:
            if is_backward_op(op):
                break
            for varname in op.input_arg_names + op.output_arg_names:
                var = self._block._find_var_recursive(varname)
                if var is None or var.type not in _valid_types:
                    continue
                if var.is_parameter:
                    fetch_set.add(varname)

        return list(fetch_set)

    def get_param_grad_var(self):
        fetch_set = set()
        for op in self._block.ops:
            if is_forward_op(op):
                continue

            for varname in op.input_arg_names + op.output_arg_names:
                if "@GRAD" not in varname:
                    continue
                fwd_varname = varname.split("@GRAD")[0]
                fwd_var = self._block._find_var_recursive(fwd_varname)
                if fwd_var is None or fwd_var.type not in _valid_types:
                    continue
                if fwd_var.is_parameter is False:
                    continue
                var = self._block._find_var_recursive(varname)
                if var is None or var.type not in _valid_types:
                    continue
                fetch_set.add(varname)

        return list(fetch_set)

    def get_forward_tmp_var(self):
        fetch_set = set()
        loss_lr_list = self.get_loss_lr_var()

        for op in self._block.ops:
            if is_backward_op(op):
                break

            for varname in op.input_arg_names + op.output_arg_names:
                if varname in loss_lr_list:
                    continue
                var = self._block._find_var_recursive(varname)
                if var is None or var.type not in _valid_types:
                    continue
                if var.is_data or var.is_parameter:
                    continue
                fetch_set.add(varname)

        return list(fetch_set)

    def get_backward_tmp_var(self):
        fetch_set = set()
        loss_lr_list = self.get_loss_lr_var()

        for op in self._block.ops:
            if is_backward_op(op) is False:
                continue

            for varname in op.input_arg_names + op.output_arg_names:
                if varname in loss_lr_list:
                    continue
                if "@GRAD" in varname:
                    fwd_varname = varname.split("@GRAD")[0]
                    fwd_var = self._block._find_var_recursive(fwd_varname)
                    if fwd_var is not None and fwd_var.type in _valid_types:
                        if fwd_var.is_parameter:
                            continue
                var = self._block._find_var_recursive(varname)
                if var is None or var.type not in _valid_types:
                    continue
                if var.is_data or var.is_parameter:
                    continue
                fetch_set.add(varname)

        return list(fetch_set)


if __name__ == "__main__":
    import warnings

    import numpy as np

    from paddle import fluid, nn, optimizer, static
    from paddle.vision.datasets import MNIST

    warnings.filterwarnings("ignore")
    paddle.enable_static()
    paddle.set_device("gpu")

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    class MnistDataset(MNIST):
        def __init__(self, mode, return_label=True):
            super().__init__(mode=mode)
            self.return_label = return_label

        def __getitem__(self, idx):
            img = np.reshape(self.images[idx], [1, 28, 28])
            if self.return_label:
                return img, np.array(self.labels[idx]).astype('int64')
            return (img,)

        def __len__(self):
            return len(self.images)

    dataset = MnistDataset("train")
    place = paddle.CUDAPlace(0)
    with fluid.program_guard(main_program, startup_program):
        inputs = static.data(
            name="image", shape=[256, 1, 28, 28], dtype="float32"
        )
        labels = static.data(name="label", shape=[256, 1], dtype="int64")
        z = nn.Conv2D(1, 6, 3, 1, 1).forward(inputs)
        z = nn.ReLU().forward(x=z)
        z = nn.MaxPool2D(2, 2).forward(x=z)
        z = nn.Conv2D(6, 16, 5, 1, 0).forward(x=z)
        z = nn.ReLU().forward(x=z)
        z = nn.MaxPool2D(2, 2).forward(x=z)
        z = nn.Flatten().forward(z)
        z = static.nn.fc(name="fc1", x=z, size=120)
        z = static.nn.fc(name="fc2", x=z, size=84)
        z = static.nn.fc(name="fc3", x=z, size=10)
        losses = nn.CrossEntropyLoss()(z, labels)

        optim = optimizer.SGD(0.001)
        optim.minimize(losses)

    executor = fluid.Executor()
    executor.run(startup_program)

    align_tool = AutoAlign(main_program, 1, [losses.name])

    for epoch in range(5):
        images = np.zeros([256, 1, 28, 28], np.float32)
        labels = np.zeros([256, 1], np.int64)
        for i, data in enumerate(dataset):
            images[i % 256] = data[0]
            labels[i % 256] = data[1]
            if i % 255 == 0 and i > 0:
                fetch_list = align_tool.get_var(5, 1)
                ans = executor.run(
                    main_program,
                    feed={"image": images, "label": labels},
                    fetch_list=fetch_list,
                )
                print()
