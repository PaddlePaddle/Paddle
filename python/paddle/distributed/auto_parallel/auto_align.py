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

import copy
import os
import pickle

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.converter import Converter
from paddle.distributed.auto_parallel.dist_context import (
    DistributedContext,
    get_default_distributed_context,
)
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
    def __init__(
        self, program: Program, step=1, fetch_list: DistributedContext = None
    ):
        assert isinstance(program, Program)
        self._program = program
        self._blocks = program.blocks
        self._step = step
        self._fetch_list = fetch_list
        assert self._blocks is not None

    def set_step(self, step):
        self._step = step

    def get_var(self, level, step):
        """
        level in [0,1,2,3,4,5]
        """
        if step < self._step:
            return self._fetch_list
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
        self._blocks = program.blocks
        assert self._blocks is not None

    def get_loss_lr_var(self):
        fetch_set = set()
        loss_ops = []
        for block in self._blocks:
            for op in block.ops:
                if is_loss_op(op):
                    assert (
                        len(op.desc.output_arg_names()) == 1
                    ), "loss op should only output loss var"
                    loss_ops.append(op)

        for block in self._blocks:
            for varname in block.vars:
                var = block._find_var_recursive(varname)

                if var is None or var.type not in _valid_types:
                    continue

                if "learning_rate" in var.name:
                    fetch_set.add(var.name)

        assert len(loss_ops) == 1, "num of loss op is not equal to one"

        fetch_set.add(loss_ops[0].output_arg_names[0])

        return list(fetch_set)

    def get_data_var(self):
        fetch_set = set()
        for block in self._blocks:
            for varname in block.vars:
                var = block._find_var_recursive(varname)

                if var is None or var.type not in _valid_types:
                    continue

                if var.is_data:
                    fetch_set.add(var.name)
        return list(fetch_set)

    def get_param_var(self):
        fetch_set = set()
        for block in self._blocks:
            for op in block.ops:
                if is_backward_op(op):
                    break
                for varname in op.input_arg_names + op.output_arg_names:
                    var = block._find_var_recursive(varname)
                    if var is None or var.type not in _valid_types:
                        continue
                    if var.is_parameter:
                        fetch_set.add(varname)

        return list(fetch_set)

    def get_param_grad_var(self):
        fetch_set = set()
        for block in self._blocks:
            for op in block.ops:
                if is_forward_op(op):
                    continue
                for varname in op.input_arg_names + op.output_arg_names:
                    if "@GRAD" not in varname:
                        continue
                    fwd_varname = varname.split("@GRAD")[0]
                    fwd_var = block._find_var_recursive(fwd_varname)
                    if fwd_var is None or fwd_var.type not in _valid_types:
                        continue
                    if fwd_var.is_parameter is False:
                        continue
                    var = block._find_var_recursive(varname)
                    if var is None or var.type not in _valid_types:
                        continue
                    fetch_set.add(varname)

        return list(fetch_set)

    def get_forward_tmp_var(self):
        fetch_set = set()
        loss_lr_list = self.get_loss_lr_var()
        for block in self._blocks:
            for op in block.ops:
                if is_backward_op(op):
                    break
                for varname in op.input_arg_names + op.output_arg_names:
                    if varname in loss_lr_list:
                        continue
                    var = block._find_var_recursive(varname)
                    if var is None or var.type not in _valid_types:
                        continue
                    if var.is_data or var.is_parameter:
                        continue
                    fetch_set.add(varname)

        return list(fetch_set)

    def get_backward_tmp_var(self):
        fetch_set = set()
        loss_lr_list = self.get_loss_lr_var()
        for block in self._blocks:
            for op in block.ops:
                if is_backward_op(op) is False:
                    continue
                for varname in op.input_arg_names + op.output_arg_names:
                    if varname in loss_lr_list:
                        continue
                    if "@GRAD" in varname:
                        fwd_varname = varname.split("@GRAD")[0]
                        fwd_var = block._find_var_recursive(fwd_varname)
                        if fwd_var is not None and fwd_var.type in _valid_types:
                            if fwd_var.is_parameter:
                                continue
                    var = block._find_var_recursive(varname)
                    if var is None or var.type not in _valid_types:
                        continue
                    if var.is_data or var.is_parameter:
                        continue
                    fetch_set.add(varname)

        return list(fetch_set)

    def save(self, save_dir, vars, fetch_list, dist_context=None):
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)
        if dist_context is None:
            dist_context = get_default_distributed_context()
        assert os.path.exists(save_dir)
        if dist.get_world_size() == 1:
            vars_path = os.path.join(save_dir, "vars.pkl")
            program_path = os.path.join(save_dir, "program.pdmodel")
            dist_attr_path = os.path.join(save_dir, "dist_attr.pkl")
        else:
            vars_path = os.path.join(
                save_dir, "vars_rank{}.pkl".format(dist.get_rank())
            )
            program_path = os.path.join(
                save_dir, "program_rank{}.pdmodel".format(dist.get_rank())
            )
            dist_attr_path = os.path.join(
                save_dir, "dist_attr_rank{}.pkl".format(dist.get_rank())
            )
        if vars is not None:
            vars_dict = dict()
            assert len(fetch_list) == len(vars)
            for i in range(len(fetch_list)):
                if vars[i] is None:
                    continue
                vars_dict[fetch_list[i]] = vars[i]
            pickle.dump(vars_dict, open(vars_path, "wb"))
            dist_attr = {}
            for var in self._program.list_vars():
                if var.name not in fetch_list:
                    continue
                tensor_dist_attr = (
                    dist_context.get_tensor_dist_attr_for_program(var)
                )
                if tensor_dist_attr is None:
                    continue
                process_mesh = tensor_dist_attr.process_mesh
                dims_mapping = tensor_dist_attr.dims_mapping
                dist_attr[var.name] = {
                    "process_shape": process_mesh.topology,
                    "process_group": process_mesh.processes,
                    "dims_mapping": dims_mapping,
                }
            if len(dist_attr) > 0:
                pickle.dump(dist_attr, open(dist_attr_path, "wb"))
        if self._program is not None:
            paddle.save(self._program, program_path)

    @staticmethod
    def load(save_dir):
        assert os.path.exists(save_dir)
        filename_list = os.listdir(save_dir)
        vars_list = []
        program_list = []
        dist_attr_list = []
        for filename in filename_list:
            filepath = os.path.join(save_dir, filename)
            assert os.path.isfile(filepath)
            if "vars" in filename:
                assert filename.endswith("pkl")
                vars_list.append(pickle.load(open(filepath, "rb")))
            elif "program" in filename:
                assert filename.endswith("pdmodel")
                program_list.append(paddle.load(filepath))
            elif "dist_attr" in filename:
                assert filename.endswith("pkl")
                dist_attr_list.append(pickle.load(open(filepath, "rb")))

        dist_attr_map = dict()
        for dist_attrs in dist_attr_list:
            for dist_attr_name in dist_attrs.keys():
                if dist_attr_name not in dist_attr_map:
                    dist_attr_map[dist_attr_name] = dist_attrs[dist_attr_name]
                else:
                    assert (
                        dist_attr_map[dist_attr_name]
                        == dist_attrs[dist_attr_name]
                    )
        assert len(vars_list) == len(program_list)
        return vars_list, program_list, dist_attr_map

    @staticmethod
    def convert_dist_tensor_2_serial_tensor(vars_list, dist_attr_map):
        assert len(vars_list) >= 1
        if dist_attr_map is None or len(dist_attr_map) == 0:
            return vars_list[0]

        complete_strategys = dict()
        dist_strategys = dict()
        tensors_dict = dict()

        convert_tensor_dict = None
        for var_name in dist_attr_map.keys():
            assert var_name not in tensors_dict
            assert var_name not in complete_strategys
            dist_vars = []
            for vars in vars_list:
                if var_name in vars.keys():
                    dist_vars.append(vars[var_name])
            if len(dist_vars) == 0:
                continue

            tensors_dict[var_name] = dist_vars
            complete_strategys[var_name] = copy.deepcopy(
                dist_attr_map[var_name]
            )
            dist_strategys[var_name] = copy.deepcopy(dist_attr_map[var_name])
            for i in range(len(complete_strategys[var_name]["dims_mapping"])):
                complete_strategys[var_name]["dims_mapping"][i] = -1

            converter = Converter(
                tensors_dict, dist_strategys, complete_strategys
            )
            convert_tensor_dict = converter.convert()
        for vars in vars_list:
            for var_name in vars.keys():
                if var_name not in convert_tensor_dict:
                    convert_tensor_dict[var_name] = vars[var_name]

        return convert_tensor_dict

    @staticmethod
    def find_diff_vars(vars_map1, vars_map2):
        diff_var_name_list = []
        for var_name1 in vars_map1.keys():
            if var_name1 in vars_map2:
                if not np.allclose(vars_map1[var_name1], vars_map2[var_name1]):
                    diff_var_name_list.append(var_name1)
        return diff_var_name_list

    @staticmethod
    def diff_informations(save_dir1, save_dir2):
        vars_list1, program_list1, dist_attr_map1 = AutoAlign.load(save_dir1)
        vars_list2, program_list2, dist_attr_map2 = AutoAlign.load(save_dir2)
        tensors_dict1 = AutoAlign.convert_dist_tensor_2_serial_tensor(
            vars_list1, dist_attr_map1
        )
        tensors_dict2 = AutoAlign.convert_dist_tensor_2_serial_tensor(
            vars_list2, dist_attr_map2
        )
        diff_var_name_list = AutoAlign.find_diff_vars(
            tensors_dict1, tensors_dict2
        )

        diff_ops_varname_dict = dict()

        for program1 in program_list1:
            for block1 in program1.blocks:
                for op1 in block1.ops:
                    for varname in op1.input_arg_names + op1.output_arg_names:
                        if varname in diff_var_name_list:
                            print(
                                op1, ";different varname is:{}".format(varname)
                            )
                            if op1 not in diff_ops_varname_dict:
                                diff_ops_varname_dict[op1] = [varname]
                            else:
                                diff_ops_varname_dict[op1].append(varname)

        return diff_ops_varname_dict


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
                fetch_list = align_tool.get_var(0, 1)
                fetch_list = align_tool.get_var(1, 1)
                fetch_list = align_tool.get_var(2, 1)
                fetch_list = align_tool.get_var(3, 1)
                fetch_list = align_tool.get_var(4, 1)
                fetch_list = align_tool.get_var(5, 1)
                vars = executor.run(
                    main_program,
                    feed={"image": images, "label": labels},
                    fetch_list=fetch_list,
                )
                align_tool.save(
                    "/workspace/Paddle/save_dir/serial", vars, fetch_list
                )
                ans = align_tool.load("/workspace/Paddle/save_dir/serial")
                print()
