# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import defaultdict
from paddle.fluid.framework import Program, Block, Operator
from paddle.fluid.framework import in_dygraph_mode
import paddle.fluid.core as core


class PipelineInferHelper(object):
    """
    A helper class to split program for inference with pipeline parallelism.
    
    Args:
        startup_program (Program): the startup program.
        main_program (Program): the main program.
        stage (int): the current pipeline stage.
    
    Returns:
        None.
    """

    def __init__(self, startup_program, main_program, stage):
        assert isinstance(startup_program, Program)
        assert isinstance(main_program, Program)
        assert isinstance(stage, int)

        self._device = None
        if core.is_compiled_with_npu():
            self._device = "npu"
        elif core.is_compiled_with_cuda():
            self._device = "gpu"
        assert self._device, "Only gpu and npu are supported."
        assert not in_dygraph_mode(), "Only static mode is supported."
        self._stage = stage

        op_maker = core.op_proto_and_checker_maker
        self._op_role = op_maker.OpRole
        self._op_role_key = op_maker.kOpRoleAttrName()
        self._op_device_key = op_maker.kOpDeviceAttrName()

        self._param_device_map = None

        self._pipeline_pair = []
        self._pp_ring_map = dict()

        self._output_var_to_op = None
        self._input_var_to_op = None
        self._main_program = main_program
        self._startup_program = startup_program

    def _get_input_output_info(self, block):
        '''
        Get info of op input and output.
        '''
        # A map from output var to op which generate it.
        output_var_to_op = defaultdict(list)
        # A map from var to op which takes it as input.
        input_var_to_op = defaultdict(list)

        for index, op in enumerate(block.ops):
            for var_name in op.input_arg_names:
                input_var_to_op[var_name].append([op, index])
            for var_name in op.output_arg_names:
                output_var_to_op[var_name].append([op, index])

        return output_var_to_op, input_var_to_op

    def _update_param_device_map(self):
        """
        Get the device info for parameters.
        """
        params = self._main_program.all_parameters()
        for each_block in self._main_program.blocks:
            for op in each_block.ops:
                for var_name in op.input_arg_names:
                    if not var_name in params or var_name in self._param_device_map:
                        continue
                    device = op.attr(self._op_device_key)
                    self._param_device_map[var_name] = device

    def _split_program(self, program, stage, block_idx):
        """
        Split a program and get the one with the given pipeline stage.

        Args:
            stage (int): pipeline stage
            block_idx (int): block index
        """

        used_var_names = set()
        block = program.block(block_idx)
        op_idx = 0
        for op in list(block.ops):
            op_stage = op.attr(self._op_device_key).split(':')[1]
            # Copy ops whose op_device set to "gpu:all" to all sections.
            if op_stage == "all" or int(op_stage) == stage:
                for var_name in op.input_arg_names + op.output_arg_names:
                    used_var_names.add(var_name)
                op_idx += 1
                if op.type == "while":
                    sub_block_id = int(op.attr('sub_block').id)
                    self._split_program(program, stage, sub_block_id)
            else:
                block._remove_op(op_idx)

        for var_name in list(block.vars.keys()):
            if not var_name in used_var_names: block._remove_var(var_name)

    def _find_post_op(self, index, var_name):
        """
        Find the post op that has variable named var_name as input.
        """
        # bugfix for uniform hybrid parallelism
        if '.cast_fp32' in var_name:
            var_name = var_name.replace('.cast_fp32', '')
        if '.cast_fp16' in var_name:
            var_name = var_name.replace('.cast_fp16', '')

        post_ops = self._input_var_to_op[var_name]
        if post_ops == None: return None
        result_op = None
        for post_op, post_idx in reversed(post_ops):
            if post_idx > index:
                result_op = post_op
                break
        return result_op

    def _find_prev_op(self, index, var_name):
        """
        Find the previous op of op with index that outputs
        variable named var_name.
        """
        prev_ops = self._output_var_to_op[var_name]
        if prev_ops == None: return None
        result_op = None
        for prev_op, prev_idx in reversed(prev_ops):
            if prev_idx < index:
                result_op = prev_op
                break
        return result_op

    def _add_op_device_attr(self, block):
        """
        Add op_device attrribute for ops in block that have 
        not that attribute set.
        
        Args:
            block (Block): the block to process.
        """
        assert isinstance(block, Block)

        # Ops should be copied to all pipeline stages.
        device_all_ops = [
            "create_py_reader",
            "read",
            "create_double_buffer_reader",
            "while",
        ]

        for op in block.ops:
            if op.type in device_all_ops:
                # We use "gpu:all" to represent an op should be put on all
                # pipeline stages, such as read ops. Note that: "gpu:all"
                # is only used by pipeline as an indicator.
                op._set_attr(self._op_device_key, self._device + ":all")
            if op.type == "while":
                sub_block_id = op.attr('sub_block').id
                sub_block = block.program.block(sub_block_id)
                self._add_op_device_attr(sub_block)

    def _check_validation(self, block):
        """
        Check whether ops in a block have both the op_device and the 
        op_role attributes set.
        """
        assert isinstance(block, Block)

        pre_stage_id = None
        for op in block.ops:
            assert op.has_attr(self._op_role_key), (
                "{} has no {} set .".format(op.type, self._op_role_key))
            op_role = op.attr(self._op_role_key)
            assert op_role == int(self._op_role.Forward), (
                "Only forward is supported for inference.")
            if not op._has_kernel(op.type):
                assert op.type == "while", (
                    "The only supported op without kernel is while.")
                sub_block_id = op.attr('sub_block').id
                sub_block = block.program.block(sub_block_id)
                self._check_validation(sub_block)
            assert op.has_attr(self._op_device_key), (
                "{} has no {} set.".format(op.type, self._op_device_key))

            device = op.attr(self._op_device_key)
            assert device, (
                "{} has no {} set.".format(op.type, self._op_device_key))
            if device.split(':')[1] == "all": continue

            dev_type = device.split(':')[0]
            assert dev_type == self._device
            stage_id = int(device.split(':')[1])

            if pre_stage_id is not None:
                interval = stage_id - pre_stage_id
                assert 0 <= interval <= 1, (
                    "The stage in the pipeline must be consecutive.")
            pre_stage_id = stage_id

    def _insert_sendrecv_ops_for_boundaries(self, block):
        """
        Insert a pair of send and recv ops for every two
        consecutive ops on different devices.
        """
        # A map from var to device where op takes it as input,
        # avoiding multiple send and recv ops.
        input_var_to_device = dict()

        extra_index_info = {'index': 0, }

        for index, op in enumerate(list(block.ops)):
            cur_device = op.attr(self._op_device_key)
            if cur_device.split(':')[-1] == "all": continue
            for var_name in op.input_arg_names:
                if not block.has_var(var_name) and block._find_var_recursive(
                        var_name):
                    continue
                var = block.var(var_name)
                # skip data var
                if var.is_data: continue
                prev_device = None
                generate_ops = self._output_var_to_op.get(var_name)
                if generate_ops is None:
                    if var_name not in self._param_device_map:
                        continue
                    prev_device = self._param_device_map[var_name]

                prev_op = self._find_prev_op(index, var_name)

                if not prev_device:
                    prev_device = prev_op.attr(self._op_device_key) \
                        if prev_op else None

                if prev_device is None or prev_device.split(":")[-1] == "all":
                    continue

                if prev_device == cur_device: continue

                if var_name not in input_var_to_device:
                    input_var_to_device[var_name] = []
                if (cur_device, prev_device) in input_var_to_device[var_name]:
                    continue

                device_type = cur_device.split(':')[0] + ':'

                def _insert_send_recv(cur_id, prev_id):
                    assert cur_id > prev_id
                    cur_dev = device_type + str(cur_id)
                    prev_dev = device_type + str(prev_id)
                    if (cur_dev, prev_dev) in input_var_to_device[var_name]:
                        return

                    if cur_id - prev_id > 1:
                        _insert_send_recv(cur_id - 1, prev_id)
                        _insert_send_recv(cur_id, cur_id - 1)
                        input_var_to_device[var_name].append(
                            (cur_dev, prev_dev))
                        return

                    assert cur_id - prev_id == 1
                    input_var_to_device[var_name].append((cur_dev, prev_dev))

                    op_role = op.attr(self._op_role_key)
                    var = block.vars[var_name]
                    pair = (prev_id, cur_id)
                    # 1000 is just a magic number
                    pair_key = prev_id * 1000 + cur_id
                    if pair not in self._pipeline_pair:
                        self._pipeline_pair.append(pair)
                        self._pp_ring_map[pair_key] = self.ring_id
                        ring_id = self.ring_id
                        self.ring_id += 1
                    else:
                        ring_id = self._pp_ring_map[pair_key]

                    block._insert_op_without_sync(
                        index=index + extra_index_info['index'],
                        type='send_v2',
                        inputs={'X': var},
                        attrs={
                            self._op_device_key: prev_dev,
                            self._op_role_key: op_role,
                            'use_calc_stream': True,
                            'peer': 1,
                            'ring_id': ring_id
                        })
                    extra_index_info['index'] += 1
                    var_shape = list(var.shape)
                    var_shape[0] = self.micro_batch_size if var_shape[
                        0] < 0 else var_shape[0]
                    block._insert_op_without_sync(
                        index=index + extra_index_info['index'],
                        type='recv_v2',
                        outputs={'Out': [var]},
                        attrs={
                            'out_shape': var_shape,
                            'dtype': var.dtype,
                            self._op_device_key: cur_dev,
                            self._op_role_key: op_role,
                            'use_calc_stream': True,
                            'peer': 0,
                            'ring_id': ring_id
                        })
                    extra_index_info['index'] += 1

                _insert_send_recv(
                    int(cur_device.split(':')[1]),
                    int(prev_device.split(':')[1]))
        block._sync_with_cpp()

    def _get_while_block():
        """
        Get the while sub-block.
        """
        main_block = self._main_program.global_block()
        found = False
        sub_block_id = None
        for op in main_block.ops:
            assert not sub_block_id, "More than one while op found."
            if op.type == 'while':
                sub_block_id = op.attr('sub_block').id
        if sub_block_id: return self._main_program.block(sub_block_id)
        return None

    def gen_infer_program(self):
        """
        Generate inference program.
        """
        main_block = self._main_program.global_block()
        startup_block = self._startup_program.global_block()

        # step1: add op_device attribute for all ops
        self._add_op_device_attr(startup_block)
        self._check_validation(startup_block)
        self._add_op_device_attr(main_block)
        self._check_validation(main_block)

        # step2: add send/recv ops
        self._update_param_device_map()
        # step2.1: add send/recv for main_block
        out_var_to_op, in_var_to_op = self._get_input_output_info(main_block)
        self._output_var_to_op = out_var_to_op
        self._input_var_to_op = in_var_to_op
        self._insert_sendrecv_ops_for_boundaries(main_block)
        # step2.2: add send/recv for while_block
        while_block = self._get_while_block()
        if while_block:
            out_var_to_op, in_var_to_op = self._get_input_output_info(
                while_block)
            self._output_var_to_op = out_var_to_op
            self._input_var_to_op = in_var_to_op
            self._insert_sendrecv_ops_for_boundaries(while_block)

        # step3: split programs
        self._split_program(self._startup_program, self._stage, 0)
        self._split_program(self._main_program, self._stage, 0)


if __name__ == "__main__":
    import numpy as np
    import paddle
    import paddle.fluid as fluid
    paddle.enable_static()
    main_prog = paddle.fluid.Program()
    startup_prog = paddle.fluid.Program()
    with paddle.fluid.program_guard(main_prog, startup_prog):
        with fluid.device_guard("gpu:0"):
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            loop_len = fluid.layers.fill_constant(
                shape=[1], dtype='int64', value=10)
            one = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=1)
            data = fluid.data(name='data', shape=[1], dtype='float32')
            sums = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=0)

            cond = fluid.layers.less_than(x=i, y=loop_len)
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():
                sums_tensor = fluid.layers.elementwise_add(x=data, y=data)
                fluid.layers.assign(input=sums_tensor, output=sums)
                i = fluid.layers.increment(x=i, value=1, in_place=True)
                data = fluid.layers.elementwise_add(x=data, y=one)
                fluid.layers.less_than(x=i, y=loop_len, cond=cond)

    with open("./while_main_raw", 'w') as f:
        f.writelines(str(main_prog))
    helper = PipelineInferHelper(startup_prog, main_prog, 0)
    helper.gen_infer_program()
    with open("./while_main", 'w') as f:
        f.writelines(str(main_prog))

    feed_data = np.ones([1]).astype('float32')
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    res = exe.run(fluid.default_main_program(),
                  feed={'data': feed_data},
                  fetch_list=sums)
    print(res[0])
