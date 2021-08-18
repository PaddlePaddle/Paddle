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

from paddle.fluid.framework import Block, Operator
from paddle.fluid.framework import in_dygraph_mode
import paddle.fluid.core as core


class PipelineInferHelper(object):
    """
    A helper class to split program for inference with pipeline parallelism.
    """

    def __init__(
            self,
            startup_program,
            main_program, ):
        self._device = None
        if core.is_compiled_with_npu():
            self._device = "npu"
        elif core.is_compiled_with_cuda():
            self._device = "gpu"
        assert self._device, "Now only gpu and npu are supported."
        assert in_dygraph_mode(), "Now only static mode is supported."
        op_maker = core.op_proto_and_checker_maker
        self._op_role_key = op_maker.kOpRoleAttrName()
        self._op_device_key = op_maker.kOpDeviceAttrName()
        self._param_device_map = None
        self._pipeline_pair = []
        self._pp_ring_map = dict()
        self._output_var_to_op = None
        self._input_var_to_op = None
        self._main_program = main_program
        self._startup_program = startup_program

    def _split_program(self, stage, block_idx):
        """
        Split a program and get the one with the given pipeline stage.

        Args:
            stage (int): pipeline stage
            block_idx (int): block index
        """

        used_var_names = set()
        block = self._main_program.block(block_idx)
        op_idx = 0
        for op in list(block.ops):
            op_stage = op.attr(self._op_device_key).split(':')[1]
            # Copy ops whose op_device set to "gpu:all" to all sections.
            if op_stage == "all" or int(op_stage) == stage:
                for var_name in op.input_arg_names + op.output_arg_names:
                    used_var_names.add(var_name)
                op_idx += 1
                if op.type == "while":
                    sub_block_id = int(op.attr('sub_block'))
                    self._split_program(stage, sub_block_id)
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

        post_ops = self.input_var_to_op[var_name]
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
        prev_ops = self.output_var_to_op[var_name]
        if prev_ops == None: return None
        result_op = None
        for prev_op, prev_idx in reversed(prev_ops):
            if prev_idx < index:
                result_op = prev_op
                break
        return result_op

    def _rename_arg(self, op, old_name, new_name):
        op._rename_input(old_name, new_name)
        op._rename_output(old_name, new_name)

    def _get_op_device_attr(self, op):
        """
        Get the op_device attribute of a op.
        
        Args:
            op (Operator): the op to process.
        """
        assert isinstance(op, Operator)

        device = op.attr(self._op_device_key) \
            if op.has_attr(self._op_device_key) else None
        if device:
            assert device[0:3] == 'gpu' or device[0:3] == 'npu', (
                "Only gpu and npu devices are supported in pipeline parallemism."
            )
        return device

    def _add_op_device_attr(self, block):
        """
        Add op_device attrribute for ops in block that have 
        not that attribute set.
        
        Args:
            block (Block): the block to process.
            device_type (str): the device type, such as 'gpu', 'npu'
        """
        assert isinstance(block, Block)
        assert isinstance(device_type, str)

        read_ops = [
            "create_py_reader",
            "read",
            "create_double_buffer_reader",
        ]

        for idx, op in enumerate(list(block.ops)):
            if op.type in read_ops:
                # Copy read related ops to all section to make them exit 
                # after each epoch.
                # We use "gpu:all" to represent ops should be put on all
                # sub-programs, such as lr ops. Note that: "gpu:all"
                # is only used by pipeline as an indicator.
                op._set_attr(self._op_device_key, self._device + ":all")
                continue

    def _check_validation(self, block):
        """
        Check whether ops in a block have both the op_device and the 
        op_role attributes set.
        """
        pre_stage_id = None

        for op in block.ops:
            assert op.has_attr(self._op_role_key), \
                "op ({}) has no {} attribute.".format(op.type, self._op_role_key)
            op_role = op.attr(self._op_role_key)
            assert op_role == int(_OP_ROLE.Forward)
            if not op._has_kernel(op.type):
                assert op.type == "while", (
                    "Now, the only supported op without kernel is "
                    "while with the op_role Forward.")

            assert op.has_attr(self._op_device_key), (
                "op ({}) has no {} attribute.".format(op.type,
                                                      self._op_device_key))

            device = op.attr(self._op_device_key)
            assert device, ("op_device attribute for op "
                            "{} has not been set.".format(op.type))
            if device.split(':')[1] == "all": continue

            dev_type = device.split(':')[0]
            stage_id = int(device.split(':')[1])
            assert dev_type == "gpu" or dev_type == 'npu', (
                "Now only gpu and npu devices are supported "
                "for pipeline parallelism.")

            if device not in device_list:
                device_list.append(device)

            if pre_stage_id is not None:
                interval = stage_id - pre_stage_id
                assert interval >= 0 and interval <= 1, \
                    "The stage interval of two consecutive ops in the pipeline must be < = 1," \
                    "but the interval of op={} and prev op is {}".format(op, interval)
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
            if op.type == 'while':
                sub_block_id = int(op.attr('sub_block'))
                self._insert_sendrecv_ops_for_boundaries(
                    block.program.block(sub_block_id))
            cur_device = op.attr(_OP_DEVICE_KEY)
            if cur_device.split(':')[-1] == "all": continue
            for var_name in op.input_arg_names:
                var = block.var(var_name)
                # skip data var
                if var.is_data: continue
                prev_device = None
                generate_ops = self.output_var_to_op.get(var_name)
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

    def gen_infer_program(self):
        """
        Generate inference program.
        
        Returns:
            tuple(main_program, startup_program)
        """
