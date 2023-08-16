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

import collections
import re
import warnings

from paddle.common_ops_import import LayerHelper, check_type, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.utils import flatten, map_structure

# NOTE: Borrowed from `python/paddle/static/nn/control_flow.py`
from .control_flow import (
    BlockGuard,
    copy_var_to_parent_block,
    get_inputs_outputs_in_block,
)


class StaticPyLayerBlockGuard(BlockGuard):
    def __init__(self, block):
        check_type(
            block, "block", StaticPyLayerBlock, "StaticPyLayerBlockGuard"
        )
        super().__init__(block.helper.main_program)
        self.block = block

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.block.complete()
        return super().__exit__(exc_type, exc_val, exc_tb)


class StaticPyLayerBlock:
    def __init__(self, inputs, name=None):
        for each_input in inputs:
            check_type(each_input, "input", Variable, "StaticPyLayerBlock")

        self.helper = LayerHelper("static_pylayer_block", name=name)

    def block(self, is_backward_block=False):
        self.is_backward_block = is_backward_block
        return StaticPyLayerBlockGuard(self)

    @property
    def inside_block_index(self):
        return self.block_id

    @property
    def op_index(self):
        return self.op_id

    def complete(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

        self.block_id = inside_block.idx

        if self.is_backward_block:
            # set OpRole to `backward`
            for op in inside_block.ops:
                op_role_attr_name = (
                    core.op_proto_and_checker_maker.kOpRoleAttrName()
                )
                backward = core.op_proto_and_checker_maker.OpRole.Backward
                op.desc._set_attr(op_role_attr_name, backward)

            # exit, because there is no need to append 'static_pylayer' op
            return

        intermediate = set()  # inner_outputs
        params = set()  # inner_inputs
        params, intermediate = get_inputs_outputs_in_block(
            inside_block, params, intermediate, helper=self.helper
        )

        param_list = [
            parent_block._var_recursive(each_name) for each_name in params
        ]

        out_list = []
        for inner_out_name in intermediate:
            inner_var = parent_block._find_var_recursive(inner_out_name)
            if inner_var:
                out_list.append(inner_var)

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES
        )

        static_pylayer_op = parent_block.append_op(
            type='static_pylayer',
            inputs={
                'Input': param_list,
            },
            outputs={"Out": out_list, "Scope": [step_scope]},
            attrs={
                'blocks': [inside_block],
            },
        )
        self.op_id = static_pylayer_op.idx


# NOTE: Borrowed from `backward.py`
def _strip_grad_suffix_(name):
    """
    Strip the grad suffix from the given variable name
    e.g. x@GRAD ==> x
         x@GRAD@GRAD ==> x
         y@GRAD@RENAME@1 ==> y
         z@GRAD_slice_0@GRAD ==> z@GRAD_slice_0
         grad/grad/z@GRAD@RENAME@block0@1@GRAD ==> z
    """
    pos = re.search(f'{core.grad_var_suffix()}+@', name) or re.search(
        f'{core.grad_var_suffix()}$', name
    )
    new_name = name[: pos.start()] if pos is not None else name
    new_pos = name.rfind('grad/')
    return new_name[new_pos + 5 :] if new_pos != -1 else new_name


# NOTE: Borrowed from `backward.py`
def _rename_arg_(op_descs, old_name, new_name, begin_idx=None, end_idx=None):
    """
    Traverse all ops in op_descs[begin_idx : end_idx],
    if any op has inputs/outputs named "old_name", rename it as 'new_name'
    """
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_descs)
    if isinstance(op_descs, (list, tuple)):
        for i in range(begin_idx, end_idx):
            op_desc = op_descs[i]
            if isinstance(op_desc, tuple):
                op_desc = op_desc[0]
            op_desc._rename_input(old_name, new_name)
            op_desc._rename_output(old_name, new_name)
    if isinstance(op_descs, collections.OrderedDict):
        for key, value in op_descs.items():
            if isinstance(value, (list, tuple)):
                for op_desc in value:
                    op_desc._rename_input(old_name, new_name)
                    op_desc._rename_output(old_name, new_name)


# NOTE: Borrowed from `backward.py`
def _append_grad_suffix_(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    return name + core.grad_var_suffix()


#TODO(MarioLulab): Now the backward block will be created whether or not 
# a gradient is required. This will be fixed later.
):
    """
    This API returns ``forward_fn(inputs)``, and two sub-block are created based on 
    the logic of ``forward_fn`` and ``backward_fn``, with the operator ``static_pylayer``
    holding information about the two blocks. 
    
    ``forward_fn`` and ``backward_fn`` should return same nest structure of tensors
    or both return ``None`` if user doens't like to return anything. A nest
    structure of tensors in PaddlePaddle is tensor(s), or tuple of tensors, or
    list of tensors.
    
    Note:
        1. User needs to keep the number of inputs to ``forward_fn`` the same as the 
        number of outputs to ``backward_fn``, and the number of outputs to ``forward_fn`` 
        the same as the number of inputs to ``backward_fn``.
    
        2. This API can only be used under static graph mode.
            
    Args:
        forward_fn (callable): A callable to be performed in forward pass
        inputs (list[Variable]): The list of if input Variable to the ``forward_fn`` 
        backward_fn (callable): A callable to be performed in backward pass
        name (str, optional ): The default value is ``None`` . Normally users
            don't have to set this parameter.

    Returns:
        Variable|list(Variable)|tuple(Variable): returns the output of ``forward_fn(inputs)``
    
    Examples:
        .. code-block: python
            
            import paddle
            import numpy as np

            #
            # pseudocode:
            # y = exp(x)
            # dx = 2 * exp(dy)
            #

            paddle.enable_static()

            def forward_fn(x):
                return paddle.exp(x)

            def backward_fn(dy):
                return 2 * paddle.exp(dy)

            main_program = paddle.static.Program()
            start_program = paddle.static.Program()

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            with paddle.static.program_guard(main_program, start_program):
                data = paddle.static.data(name="X", shape=[None, 5], dtype="float32")
                data.stop_gradient = False
                ret = paddle.static.nn.static_pylayer(forward_fn, [data], backward_fn)
                data_grad = paddle.static.gradients([ret], data)[0]

            exe = paddle.static.Executor(place)
            exe.run(start_program)
            x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
            x, x_grad, y = exe.run(
                main_program,
                feed={"X": x},
                fetch_list=[
                    data.name,
                    data_grad.name,
                    ret.name
                ],
            )
            # x is Numpy
            # x.data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
            # x.shape = [1, 5]
            # y is Numpy
            # y.data = [[2.7182817, 7.389056, 20.085537, 54.59815, 148.41316]]
            # y.shape = [1, 5]
            # x_grad is Numpy
            # x_grad.data = [[5.4365635, 5.4365635, 5.4365635, 5.4365635, 5.4365635]]
            # x_grad.shape = [1, 5]     
    """
    assert(
        in_dygraph_mode() is False
    ), "please use PyLayer instead of static_pylayer in dygraph mode"
    
    check_type(name, "name", (str, type(None)), "fluid.layers.static_pylayer")
    helper = LayerHelper('static_pylayer', **locals())
    copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)

    assert forward_fn is not None and callable(forward_fn)
    assert isinstance(inputs, list)
    static_pylayer_block = StaticPyLayerBlock(inputs)
    with static_pylayer_block.block():
        origin_output = forward_fn(*inputs)
        if origin_output is not None:
            output = map_structure(copy_to_parent_func, origin_output)

    current_block = helper.main_program.current_block()
    forward_block_desc = current_block.program.block(
        static_pylayer_block.block_id
    ).desc

    # 根据 op 的 output 构造 backward 的输入, 以此构建 backward block, 随后删掉
    static_pylayer_op = current_block.ops[static_pylayer_block.op_index]
    grad_var_name_ins = [
        _append_grad_suffix_(var_name)
        for var_name in static_pylayer_op.desc.output("Out")
    ]

    # push
    # NOTE: may be no need to get arg name by `grad_op_descs`
    grad_var_ins = []
    for arg in grad_var_name_ins:
        # do some judge
        # ....

        fwd_name = _strip_grad_suffix_(arg)
        var = current_block.create_var(name=arg)

        if current_block.desc.has_var_recursive(fwd_name.encode()):
            fwd_var = current_block.desc.find_var_recursive(fwd_name.encode())
            var.desc.set_dtype(fwd_var.dtype())
            var.desc.set_shape(fwd_var.shape())
        else:
            # TODO(jiabin): Maybe we should not to this to cause some unexpected error on dtype
            warnings.warn(
                "Set grad var: {} dtype to default FP32, since we can't find its related forward var".format(
                    arg
                )
            )
            var.set_dtype(core.VarDesc.VarType.FP32)

        grad_var_ins.append(var)

    assert backward_fn is not None and callable(backward_fn)
    assert isinstance(grad_var_ins, list)
    static_pylayer_backward_block = StaticPyLayerBlock(grad_var_ins)
    var_old_to_new = {}
    with static_pylayer_backward_block.block(is_backward_block=True):
        grad_origin_output = backward_fn(*grad_var_ins)
        if grad_origin_output is not None:
            flat_grad_origin = flatten(grad_origin_output)
            forward_input_names = current_block.ops[
                static_pylayer_block.op_index
            ].desc.input_arg_names()
            for idx, grad_out_old in enumerate(flat_grad_origin):
                # attach old var name into new
                forward_input_name = forward_input_names[idx]
                grad_out_new = _append_grad_suffix_(forward_input_name)
                var_old_to_new[grad_out_old.name] = grad_out_new

    # pop
    for arg in grad_var_name_ins:
        current_block._remove_var(arg)

    backward_block_desc = current_block.program.block(
        static_pylayer_backward_block.block_id
    ).desc
    for old_var_name, new_var_name in var_old_to_new.items():
        backward_block_desc._rename_var(
            old_var_name.encode(), new_var_name.encode()
        )
        for op_idx in range(backward_block_desc.op_size()):
            op = backward_block_desc.op(op_idx)
            op._rename_input(old_var_name, new_var_name)
            op._rename_output(old_var_name, new_var_name)

    # update 'blocks' attr to forward op
    static_pylayer_op.desc.set_blocks_attr(
        "blocks", [forward_block_desc, backward_block_desc]
    )

    if output is None:
        return None

    return output
