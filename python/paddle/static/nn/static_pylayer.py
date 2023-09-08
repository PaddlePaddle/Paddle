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


from paddle.base import core
from paddle.base.backward import _append_grad_suffix_
from paddle.base.framework import Variable
from paddle.common_ops_import import LayerHelper, check_type, in_dygraph_mode
from paddle.utils import flatten, map_structure

# NOTE(MarioLulab): Borrowed from `python/paddle/static/nn/control_flow.py`
from .control_flow import BlockGuard, copy_var_to_parent_block


class StaticPyLayerBlockGuard(BlockGuard):
    def __init__(self, block_manager):
        check_type(
            block_manager,
            "block",
            StaticPyLayerBlock,
            "StaticPyLayerBlockGuard",
        )
        super().__init__(block_manager.helper.main_program)
        self.block_manager = block_manager

    def __enter__(self):
        super().__enter__()
        return self.block_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.block_manager.complete()
        return super().__exit__(exc_type, exc_val, exc_tb)


class StaticPyLayerBlock:
    def __init__(self, inputs, name=None):
        for each_input in inputs:
            check_type(each_input, "input", Variable, "StaticPyLayerBlock")

        # used to specify the `Input` to `pylayer` op
        self.fwd_inputs = inputs
        # used to specify the `Out` to `pylayer` op
        self.fwd_outputs = []

        self.helper = LayerHelper("static_pylayer_block", name=name)
        self.fwd_op_id = None
        self._forward_block_id = None
        self._backward_block_id = None
        self.var_old_to_new = {}

    def block(self, is_backward_block=False):
        self.is_backward_block = is_backward_block
        return StaticPyLayerBlockGuard(self)

    @property
    def forward_block_index(self):
        return self._forward_block_id

    @property
    def backward_block_index(self):
        return self._backward_block_id

    @property
    def fwd_op_index(self):
        return self.fwd_op_id

    def complete_forward_block(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)
        self._forward_block_id = inside_block.idx

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES
        )

        pylayer_op = parent_block.append_op(
            type='pylayer',
            inputs={
                'Input': self.fwd_inputs,
            },
            outputs={"Out": self.fwd_outputs, "Scope": [step_scope]},
            attrs={
                'blocks': [inside_block],
            },
        )

        self.fwd_op_id = pylayer_op.idx

    def complete_backward_block(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

        self._backward_block_id = inside_block.idx
        # set OpRole to `backward`
        for op in inside_block.ops:
            op_role_attr_name = (
                core.op_proto_and_checker_maker.kOpRoleAttrName()
            )
            backward = core.op_proto_and_checker_maker.OpRole.Backward
            op.desc._set_attr(op_role_attr_name, backward)
        inside_block._set_forward_block_idx(self.forward_block_index)

        # NOTE(MarioLulab): The reason of renaming the var name in the inside block is that
        # we need to associating `inside_grads` and `outside_grads` at
        # runtime `RunImpl` in pylayer op
        for old_var_name, new_var_name in self.var_old_to_new.items():
            # TODO(MarioLulab): need to remove recursively in ``sub_block``

            # NOTE(MarioLulab): The reason why not using Block._rename_var is that `old_var_name` does not correspond to a Variable instance in Block
            # and Block._rename_var will raise ValueError.
            inside_block.desc._rename_var(
                old_var_name.encode(), new_var_name.encode()
            )

        # update `blocks` attr by appending backward_block
        forward_block_desc = parent_block.program.block(
            self.forward_block_index
        ).desc
        backward_block_desc = inside_block.desc
        parent_block.ops[self.fwd_op_index].desc.set_blocks_attr(
            "blocks", [forward_block_desc, backward_block_desc]
        )

    def complete(self):
        if not self.is_backward_block:
            return self.complete_forward_block()
        else:
            return self.complete_backward_block()


# TODO(MarioLulab):
# Need to support non-Variable in ``inputs``
def static_pylayer(forward_fn, inputs, backward_fn=None, name=None):
    """
    This API returns ``forward_fn(inputs)``, and two sub-block are created based on
    the logic of ``forward_fn`` and ``backward_fn``, with the operator ``pylayer``
    holding information about the two blocks.

    ``forward_fn`` and ``backward_fn`` should return a nest structure of tensors.
    A nest structure of tensors in PaddlePaddle is tensor(s), or tuple of tensors, or
    list of tensors.

    Note:
        1. If ``backward_fn`` is not None, user needs to keep the number of inputs to ``forward_fn`` the same as the
        number of outputs to ``backward_fn``, and the number of outputs to ``forward_fn``
        the same as the number of inputs to ``backward_fn``.

        2. If ``backward_fn`` is None, ``stop_gradient`` attr of all Variable in ``inputs`` is expected to be True.
        Otherwise it might get unexpected results in backward pass.

        3. This API can only be used under static graph mode.

    Args:
        forward_fn (callable): A callable to be performed in forward pass
        inputs (list[Variable]): The list of if input Variable to the ``forward_fn``
        backward_fn (callable, optional): A callable to be performed in backward pass
        name (str, optional): The default value is ``None`` . Normally users
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
    assert (
        in_dygraph_mode() is False
    ), "please use PyLayer instead of static_pylayer in dygraph mode"

    assert isinstance(inputs, list)
    if backward_fn is None:
        for input_var in inputs:
            if input_var.stop_gradient is False:
                raise ValueError(
                    "``stop_gradient`` attr of all inputs to ``forward_fn`` are expected to be True, when ``backward_fn == None``, but {}.stop_gradient got {}".format(
                        input_var.name, input_var.stop_gradient
                    )
                )

    check_type(name, "name", (str, type(None)), "base.layers.static_pylayer")
    helper = LayerHelper('static_pylayer', **locals())
    copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)

    assert forward_fn is not None and callable(forward_fn)
    pylayer_block_manager = StaticPyLayerBlock(inputs)
    with pylayer_block_manager.block(is_backward_block=False) as mgr:
        origin_output = forward_fn(*inputs)
        if origin_output is not None:
            output = map_structure(copy_to_parent_func, origin_output)
            mgr.fwd_outputs = flatten(output)
        else:
            mgr.fwd_outputs = []

    current_block = helper.main_program.current_block()
    current_block._sync_with_cpp()
    if backward_fn is not None:
        assert callable(backward_fn)
        if origin_output is None:
            output = []

        # **Create the backward input** from the output of the op to build the
        # backward block, and then delete it.
        grad_var_ins = []
        for fwd_var in flatten(output):
            fwd_var_name = fwd_var.name
            bwd_var_name = _append_grad_suffix_(fwd_var_name)
            var = current_block.create_var(name=bwd_var_name)
            if not current_block.desc.has_var_recursive(fwd_var_name.encode()):
                raise ValueError(
                    "Grad var {} , we can't find its related forward var {}".format(
                        bwd_var_name, fwd_var_name
                    )
                )

            var.desc.set_dtype(fwd_var.dtype)
            var.desc.set_shape(fwd_var.shape)

            grad_var_ins.append(var)

        assert isinstance(grad_var_ins, list)
        with pylayer_block_manager.block(is_backward_block=True) as mgr:
            grad_origin_output = backward_fn(*grad_var_ins)
            if grad_origin_output is not None:
                flat_grad_origin = flatten(grad_origin_output)
                # NOTE(MarioLulab): ``current_block`` was defined outside
                forward_input_names = current_block.ops[
                    pylayer_block_manager.fwd_op_index
                ].desc.input_arg_names()
                assert len(forward_input_names) == len(
                    flat_grad_origin
                ), f"needs to keep the number of inputs to ``forward_fn`` the same as the number of outputs to ``backward_fn``, \
                    but got {len(forward_input_names)} and {len(flat_grad_origin)}"

                for bwd_output_name, fwd_input_name in zip(
                    flat_grad_origin, forward_input_names
                ):
                    # NOTE(MarioLulab): Because `flat_grad_origin` are the Variables inside the backward block, which one by one corresponds
                    # to the gradients of the inputs to the forward function, we need to establish a link between `flat_grad_origin`,
                    # and the Variable outside the backward block which represent the gradient of the input ot the forward function.
                    # The approach we have taken is renaming `flat_grad_origin` by forward input name with suffix of "@GRAD", and aligning
                    # the order of `Out@GRAD` in `pylayer_grad` op with `flat_grad_origin`. And in the runtime `RunImpl` in `pylayer_grad` op,
                    # we will find inside_grad with the name of forward input name with suffix of "@GRAD" in the scope, and assign `inside_grads`
                    # to `outside_grads`.
                    #
                    # Example:
                    # after run the code below to create forward and backward block:
                    #
                    #    out = forward_fn(x, y)                 # create forward block
                    #    x_grad, y_grad = backward_fn(out_grad) # create backward block
                    #
                    # x.name is "X", y.name is "Y", and out.name is "tmp_0", but x_grad.name is "_generate_0", y_grad.name is "_generate_1".
                    # we rename x_grad by "X@GRAD", and y_grad by "Y@GRAD" inside backward block.
                    # One thing to keep in mind is that we assume there were no Variable naming "X@GRAD" inside backward block before performing rename operation.
                    # TODO(MarioLulab): We will validate the assumption above is whether a strong hypothesis or not.

                    # attach old var name into new
                    bwd_out_new = _append_grad_suffix_(
                        fwd_input_name
                    )  # "X" => "X@GRAD"
                    mgr.var_old_to_new[
                        bwd_output_name.name
                    ] = bwd_out_new  # e.g. "tmp_0.mean_0": "X@GRAD"

        # **Delete the backward input**
        for bwd_var in grad_var_ins:
            current_block._remove_var(bwd_var.name)

    if origin_output is None:
        return None

    return output
