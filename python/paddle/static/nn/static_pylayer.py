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

import paddle
from paddle.base import core
from paddle.base.backward import _append_grad_suffix_
from paddle.base.framework import Variable, in_pir_mode
from paddle.base.libpaddle.pir import build_pylayer_op, cf_yield
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
    def __init__(self, inputs, name=None, pylayer_context=None):
        # used to specify the Variable type `Input` to `pylayer` op
        self.fwd_inputs = [
            each_input
            for each_input in inputs
            if isinstance(each_input, Variable)
        ]  # filter non-Variable inputs

        # used to specify the `Out` to `pylayer` op
        self.fwd_outputs = []

        self.context = pylayer_context

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
        self.helper.main_program._sync_with_cpp()

    def complete_backward_block(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

        self._backward_block_id = inside_block.idx
        # Set OpRole to `backward`. The operators marked as `backward` are expected to be pruned in PruneBackward.
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
        _rename_var_recursively_(inside_block, self.var_old_to_new)

        # update `blocks` attr by appending backward_block
        forward_block_desc = parent_block.program.block(
            self.forward_block_index
        ).desc
        backward_block_desc = inside_block.desc
        parent_block.ops[self.fwd_op_index].desc.set_blocks_attr(
            "blocks", [forward_block_desc, backward_block_desc]
        )

        # remove temporary vars created by `StaticPyLayerContext.saved_tensor`
        if self.context:
            for var in self.context.saved_vars:
                if not inside_block.has_var(var.name):
                    raise ValueError(
                        f"{var.name} was saved in forward block but could not be found in backward block. Maybe {var.name} was renamed somewhere."
                    )
                inside_block._remove_var(var.name)

        self.helper.main_program._sync_with_cpp()

    def complete(self):
        if not self.is_backward_block:
            return self.complete_forward_block()
        else:
            return self.complete_backward_block()


def _get_ctx_from_func_(func):
    if func is None:
        return None

    fn_bind_args = getattr(func, "args", None)
    if fn_bind_args is None:
        return None

    from paddle.jit.dy2static.py_layer import StaticPyLayerContext

    fn_ctx = None
    if len(fn_bind_args) > 0 and isinstance(
        fn_bind_args[0], StaticPyLayerContext
    ):
        fn_ctx = fn_bind_args[0]

    return fn_ctx


def _rename_var_recursively_(cur_block, var_old_to_new):
    """
    Rename the var both the Variable instances and all ops' input and output arg names
    in `cur_block` based on dict `var_old_to_new`.
    Dict `var_old_to_new` should be the following format:
    {
        old_name_0 : new_name_0,
        old_name_1 : new_name_1,
        ...
        old_name_n : new_name_n,
    }
    """

    for old_var_name, new_var_name in var_old_to_new.items():
        # NOTE(MarioLulab): The reason why not using `Block._rename_var`` is that `Block._rename_var` will raise ValueError, when `old_var_name` does not correspond to a Variable instance in Block.

        if cur_block.has_var(old_var_name):
            # `Block.desc._rename_var` can rename var in block and then rename var name in all ops
            cur_block.desc._rename_var(
                old_var_name.encode(), new_var_name.encode()
            )
        else:
            # When cur_block does not have the var, `Block.desc._rename_var` can't rename var name in ops.
            # In this case, we should traverse all ops and perform renaming manually.
            for op in cur_block.ops:
                op._rename_input(old_var_name, new_var_name)
                op._rename_output(old_var_name, new_var_name)

    # NOTE(MarioLulab): block attr type with the name of "blocks" or "sub_block" indicates
    # the block might be executed. We should rename the var name in these blocks recursively
    block_attr_names = ["blocks", "sub_block"]

    for op in cur_block.ops:
        for attr_name in op.all_attrs():
            if attr_name not in block_attr_names:
                continue

            if op.attr_type(attr_name) == core.AttrType.BLOCK:
                sub_block_id = op._block_attr_id(attr_name)
                sub_block = cur_block.program.block(sub_block_id)
                _rename_var_recursively_(sub_block, var_old_to_new)
            elif op.attr_type(attr_name) == core.AttrType.BLOCKS:
                sub_blocks_ids = op._blocks_attr_ids(attr_name)
                for sub_block_id in sub_blocks_ids:
                    sub_block = cur_block.program.block(sub_block_id)
                    _rename_var_recursively_(sub_block, var_old_to_new)


def copy_var_from_parent_block(parent_block_var, layer_helper):
    if not isinstance(parent_block_var, Variable):
        return parent_block_var
    prog = layer_helper.main_program
    current_block = prog.current_block()

    if (
        parent_block_var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY
        and current_block._find_var_recursive(parent_block_var.name)
    ):
        current_block_var = parent_block_var
    else:
        current_block_var = current_block.create_var(
            dtype=parent_block_var.dtype,
            shape=parent_block_var.shape,
            type=parent_block_var.type,
        )
        paddle.assign(parent_block_var, current_block_var)
    return current_block_var


class PyLayerBackwardFunction:
    _register_backward_funcs = []

    def __init__(self, backward_function, hook_check_func):
        if backward_function is None or not callable(backward_function):
            raise TypeError('func must be a Python function')

        self._func = backward_function

        # Note: Used to verify the number of `Value` inputs to ``forward_fn`` the same as the
        # number of `Value` outputs to ``backward_fn``, and the number of `Value` outputs to ``forward_fn``
        # the same as the number of `Value` inputs to ``backward_fn``.
        self._hook_check_func = hook_check_func

        '''
        Why record self here?
           For increasing reference count of self.
           It seems that to release Python object
           whose reference count is 1 would cause
           segmentation fault error in C++ side.
           May be lack of Python GC in C++ side?
        '''
        PyLayerBackwardFunction._register_backward_funcs.append(self)

    def __call__(self, *output_grads):
        assert self._hook_check_func

        input_grads = self._func(*output_grads)
        if not isinstance(input_grads, (list, tuple)):
            input_grads = (input_grads,)

        self._hook_check_func(output_grads, input_grads)

        return input_grads


def static_pylayer(forward_fn, inputs, backward_fn=None, name=None):
    """
    This API returns ``forward_fn(inputs)``, and two sub-block are created based on
    the logic of ``forward_fn`` and ``backward_fn``, with the operator ``pylayer``
    holding information about the two blocks.

    ``forward_fn`` and ``backward_fn`` should return a nest structure of Variables.
    A nest structure of Variables in PaddlePaddle is Variable(s), or tuple of Variables, or
    list of Variables.

    Note:
        1. If ``backward_fn`` is not None, user needs to keep the number of `Variable` inputs to ``forward_fn`` the same as the
        number of `Variable` outputs to ``backward_fn``, and the number of `Variable` outputs to ``forward_fn``
        the same as the number of `Variable` inputs to ``backward_fn``.

        2. If ``backward_fn`` is None, ``stop_gradient`` attr of all Variable in ``inputs`` is expected to be True.
        Otherwise it might get unexpected results in backward propagation.

        3. This API can only be used under static graph mode.

    Args:
        forward_fn (callable): A callable to be performed in forward propagation
        inputs (list[Variable]): The list of input Variable to the ``forward_fn``
        backward_fn (callable, optional): A callable to be performed in backward propagation. Default: None, which means no need to do backward propagation.
        name (str, optional): The default value is ``None`` . Normally users
            don't have to set this parameter. For more information, please
            refer to :ref:`api_guide_Name` .

    Returns:
        Variable|list(Variable)|tuple(Variable): returns the output of ``forward_fn(inputs)``

    Examples:
        .. code-block:: python

                >>> import paddle
                >>> import numpy as np

                >>> paddle.enable_static()

                >>> def forward_fn(x):
                ...     return paddle.exp(x)

                >>> def backward_fn(dy):
                ...     return 2 * paddle.exp(dy)

                >>> main_program = paddle.static.Program()
                >>> start_program = paddle.static.Program()

                >>> place = paddle.CPUPlace()
                >>> exe = paddle.static.Executor(place)
                >>> with paddle.static.program_guard(main_program, start_program):
                ...     data = paddle.static.data(name="X", shape=[None, 5], dtype="float32")
                ...     data.stop_gradient = False
                ...     ret = paddle.static.nn.static_pylayer(forward_fn, [data], backward_fn)
                ...     data_grad = paddle.static.gradients([ret], data)[0]

                >>> exe.run(start_program)
                >>> x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
                >>> x, x_grad, y = exe.run(
                ...     main_program,
                ...     feed={"X": x},
                ...     fetch_list=[
                ...         data.name,
                ...         data_grad.name,
                ...         ret.name
                ...     ],
                ... )

                >>> print(x)
                [[1. 2. 3. 4. 5.]]
                >>> print(x_grad)
                [[5.4365635 5.4365635 5.4365635 5.4365635 5.4365635]]
                >>> print(y)
                [[  2.7182817   7.389056   20.085537   54.59815   148.41316  ]]
    """
    assert (
        in_dygraph_mode() is False
    ), "please use PyLayer instead of static_pylayer in dygraph mode"

    assert isinstance(inputs, list)
    if backward_fn is None:
        for input_var in inputs:
            if input_var.stop_gradient is False:
                raise ValueError(
                    f"``stop_gradient`` attr of all inputs to ``forward_fn`` are expected to be True, when ``backward_fn == None``, but {input_var.name}.stop_gradient got {input_var.stop_gradient}"
                )

    # judge if in dy2st or not, by checking binding args of `forward_fn` and `backward_fn`
    fwd_fn_ctx = _get_ctx_from_func_(forward_fn)
    bwd_fn_ctx = _get_ctx_from_func_(backward_fn)
    static_pylayer_context = (
        fwd_fn_ctx if fwd_fn_ctx and (fwd_fn_ctx == bwd_fn_ctx) else None
    )

    if in_pir_mode():
        fwd_inputs = [
            inp for inp in inputs if isinstance(inp, paddle.pir.Value)
        ]
        pylayer_op = build_pylayer_op(fwd_inputs)
        outputs = None
        if forward_fn is not None:
            if not callable(forward_fn):
                raise ValueError("`forward_fn` should be callable")
            with pylayer_op.forward_block():
                outputs = forward_fn(*inputs)

            if outputs is None:
                return None

            fwd_outputs = [
                out
                for out in flatten(outputs)
                if isinstance(out, paddle.pir.Value)
            ]

            with pylayer_op.forward_block():
                if fwd_outputs is not None:
                    cf_yield(flatten(fwd_outputs))
            pylayer_op.update_output()
        if backward_fn is not None:
            if not callable(backward_fn):
                raise ValueError("`bakcward_fn` should be callable")

            def hook_inputs_outputs_check_function(output_grads, input_grads):
                # 1. Verify the number of `Value` inputs to ``forward_fn`` the same as the
                # number of `Value` outputs to ``backward_fn``
                forward_inputs = [
                    x
                    for x in flatten(inputs)
                    if isinstance(x, paddle.pir.Value)
                ]
                if len(input_grads) != len(forward_inputs):
                    raise ValueError(
                        f"The number of input grads should be equal to the number of inputs, but got {len(input_grads)} and {len(inputs)}."
                    )
                for inp_grad, fwd_input in zip(input_grads, forward_inputs):
                    assert (
                        inp_grad.dtype == fwd_input.dtype
                    ), f"dtype of inp_grad({inp_grad.dtype}) and fwd_input({fwd_input.dtype}) should be the same"
                    assert (
                        inp_grad.shape == fwd_input.shape
                    ), f"shape of inp_grad({inp_grad.shape}) and fwd_input({fwd_input.shape}) should be the same"
                    assert (
                        inp_grad.type() == fwd_input.type()
                    ), f"type of inp_grad({inp_grad.type}) and fwd_input({fwd_input.type}) should be the same"

                # 2. Verify the number of `Value` outputs to ``forward_fn``
                # the same as the number of `Value` inputs to ``backward_fn``
                forward_outputs = [
                    x
                    for x in flatten(fwd_outputs)
                    if isinstance(x, paddle.pir.Value)
                ]
                if len(output_grads) != len(forward_outputs):
                    raise ValueError(
                        f"The number of output grads should be equal to the number of outputs, but got {len(output_grads)} and {len(fwd_outputs)}."
                    )
                for out_grad, fwd_output in zip(output_grads, forward_outputs):
                    assert (
                        out_grad.dtype == fwd_output.dtype
                    ), f"dtype of out_grad({out_grad.dtype}) and fwd_output({fwd_output.dtype}) should be the same"
                    assert (
                        out_grad.shape == fwd_output.shape
                    ), f"shape of out_grad({out_grad.shape}) and fwd_output({fwd_output.shape}) should be the same"
                    assert (
                        out_grad.type() == fwd_output.type()
                    ), f"type of out_grad({out_grad.type}) and fwd_output({fwd_output.type}) should be the same"

            bwd_fn = PyLayerBackwardFunction(
                backward_fn, hook_check_func=hook_inputs_outputs_check_function
            )
            pylayer_op.register_backward_function(bwd_fn)

        # NOTE: Replace pir.Value of `outputs` with pylayer_op.result, because value of `outputs` which is inside pylayer block can't be reference outside the block.
        op_result_idx = 0
        outputs = flatten(outputs)
        for i in range(len(outputs)):
            if isinstance(outputs[i], paddle.pir.Value):
                outputs[i] = pylayer_op.results()[op_result_idx]
                op_result_idx += 1
        return outputs[0] if len(outputs) == 1 else outputs

    check_type(name, "name", (str, type(None)), "base.layers.static_pylayer")
    helper = LayerHelper('static_pylayer', **locals())
    copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)

    assert forward_fn is not None and callable(forward_fn)
    pylayer_block_manager = StaticPyLayerBlock(
        inputs, pylayer_context=static_pylayer_context
    )
    with pylayer_block_manager.block(is_backward_block=False) as mgr:
        origin_output = forward_fn(*inputs)
        if origin_output is not None:
            output = map_structure(copy_to_parent_func, origin_output)
            mgr.fwd_outputs = [
                x for x in flatten(output) if isinstance(x, Variable)
            ]
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
        for fwd_var in pylayer_block_manager.fwd_outputs:
            fwd_var_name = fwd_var.name
            bwd_var_name = _append_grad_suffix_(fwd_var_name)
            if not current_block.desc.has_var_recursive(fwd_var_name.encode()):
                raise ValueError(
                    f"Grad var {bwd_var_name} , we can't find its related forward var {fwd_var_name}"
                )

            var = current_block.create_var(
                dtype=fwd_var.dtype,
                shape=fwd_var.shape,
                type=fwd_var.type,
                name=bwd_var_name,
            )

            grad_var_ins.append(var)

        copy_from_parent_func = lambda var: copy_var_from_parent_block(
            var, helper
        )
        assert isinstance(grad_var_ins, list)
        with pylayer_block_manager.block(is_backward_block=True) as mgr:
            # Step1. Copy var from parent block
            inside_block_inputs = map_structure(
                copy_from_parent_func, grad_var_ins
            )

            # Step2. Do backward propagation
            grad_origin_output = backward_fn(*inside_block_inputs)

            if grad_origin_output is not None:
                # Step3. Check the number of inputs to ``forward_fn`` the
                # same as the number of outputs to ``backward_fn``
                flat_grad_origin = flatten(grad_origin_output)

                # NOTE(MarioLulab): ``current_block`` was defined outside
                forward_input_names = current_block.ops[
                    pylayer_block_manager.fwd_op_index
                ].desc.input_arg_names()
                assert len(forward_input_names) == len(
                    flat_grad_origin
                ), f"needs to keep the number of inputs to ``forward_fn`` the same as the number of outputs to ``backward_fn``, \
                    but got {len(forward_input_names)} and {len(flat_grad_origin)}"

                # Step4. Rename var name with suffix of "@GRAD"
                for bwd_output, fwd_input_name in zip(
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
                    if isinstance(bwd_output, Variable):
                        bwd_out_new = _append_grad_suffix_(
                            fwd_input_name
                        )  # "X" => "X@GRAD"
                        mgr.var_old_to_new[
                            bwd_output.name
                        ] = bwd_out_new  # e.g. "tmp_0.mean_0": "X@GRAD"

        # **Delete the backward input**
        for bwd_var in grad_var_ins:
            current_block._remove_var(bwd_var.name)

    if origin_output is None:
        return None

    return output
