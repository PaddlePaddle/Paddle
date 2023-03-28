#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
from os import path

import paddle
from paddle.fluid.proto import framework_pb2

from ...fluid import core, unique_name
from ...fluid.framework import OpProtoHolder

try:
    from paddle.fluid.proto import pass_desc_pb2
except ModuleNotFoundError:
    import sys

    fluid_path = path.dirname(__file__) + '/../../fluid'
    sys.path.append(path.join(fluid_path, 'proto'))
    from paddle.fluid.proto import pass_desc_pb2


class RegisterPassHelper:
    _register_helpers = []

    def __init__(self, pass_pairs, pass_type='', input_specs={}):
        self._pass_type = pass_type
        self._pass_pairs = pass_pairs
        self._input_specs = input_specs
        RegisterPassHelper._register_helpers.append(self)

    def _get_args_from_func(self, func):
        args = []
        arg_specs = inspect.getfullargspec(func)
        for arg_name in arg_specs.args:
            input_spec = self._input_specs.get(arg_name)
            if isinstance(input_spec, paddle.static.InputSpec):
                args.append(
                    PassDesc.VarHelper(
                        arg_name, input_spec.shape, input_spec.dtype
                    )
                )
            elif isinstance(input_spec, paddle.ParamAttr):
                args.append(paddle.ParamAttr(arg_name))
            else:
                args.append(PassDesc.VarHelper(arg_name, [-1]))
        return args

    def _prune_program_desc(self, ops):
        for op_desc in ops:
            default_attrs = core.get_op_attrs_default_value(
                op_desc.type.encode()
            )
            remove_attrs = []
            for attr in op_desc.attrs:
                # attr must not in
                if attr.name not in [
                    "op_namescope",
                    "op_callstack",
                    "op_device",
                ]:
                    attr_list_fields = attr.ListFields()
                    # attr format must be: name, type, value
                    if len(attr_list_fields) == 3:
                        attr_value = attr.ListFields()[-1][-1]
                        default_attr_value = default_attrs.get(attr.name)
                        # value must not default
                        if default_attr_value != attr_value:
                            continue
                remove_attrs.append(attr)
            for attr in remove_attrs:
                op_desc.attrs.remove(attr)

    def _func_to_program_desc(self, func, ops):
        vars = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            args = self._get_args_from_func(func)
            vars.extend(args)
            outs = func(*args)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            for out in outs:
                if isinstance(out, PassDesc.OpHelper):
                    op_outs = out.Outputs()
                    if len(op_outs) != 1:
                        raise ValueError(
                            "Operator '{}' has multiple outputs, please specify one output variable.".format(
                                out._type
                            )
                        )
                    for op_out in op_outs.values():
                        vars.extend(op_out)
                else:
                    vars.append(out)
        block_desc = program.current_block().desc
        for i in range(block_desc.op_size()):
            ops.add().ParseFromString(block_desc.op(i).serialize_to_string())
        self._prune_program_desc(ops)
        return vars, program.current_block().ops

    def _convert_vars_to_pass_desc(self, patterns, replaces, desc):
        def _add_element_conditions(conditions, elements):
            for element in elements:
                if element._condition:
                    conditions.append(element._condition)
                _add_element_conditions(conditions, element._elements)

        for (pattern, replace) in zip(patterns, replaces):
            # Convert maps of inputs and outputs.
            var_map = desc.var_maps.add()
            var_map.pattern_var = pattern.name
            var_map.replace_var = replace.name
            conditions = desc.var_attr_conditions
            # Convert shape condition.
            if pattern.name in self._input_specs:
                condition = conditions.add()
                pattern.Attr("shape")._to_pass_desc_attr(condition.attr)
                condition.condition_value.name = ""
                condition.condition_value.type = framework_pb2.AttrType.LONGS
                condition.condition_value.longs.extend(pattern.shape)
                condition.type = pass_desc_pb2.PassDesc.ConditionType.kEQ
            # Convert attr conditions.
            if PassDesc.VarHelper == pattern.__class__:
                for attr in pattern._attrs.values():
                    _add_element_conditions(conditions, [attr])

    def _convert_ops_to_pass_desc(self, patterns, replaces, desc):
        for replace in replaces:
            if isinstance(replace, PassDesc.OpHelper):
                for attr in replace._attrs.values():
                    # Convert attr maps.
                    mapped = attr._mapped
                    if inspect.isfunction(mapped):
                        mapped = mapped(patterns)
                    attr_map = desc.op_attr_maps.add()
                    mapped._to_pass_desc_attr(attr_map.pattern_attr)
                    attr._to_pass_desc_attr(attr_map.replace_attr)
                    if mapped._operation is not None:
                        attr_map.operation.CopyFrom(mapped._operation)

    def SerializeMultiPassDesc(self):
        switch_static_mode = paddle.in_dynamic_mode()
        if switch_static_mode:
            paddle.enable_static()
        multi_pass_desc = pass_desc_pb2.MultiPassDesc()
        multi_pass_desc.pass_type = self._pass_type
        # Traverse all pass pairs and convert them to PassDesc data.
        # Here need to add cache in the future.
        for (pattern, replace) in self._pass_pairs:
            pass_desc = multi_pass_desc.pass_descs.add()
            # Convert ProgramDescs of pattern and replace subgraphs.
            pattern_vars, pattern_ops = self._func_to_program_desc(
                pattern, pass_desc.pattern
            )
            replace_vars, replace_ops = self._func_to_program_desc(
                replace, pass_desc.replace
            )
            self._convert_vars_to_pass_desc(
                pattern_vars, replace_vars, pass_desc
            )
            self._convert_ops_to_pass_desc(pattern_ops, replace_ops, pass_desc)
        if switch_static_mode:
            paddle.disable_static()
        return multi_pass_desc.SerializeToString()


class PassDesc:
    class AttrHelper:
        def __init__(self, obj, name, element_index=None):
            self._obj = obj
            self._name = name
            self._operation_type = None
            self._element_index = element_index
            self._elements = []
            self._operation = None
            self._condition = None
            self._mapped = None

        def __getitem__(self, index):
            element = PassDesc.AttrHelper(
                self._obj, self._name, element_index=index
            )
            self._elements.append(element)
            return element

        def _to_pass_desc_attr(self, pass_desc_attr):
            if isinstance(self._obj, PassDesc.VarHelper):
                pass_desc_attr.role = pass_desc_pb2.PassDesc.RoleType.kVariable
                pass_desc_attr.var_name = self._obj.name
            else:
                pass_desc_attr.role = pass_desc_pb2.PassDesc.RoleType.kOperator
                pass_desc_attr.op_index = self._obj._index
            pass_desc_attr.name = self._name
            if self._operation_type is not None:
                pass_desc_attr.operation = self._operation_type
            if self._element_index is not None:
                pass_desc_attr.element_index = self._element_index

        def _to_op_desc_attr(self, value, op_desc_attr):
            op_desc_attr.name = ""
            if isinstance(value, int):
                op_desc_attr.type = framework_pb2.AttrType.INT
                op_desc_attr.i = value
            else:
                raise NotImplementedError("Unimplemented transform operation.")

        def _clone_with_operation(self, type, value=None):
            attr = PassDesc.AttrHelper(
                self._obj, self._name, self._element_index
            )
            self._elements.append(attr)
            if value is None:
                attr._operation_type = type
                return attr
            operation = pass_desc_pb2.PassDesc.Operation()
            operation.type = type
            if isinstance(value, PassDesc.AttrHelper):
                value._to_pass_desc_attr(operation.attr)
            else:
                self._to_op_desc_attr(value, operation.value)
            attr._operation = operation
            attr._operation_type = self._operation_type
            return attr

        def __sub__(self, value):
            return self._clone_with_operation(
                pass_desc_pb2.PassDesc.OperationType.kSub, value
            )

        def __add__(self, value):
            return self._clone_with_operation(
                pass_desc_pb2.PassDesc.OperationType.kAdd, value
            )

        def Mod(self, value):
            return self._clone_with_operation(
                pass_desc_pb2.PassDesc.OperationType.kMod, value
            )

        def Size(self):
            return self._clone_with_operation(
                pass_desc_pb2.PassDesc.OperationType.kSize
            )

        def _set_with_condition(self, type, value):
            condition = pass_desc_pb2.PassDesc.AttrCondition()
            self._to_pass_desc_attr(condition.attr)
            condition.type = type
            if isinstance(value, PassDesc.AttrHelper):
                value._to_pass_desc_attr(condition.condition_attr)
            else:
                self._to_op_desc_attr(value, condition.condition_value)
            if self._operation:
                condition.operation.CopyFrom(self._operation)
            self._condition = condition

        def EQ(self, value):
            self._set_with_condition(
                pass_desc_pb2.PassDesc.ConditionType.kEQ, value
            )

        def MappedPattern(
            self, var=None, op=None, index=0, name=None, element_index=None
        ):
            if all([var, op]):
                raise ValueError("Only mapped one of which var or op.")

            def mapped_var(pattern_ops):
                raise NotImplementedError(
                    "Mapping to variable is not implemented."
                )

            def mapped_op(pattern_ops):
                ops = [o for o in pattern_ops if o._type == op]
                if len(ops) <= index:
                    raise ValueError(
                        "Index '{}' of operator '{}' is incorrect.".format(
                            index, op
                        )
                    )
                return PassDesc.AttrHelper(
                    ops[index], name, element_index=element_index
                )

            self._mapped = mapped_op if var is None else mapped_var

    class VarHelper(paddle.static.Variable):
        def __init__(self, *args, **kwargs):
            block = paddle.static.default_main_program().current_block()
            self._var = paddle.static.data(*args, **kwargs)
            self._attrs = {}

        def __getattr__(self, name):
            return getattr(self._var, name)

        def Attr(self, name):
            attr = self._attrs.get(name)
            if attr is None:
                attr = PassDesc.AttrHelper(self, name)
                self._attrs[name] = attr
            return attr

    class OpHelper:
        def __init__(self, type=None):
            self._type = type

        def __getattr__(self, name):
            op = PassDesc.OpHelper(name)
            op.Init()
            return op

        def __call__(self, *args, **kwargs):
            if len(args) > 0:
                raise ValueError(
                    "Each input argument needs to specify a parameter name."
                )
            for (in_name, in_args) in kwargs.items():
                op_input = self._inputs.get(in_name)
                if op_input is None:
                    raise ValueError(
                        "Operator '{}' does not have input named '{}'.".format(
                            self._type, in_name
                        )
                    )
                if isinstance(in_args, (list, tuple)):
                    if len(in_args) == 0:
                        raise ValueError(
                            "Input '{}' of operator '{}' cannot be empty.".format(
                                in_name, self._type
                            )
                        )
                else:
                    in_args = [in_args]
                for in_arg in in_args:
                    if isinstance(in_arg, PassDesc.OpHelper):
                        op_outs = in_arg.Outputs()
                        if len(op_outs) != 1:
                            raise ValueError(
                                "The size of outputs of operator '{}' is not equal 1, please specify one output variable.".format(
                                    in_arg._type
                                )
                            )
                        for op_out in op_outs.values():
                            op_input.extend(op_out)
                    else:
                        op_input.append(in_arg)
                self._desc.set_input(in_name, [i.name for i in op_input])
            block = paddle.static.default_main_program().current_block()
            for out_name, op_output in self._outputs.items():
                op_output_name = unique_name.generate(self._type)
                op_output.append(block.create_var(name=op_output_name))
                self._desc.set_output(out_name, [op_output_name])
            return self

        def Init(self):
            block = paddle.static.default_main_program().current_block()
            self._proto = OpProtoHolder.instance().op_proto_map.get(self._type)
            if self._proto is None:
                raise AttributeError(
                    "type object 'OpHelper' has no attribute '{}'".format(
                        self._type
                    )
                )
            self._index = len(block.ops)
            self._desc = block.desc.append_op()
            self._desc.set_type(self._type)
            self._attrs = {}
            self._inputs = {i.name: [] for i in self._proto.inputs}
            self._outputs = {o.name: [] for o in self._proto.outputs}
            block.ops.append(self)

        def Attr(self, name):
            attr = self._attrs.get(name)
            if attr is None:
                attr = PassDesc.AttrHelper(self, name)
                self._attrs[name] = attr
            return attr

        def SetAttr(self, name, value):
            if isinstance(value, PassDesc.AttrHelper):
                self.Attr(name)._mapped = value
            else:
                self._desc._set_attr(name, value)

        def Output(self, name):
            output = self._outputs.get(name)
            if output is None:
                raise ValueError(
                    "Operator '{}' does not have output named '{}'.".format(
                        self._type, name
                    )
                )
            return output

        def Outputs(self):
            return self._outputs

        def SetOutputs(self, **kwargs):
            for param, arg in kwargs.items():
                if arg is None:
                    self._desc.remove_output(param)
                else:
                    self._desc.set_output(param, [arg.name])

    OP = OpHelper()


def RegisterPass(function=None, input_specs={}):
    """
    The function decorator of Register Pass. Decorator @RegisterPass handles
    the function and register it into a core.Pass instance. Use name of function
    as Pass type.

    Args:
        function (callable): The function with return of callable pair(s) that
            represents the pattern subgraph and the replace subgraph.
        input_specs (dict[str, InputSpec]): Dict of InputSpec to specific the shape/dtype
            information of Tensor. Some operators limit the shape and dtype of datas when
            create subgraph with Paddle APIs. So user need specify InputSpec of data to
            ensure create a correctly subgraph. Of course, this argument is not limited to
            matching subgraph. The default is dict().

    Returns:
        callables: Callable pair(s).

    Examples:
        .. code-block:: python

        import paddle
        from paddle.fluid.ir import RegisterPass

        @RegisterPass
        def multi_add_to_addn():
            def pattern(x, y, z):
                return paddle.add(paddle.add(x, y), z)
            def replace(x, y, z):
                return paddle.add_n([x, y, z])
            return pattern, replace
    """

    def _is_pass_pair(check_pair):
        if isinstance(check_pair, (list, tuple)):
            if len(check_pair) == 2:
                if all(map(inspect.isfunction, check_pair)):
                    return True
        return False

    def decorated(python_func):
        pass_type = python_func.__name__
        signature = inspect.signature(python_func)
        if len(signature.parameters) > 0:
            raise NotImplementedError(
                "Pass function with parameter is not supported now."
            )
        elif len(signature.parameters) == 0:
            pass_pairs = python_func()
            if _is_pass_pair(pass_pairs):
                pass_pairs = [pass_pairs]
            elif not all(map(_is_pass_pair, pass_pairs)):
                raise ValueError(
                    "Return value of Pass function must be (callable, callable)."
                )
            helper = RegisterPassHelper(pass_pairs, pass_type, input_specs)
            core.register_pass(pass_type, helper.SerializeMultiPassDesc)
        return python_func

    if inspect.isfunction(function):
        return decorated(function)

    return decorated
