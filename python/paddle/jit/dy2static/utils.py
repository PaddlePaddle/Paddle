# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import atexit
import builtins
import functools
import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
import textwrap
import types
import weakref
from importlib.machinery import SourceFileLoader
from typing import Any

import numpy as np

import paddle
from paddle.base import backward, core, framework, unique_name
from paddle.base.data_feeder import convert_dtype
from paddle.base.layer_helper import LayerHelper
from paddle.base.wrapped_decorator import signature_safe_contextmanager
from paddle.framework import CUDAPinnedPlace
from paddle.jit.utils import OrderedSet
from paddle.utils import flatten, gast

from .ast_utils import ast_to_source_code

__all__ = []

# Note(Aurelius): Do not forget the dot `.` to distinguish other
# module such as paddlenlp.
PADDLE_MODULE_PREFIX = 'paddle.'

ALREADY_D2S = '__already_d2s'

# NOTE(liym27): Please use `getattr(ast_node, ORIGIN_INFO)` instead of . operation to get the original information of ast node.
ORIGIN_INFO = "Original information of source code for ast node."

DEL_TEMP_DIR = True  # A flag to avoid atexit.register more than once

RE_PYNAME = '[a-zA-Z0-9_]+'
RE_PYMODULE = r'[a-zA-Z0-9_]+\.'

# Assign not support float64, use float32 value as magic number.
RETURN_NO_VALUE_VAR_NAME = "__no_value_return_var"
RETURN_NO_VALUE_MAGIC_NUM = 1.77113e27


NO_SHAPE_VAR_TYPE = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]


def data_layer_not_check(name, shape, dtype='float32'):
    """
    This function creates a Tensor on the global block. The created Tensor
    doesn't check the dtype and the shape of feed data because dygraph input
    data can be various-length. This API is used in translating dygraph into
    static graph.

    Note:
        The default :code:`stop_gradient` attribute of the Tensor created by
        this API is true, which means the gradient won't be passed backward
        through the data Tensor. Set :code:`var.stop_gradient = False` If
        user would like to pass backward gradient.

    Args:
       name (str): The name/alias of the Tensor, see :ref:`api_guide_Name`
           for more details.
       shape (list|tuple): List|Tuple of integers declaring the shape. You can
           set "None" at a dimension to indicate the dimension can be of any
           size. For example, it is useful to set changeable batch size as "None"
       dtype (np.dtype|VarType|str, optional): The type of the data. Supported
           dtype: bool, float16, float32, float64, int8, int16, int32, int64,
           uint8. Default: float32

    Returns:
        Tensor: The global Tensor that gives access to the data.
    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=core.VarDesc.VarType.DENSE_TENSOR,
        stop_gradient=True,
        is_data=True,
        need_check_feed=False,
    )


def create_undefined_variable():
    var = data_layer_not_check(
        unique_name.generate("undefined_var"), [1], "float64"
    )
    var.stop_gradient = False
    # the variable is created in block(0), we append assign in block(0) either.
    helper = LayerHelper('create_undefined_variable', **locals())
    saved_block_ids = helper.main_program.current_block_idx
    helper.main_program.current_block_idx = 0
    paddle.assign(RETURN_NO_VALUE_MAGIC_NUM, var)
    helper.main_program.current_block_idx = saved_block_ids
    return var


class UndefinedVar:
    def __init__(self, name):
        self.name = name

    def check(self):
        raise UnboundLocalError(
            "local variable '{}' should be created before using it."
        )


class Dygraph2StaticException(Exception):
    def __init__(self, message):
        super().__init__(message)


class WeakMethod:
    def __init__(self, fn, instance):
        self.fn = fn
        self.instance = weakref.ref(instance)

    @property
    def __func__(self):
        return self.fn

    @property
    def __self__(self):
        return self.instance()

    def __call__(self, *args, **kwargs):
        if self.__self__ is None:
            raise RuntimeError("The object has been destroyed")
        return self.fn(self.__self__, *args, **kwargs)


def saw(x):
    if isinstance(x, UndefinedVar):
        return x.check()
    else:
        return x


def parse_arg_and_kwargs(function):
    """
    Returns full argument names as list. e.g ['x', 'y', 'z']
    """
    fullargspec = inspect.getfullargspec(function)
    arg_names = fullargspec.args
    if arg_names and 'self' == arg_names[0]:
        arg_names = fullargspec.args[1:]

    # parse default kwargs
    default_kwargs = {}
    default_values = fullargspec.defaults
    if default_values:
        assert len(default_values) <= len(arg_names)
        default_kwarg_names = arg_names[-len(default_values) :]
        default_kwargs = dict(zip(default_kwarg_names, default_values))

    return arg_names, default_kwargs


def parse_varargs_name(function):
    """
    Returns varargs name string of function. e.g: 'input' from `foo(x, *input)`
    """
    fullargspec = inspect.getfullargspec(function)
    varargs = fullargspec.varargs
    return varargs


def type_name(v):
    return type(v).__name__


def make_hashable(x, error_msg=None):
    """
    Makes input `x` hashable.

    For some unhashable objects, such as `dict/list/set/np.ndarray`,applying hash function by using their values.
    """
    if isinstance(x, (tuple, list, set)):
        return tuple(map(make_hashable, x))

    try:
        hash(x)
    except TypeError:
        if isinstance(x, np.ndarray):
            # Note: `tostring()` will return the binary data from np.ndarray that
            # means different value will lead to different hash code.
            return hash(x.tostring())
        elif isinstance(x, dict):
            return tuple(map(make_hashable, x.values()))

        error_msg = error_msg or "Requires a hashable object."
        raise ValueError(f"{error_msg} But received type: {type_name(x)}")

    return x


# NOTE(Aurelius84): Consider the following paddle inner API as common case to
# apply @to_static code transformation as usual. Because they contains
# user-defined layer, like paddle.distributed.auto_parallel.helper.ProxyLayer.
AS_NOT_INNER_FUNC_LIST = {"paddle.nn.layer.container.Sequential.forward"}


def as_not_paddle_func(path):
    """
    Append API or class as ignored case for is_paddle_func, and they
    will be returned False while calling is_paddle_func(func).
    """
    global AS_NOT_INNER_FUNC_LIST
    AS_NOT_INNER_FUNC_LIST.add(path)


def is_paddle_func(func, ignore_white_list=True):
    """
    Return True if function is defined in Paddle module.
    Skip to check APIs in white list if specifying ignore_white_list as True.
    """

    def in_white_list(module, func_name):
        if func_name is None:
            return False
        return (module.__name__ + '.' + func_name) in AS_NOT_INNER_FUNC_LIST

    try:
        if isinstance(func, paddle.nn.Layer):
            func = func.forward
        if isinstance(
            func, paddle.jit.dy2static.program_translator.StaticFunction
        ):
            func = func.dygraph_function
        if isinstance(func, functools.partial):
            func = func.func
        if inspect.ismethod(func):
            func = func.__func__
        func_name = getattr(func, '__name__', None)
        if inspect.ismethod(func) or inspect.isfunction(func):
            func_name = func.__qualname__

        m = inspect.getmodule(func)
        flag = m is not None and m.__name__.startswith(PADDLE_MODULE_PREFIX)
        if ignore_white_list:
            flag = flag and not in_white_list(m, func_name)

        return flag
    except Exception:
        return False


def get_temp_dir():
    """
    Return @to_static temp directory.
    """
    dir_name = f"paddle/to_static_tmp/{os.getpid()}"
    temp_dir = os.path.join(os.path.expanduser('~/.cache'), dir_name)
    is_windows = sys.platform.startswith('win')
    if is_windows:
        temp_dir = os.path.normpath(temp_dir)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    return temp_dir


def wrap_as_closure(tree: gast.AST, closure_vars: list[str]) -> gast.AST:
    """
    Wrap a function to a closure function.

    Before:

        >>> def fn(x):
        ...     ...

    After:

        >>> def create_fn():
        ...     closure_var_1 = None
        ...     def fn(x):
        ...         ...
        ...     return fn
        ... fn = create_fn()
    """

    def create_assign_node(name, value) -> gast.Assign:
        return gast.Assign(
            targets=[
                gast.Name(
                    id=name,
                    ctx=gast.Store(),
                    annotation=[],
                    type_comment=[],
                )
            ],
            value=value,
            type_comment=None,
        )

    def create_wrppper_fn_def_node(name, body) -> gast.FunctionDef:
        return gast.FunctionDef(
            name=name,
            args=gast.arguments(
                args=[],
                posonlyargs=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
            type_params=[],
        )

    if not isinstance(tree, gast.Module):
        return tree
    if len(tree.body) != 1:
        return tree
    if not isinstance(tree.body[0], gast.FunctionDef):
        return tree
    fn_node = tree.body[0]
    fn_name = fn_node.name
    wrapper_fn_name = f"create_{fn_name}"
    wrapper_fn_def_node = create_wrppper_fn_def_node(
        wrapper_fn_name,
        [
            *[
                create_assign_node(var, gast.Constant(value=None, kind=None))
                for var in closure_vars
            ],
            fn_node,
            gast.Return(
                value=gast.Name(
                    id=fn_name, ctx=gast.Load(), annotation=[], type_comment=[]
                )
            ),
        ],
    )

    assign_node = create_assign_node(
        fn_name,
        gast.Call(
            func=gast.Name(
                id=wrapper_fn_name,
                ctx=gast.Load(),
                annotation=[],
                type_comment=[],
            ),
            args=[],
            keywords=[],
        ),
    )
    return gast.Module(body=[wrapper_fn_def_node, assign_node], type_ignores=[])


def wrap_cell(var: Any) -> types.CellType:
    def closure_fn():
        return var

    assert closure_fn.__closure__ is not None
    return closure_fn.__closure__[0]


def ast_to_func(ast_root, dyfunc, delete_on_exit=True):
    """
    Transform modified AST of decorated function into python callable object.
    TODO: If only decorate one of inner function instead of decorating the main
    function, the other inner functions are invisible for the decorated function.
    """

    def remove_if_exit(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    def func_prefix(func):
        prefix = func.__name__
        if hasattr(func, '__self__'):
            try:
                prefix = f"{func.__self__.__class__.__name__}_{func.__name__}"
            except:
                pass
        return prefix

    def get_new_closure(original_fn, generated_fn):
        if generated_fn.__closure__ is None:
            return None

        original_closure_vars = inspect.getclosurevars(original_fn).nonlocals
        generated_closure_vars = inspect.getclosurevars(generated_fn).nonlocals
        # NOTE(SigureMo): [Why not `assert original_fn.__closure__ is not None`?]
        # If the original function is a recursive function, the original function will
        # not capture itself as a free var, it will access itself from global. But the
        # transformed code always inside a create_xxx function, so the generated function
        # will capture itself as a free var.
        return tuple(
            wrap_cell(original_closure_vars.get(freevar_name, freevar))
            for freevar_name, freevar in generated_closure_vars.items()
        )

    def get_new_globals(original_fn, generated_fn):
        globals_attr_name = "__globals__"
        original_fn_globals = getattr(original_fn, globals_attr_name, {})
        generated_fn_globals = getattr(generated_fn, globals_attr_name, {})

        original_fn_globals_exclude_builtin = {
            k: v
            for k, v in original_fn_globals.items()
            if not (k.startswith('__') and k.endswith('__'))
        }
        return {**generated_fn_globals, **original_fn_globals_exclude_builtin}

    dyfunc_closures = inspect.getclosurevars(dyfunc).nonlocals
    ast_root = wrap_as_closure(ast_root, list(dyfunc_closures.keys()))

    source = ast_to_source_code(ast_root)
    source = _inject_import_statements() + source

    temp_dir = get_temp_dir()
    f = tempfile.NamedTemporaryFile(
        mode='w',
        prefix=func_prefix(dyfunc),
        suffix='.py',
        delete=False,
        dir=temp_dir,
        encoding='utf-8',
    )
    with f:
        module_name = os.path.basename(f.name[:-3])
        f.write(source)

    global DEL_TEMP_DIR
    if delete_on_exit and DEL_TEMP_DIR:
        # Clear temporary files in TEMP_DIR while exiting Python process
        atexit.register(remove_if_exit, dir_path=temp_dir)
        DEL_TEMP_DIR = False

    func_name = dyfunc.__name__
    loader = SourceFileLoader(module_name, f.name)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    # The 'forward' or 'another_forward' of 'TranslatedLayer' cannot be obtained
    # through 'func_name'. So set the special function name '__i_m_p_l__'.
    if hasattr(module, '__i_m_p_l__'):
        callable_func = module.__i_m_p_l__
        callable_func.__name__ = func_name
    elif hasattr(module, func_name):
        callable_func = getattr(module, func_name)
    else:
        raise ValueError(
            f'Function: {func_name} doesn\'t exist in the Module transformed from AST.'
        )
    # After transform dygraph function into callable_func saved in tmp file,
    # it lost the global and closure variables from imported statements or defined
    # in source file. Recovers the necessary variables by `__globals__` and `__closure__`
    new_fn = types.FunctionType(
        code=callable_func.__code__,
        globals=get_new_globals(dyfunc, callable_func),
        name=func_name,
        argdefs=callable_func.__defaults__,
        closure=get_new_closure(dyfunc, callable_func),
    )

    return new_fn, f.name


def _inject_import_statements():
    import_statements = [
        "import paddle",
        "from paddle import Tensor",
        "import paddle.base as base",
        "import paddle.jit.dy2static as _jst",
        "from typing import *",
        "import numpy as np",
        "import warnings",
        "warnings.filterwarnings('ignore', category=DeprecationWarning)",
    ]
    return '\n'.join(import_statements) + '\n'


def func_to_source_code(function, dedent=True):
    """
    Transforms function into raw string of source code.
    """
    if isinstance(function, functools.partial):
        function = function.func
    if isinstance(function, WeakMethod):
        function = function.__func__
    if not (inspect.isfunction(function) or inspect.ismethod(function)):
        raise TypeError(
            f"The type of 'function' should be a function or method, but received {type(function).__name__}."
        )

    source_code_list, _ = inspect.getsourcelines(function)
    # Replace comments with blank lines so that error messages are not misplaced
    source_code_list = [
        line if not line.lstrip().startswith('#') else '\n'
        for line in source_code_list
    ]
    source_code = ''.join(source_code_list)

    if dedent:
        source_code = textwrap.dedent(source_code)

    return source_code


def input_specs_compatible(src_input_specs, desired_input_specs):
    """
    Returns True if the two input specs are compatible, otherwise False.

    args:
        src_input_spec (list or tuple[InputSpec et.al]): list/tuple of
            paddle.static.InputSpec or int/str et.al
        desired_input_specs (list or tuple[InputSpec et.al]): list/tuple of
            paddle.static.InputSpec or int/str et.al
    """
    len_specs = len(src_input_specs)
    if len_specs != len(desired_input_specs):
        # NOTE(chenweihang): if the input_spec of jit.save is a subset of
        # input_spec of to_static, also compatible
        for spec in src_input_specs:
            if spec not in desired_input_specs:
                return False
    else:
        for src_spec, desired_spec in zip(src_input_specs, desired_input_specs):
            if isinstance(src_spec, paddle.static.InputSpec) or isinstance(
                desired_spec, paddle.static.InputSpec
            ):
                if not _compatible_tensor_spec(src_spec, desired_spec):
                    return False
            else:
                if not _compatible_non_tensor_spec(src_spec, desired_spec):
                    return False

    return True


def _compatible_tensor_spec(src_spec, desired_spec):
    """
    Check whether two tensor type spec is compatible.
    """
    for spec in [src_spec, desired_spec]:
        if not isinstance(spec, paddle.static.InputSpec):
            return False
    src_shape = src_spec.shape
    other_shape = desired_spec.shape
    len_shape = len(src_shape)
    if len_shape != len(other_shape):
        return False
    for j in range(len_shape):
        if src_shape[j] is None or src_shape[j] < 0:
            continue
        if other_shape[j] is None or other_shape[j] < 0:
            continue
        if src_shape[j] != other_shape[j]:
            return False

    src_dtype = convert_dtype(src_spec.dtype)
    other_dtype = convert_dtype(desired_spec.dtype)
    if src_dtype != other_dtype:
        return False

    return True


def _compatible_non_tensor_spec(src_spec, desired_spec):
    """
    Check whether two non-tensor type spec is compatible.
    """

    def hash_value(spec):
        try:
            hash_val = make_hashable(spec)
        except:
            hash_val = None
        return hash_val

    src_hash_val = hash_value(src_spec)
    desired_hash_val = hash_value(desired_spec)

    if src_hash_val != desired_hash_val:
        return False
    else:
        return True


class GetterSetterHelper:
    """we have two classes of names in setter and getter function:
    w_vars(loop_vars) + push_pop_vars
    To simplify the setter logic in convert_while and convert_cond,
    we extract the helper class here.
    """

    def __init__(self, getter_func, setter_func, *name_lists):
        name_lists = ([] if x is None else x for x in name_lists)
        name_sets = (OrderedSet(x) for x in name_lists)
        self._union = list(
            functools.reduce(lambda x, y: x | y, name_sets, OrderedSet())
        )
        self._union.sort()
        self.getter = getter_func
        self.setter = setter_func
        self.name2id = {name: idx for idx, name in enumerate(self._union)}

    def union(self):
        return self._union

    def get(self, names):
        if names is None:
            names = []
        vars = self.getter()
        if vars is None:
            return ()
        for n in names:
            assert (
                n in self.name2id
            ), f"the name `{n}` not in name union set`{self.name2id.keys()}`."
        return tuple(vars[self.name2id[n]] for n in names)

    def set(self, names, values):
        if names is None:
            names = []
        if values is None:
            values = []
        vars = self.getter()
        if vars is None:
            return
        for n in names:
            assert (
                n in self.name2id
            ), f"the name `{n}` not in name union set`{self.name2id.keys()}`."
        vars = list(vars)
        indices = [self.name2id[n] for n in names]
        for i, v in zip(indices, values):
            vars[i] = v
        self.setter(vars)


def prim_or_cinn_is_enabled(build_strategy, backend):
    return cinn_is_enabled(build_strategy, backend) or prim_is_enabled()


def cinn_is_enabled(build_strategy, backend):
    if backend == 'CINN':
        return True
    if build_strategy is not None and build_strategy.build_cinn_pass:
        return True

    value = os.getenv('FLAGS_use_cinn')
    if value is not None and value.lower() in ['true', '1']:
        return True
    return False


def cse_is_enabled():
    return paddle.get_flags(["FLAGS_enable_cse_in_dy2st"])[
        "FLAGS_enable_cse_in_dy2st"
    ]


def prim_is_enabled():
    core.check_and_set_prim_all_enabled(True)
    return core._is_bwd_prim_enabled() or core._is_fwd_prim_enabled()


def is_api_in_module_helper(obj, module_prefix):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)


def auto_layout_is_enabled():
    return paddle.get_flags(["FLAGS_enable_auto_layout_pass"])[
        "FLAGS_enable_auto_layout_pass"
    ]


def is_builtin(func, name=None):
    """predict whether a function is a builtin function with name={name}.
    if name == None, then any builtin function will return True
    """

    def name_judge():
        return name is None or func.__name__ == name

    if isinstance(func, types.BuiltinFunctionType) and name_judge():
        return True
    elif func in builtins.__dict__.values() and name_judge():
        return True
    else:
        return False


@signature_safe_contextmanager
def backend_guard(backend):
    core.check_and_set_prim_all_enabled()
    origin_fwd = core._is_fwd_prim_enabled()
    origin_bwd = core._is_bwd_prim_enabled()

    if backend == 'CINN':
        core._set_prim_all_enabled(True)
    try:
        yield
    finally:
        core._set_prim_forward_enabled(origin_fwd)
        core._set_prim_backward_enabled(origin_bwd)


def construct_grad_names(grad_info_map, x_vars, param_vars, out_vars):
    grad_var_names = {}
    fn = lambda grad_var: (
        grad_var.name
        if isinstance(grad_var, framework.Variable)
        else framework.EMPTY_VAR_NAME
    )
    x_grad_vars = backward._get_grad_vars(grad_info_map, x_vars)
    grad_var_names['x'] = list(map(fn, x_grad_vars))
    param_grad_vars = backward._get_grad_vars(grad_info_map, param_vars)
    grad_var_names['param'] = list(map(fn, param_grad_vars))
    out_grad_vars = backward._get_grad_vars(grad_info_map, out_vars)
    grad_var_names['out'] = list(map(fn, out_grad_vars))
    return grad_var_names


@signature_safe_contextmanager
def tensor_name_guard(tensors, names):
    try:
        assert len(tensors) == len(names)
        origin_names = [t.name for t in tensors]
        for t, name in zip(tensors, names):
            t.name = name
        yield
    finally:
        for t, name in zip(tensors, origin_names):
            t.name = name


def cuda_pinned_tensors_move_to_excepted_place(inputs):
    if paddle.is_compiled_with_cuda():
        expected_place = framework._current_expected_place()
        cuda_pinned_place = CUDAPinnedPlace()

        for value in flatten(inputs):
            if (
                isinstance(value, core.eager.Tensor)
                and value.stop_gradient
                and value.place._equals(cuda_pinned_place)
            ):
                var = value._copy_to(expected_place, True)
                var.stop_gradient = True
                var._share_buffer_to(value)
