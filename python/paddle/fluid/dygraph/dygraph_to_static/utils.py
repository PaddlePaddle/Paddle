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

from __future__ import print_function

import ast
import astor
import atexit
import copy
import collections
from paddle.utils import gast
import inspect
import os
import six
import tempfile
import textwrap
import numpy as np

import paddle
from paddle.fluid import unique_name
from paddle.fluid.data_feeder import convert_dtype
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import assign

# Note(Aurelius): Do not forget the dot `.` to distinguish other
# module such as paddlenlp.
PADDLE_MODULE_PREFIX = 'paddle.'
DYGRAPH_MODULE_PREFIX = 'paddle.fluid.dygraph'
DYGRAPH_TO_STATIC_MODULE_PREFIX = 'paddle.fluid.dygraph.dygraph_to_static'
GET_ARGS_FUNC_PREFIX = 'get_args'
SET_ARGS_FUNC_PREFIX = 'set_args'
ARGS_NAME = '__args'
# NOTE(liym27): Please use `getattr(ast_node, ORIGI_INFO)` instead of . operation to get the original information of ast node.
ORIGI_INFO = "Original information of source code for ast node."


class BaseNodeVisitor(gast.NodeVisitor):
    """
    Implement customized NodeVisitor inherited from gast.NodeVisitor. 
    Ancestor nodes are traced to easily support more operations of currently
    visited node.
    """

    def __init__(self):
        self.ancestor_nodes = []

    def visit(self, node):
        """Visit a node."""
        self.ancestor_nodes.append(node)

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()
        return ret


# imp is deprecated in python3
from importlib.machinery import SourceFileLoader

dygraph_class_to_static_api = {
    "CosineDecay": "cosine_decay",
    "ExponentialDecay": "exponential_decay",
    "InverseTimeDecay": "inverse_time_decay",
    "NaturalExpDecay": "natural_exp_decay",
    "NoamDecay": "noam_decay",
    "PiecewiseDecay": "piecewise_decay",
    "PolynomialDecay": "polynomial_decay",
}

FOR_ITER_INDEX_PREFIX = '__for_loop_var_index'
FOR_ITER_TUPLE_PREFIX = '__for_loop_iter_tuple'
FOR_ITER_TARGET_PREFIX = '__for_loop_iter_target'
FOR_ITER_ITERATOR_PREFIX = '__for_loop_iter_iterator'
FOR_ITER_TUPLE_INDEX_PREFIX = '__for_loop_iter_tuple_index'
FOR_ITER_VAR_LEN_PREFIX = '__for_loop_var_len'
FOR_ITER_VAR_NAME_PREFIX = '__for_loop_iter_var'
FOR_ITER_ZIP_TO_LIST_PREFIX = '__for_loop_iter_zip'

# FullArgSpec is valid from Python3. Defined a Namedtuple to
# to make it available in Python2.
FullArgSpec = collections.namedtuple('FullArgSpec', [
    'args', 'varargs', 'varkw', 'defaults', 'kwonlyargs', 'kwonlydefaults',
    'annotations'
])


def data_layer_not_check(name, shape, dtype='float32', lod_level=0):
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
       lod_level (int, optional): The LoD level of the LoDTensor. Usually users
           don't have to set this value. For more details about when and how to
           use LoD level, see :ref:`user_guide_lod_tensor` . Default: 0

    Returns:
        Tensor: The global Tensor that gives access to the data.
    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in six.moves.range(len(shape)):
        if shape[i] is None:
            shape[i] = -1

    return helper.create_global_variable(name=name,
                                         shape=shape,
                                         dtype=dtype,
                                         type=core.VarDesc.VarType.LOD_TENSOR,
                                         stop_gradient=True,
                                         lod_level=lod_level,
                                         is_data=True,
                                         need_check_feed=False)


def create_undefined_var_like(variable):
    """ create a undefined var with the same shape and dtype like varaible.
    """
    from paddle.fluid.dygraph.dygraph_to_static.return_transformer import RETURN_NO_VALUE_MAGIC_NUM
    var = data_layer_not_check(unique_name.generate("undefined_var"),
                               variable.shape, variable.dtype)
    assign(RETURN_NO_VALUE_MAGIC_NUM, var)
    return var


def create_undefined_variable():
    from paddle.fluid.dygraph.dygraph_to_static.return_transformer import RETURN_NO_VALUE_MAGIC_NUM
    var = data_layer_not_check(unique_name.generate("undefined_var"), [1],
                               "float64")
    # the variable is created in block(0), we append assign in block(0) either.
    helper = LayerHelper('create_undefined_variable', **locals())
    saved_block_ids = helper.main_program.current_block_idx
    helper.main_program.current_block_idx = 0
    assign(RETURN_NO_VALUE_MAGIC_NUM, var)
    helper.main_program.current_block_idx = saved_block_ids
    return var


class UndefinedVar:

    def __init__(self, name):
        self.name = name

    def check(self):
        raise UnboundLocalError(
            "local variable '{}' should be created before using it.")


class Dygraph2StaticException(Exception):

    def __init__(self, message):
        super().__init__(message)


def saw(x):
    if isinstance(x, UndefinedVar):
        return x.check()
    else:
        return x


def getfullargspec(target):
    if hasattr(inspect, "getfullargspec"):
        return inspect.getfullargspec(target)
    else:
        argspec = inspect.getargspec(target)
        return FullArgSpec(args=argspec.args,
                           varargs=argspec.varargs,
                           varkw=argspec.keywords,
                           defaults=argspec.defaults,
                           kwonlyargs=[],
                           kwonlydefaults=None,
                           annotations={})


def parse_arg_and_kwargs(function):
    """
    Returns full argument names as list. e.g ['x', 'y', 'z']
    """
    fullargspec = getfullargspec(function)
    arg_names = fullargspec.args
    if arg_names and 'self' == arg_names[0]:
        arg_names = fullargspec.args[1:]

    # parse default kwargs
    default_kwargs = {}
    default_values = fullargspec.defaults
    if default_values:
        assert len(default_values) <= len(arg_names)
        default_kwarg_names = arg_names[-len(default_values):]
        default_kwargs = dict(zip(default_kwarg_names, default_values))

    return arg_names, default_kwargs


def parse_varargs_name(function):
    """
    Returns varargs name string of function. e.g: 'input' from `foo(x, *input)`
    """
    fullargspec = getfullargspec(function)
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
        raise ValueError(error_msg + " But received type: %s" % type_name(x))

    return x


def _is_api_in_module_helper(obj, module_prefix):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)


def is_api_in_module(node, module_prefix):
    assert isinstance(node, gast.Call), "Input non-Call node for is_dygraph_api"

    # Python can have gast.Call as function, for example: covert_call(func)(x)
    # We only check the most outside function
    func_node = node.func
    while isinstance(func_node, gast.Call):
        func_node = func_node.func

    func_str = astor.to_source(gast.gast_to_ast(func_node)).strip()
    try:
        # TODO(liym27):
        #  Consider a better to import modules like:
        #  source_file = inspect.getfile(dyfunc)
        #  import_statements = ImportVisitor(source_file).transform()
        #  import_str = "".join(import_statements)
        import paddle
        import paddle.fluid as fluid
        import paddle.fluid.dygraph as dygraph
        import paddle.fluid.layers as layers
        import paddle.jit.dy2static as _jst

        from paddle.fluid.dygraph import to_variable
        from paddle import to_tensor

        return eval("_is_api_in_module_helper({}, '{}')".format(
            func_str, module_prefix))
    except Exception:
        return False


def is_dygraph_api(node):

    # Note: A api in module dygraph_to_static is not a real dygraph api.
    if is_api_in_module(node, DYGRAPH_TO_STATIC_MODULE_PREFIX):
        return False

    # TODO(liym27): A better way to determine whether it is a dygraph api.
    #  Consider the decorator @dygraph_only
    return is_api_in_module(node, DYGRAPH_MODULE_PREFIX)


def is_paddle_api(node):
    return is_api_in_module(node, PADDLE_MODULE_PREFIX)


def is_paddle_func(func):
    m = inspect.getmodule(func)
    return m is not None and m.__name__.startswith(PADDLE_MODULE_PREFIX)


# Is numpy_api cannot reuse is_api_in_module because of numpy module problem
def is_numpy_api(node):
    assert isinstance(node, gast.Call), "Input non-Call node for is_numpy_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        import numpy as np
        module_result = eval("_is_api_in_module_helper({}, '{}')".format(
            func_str, "numpy"))
        # BUG: np.random.uniform doesn't have module and cannot be analyzed
        # TODO: find a better way
        if not module_result:
            return func_str.startswith("numpy.") or func_str.startswith("np.")
    except Exception:
        return False


def is_control_flow_to_transform(node,
                                 static_analysis_visitor=None,
                                 var_name_to_type=None):
    """
    Determines whether the node is a PaddlePaddle control flow statement which needs to
    be transformed into a static graph control flow statement.
    """
    assert isinstance(node, gast.AST), \
        "The type of input node must be gast.AST, but received %s." % type(node)
    visitor = IsControlFlowVisitor(node,
                                   static_analysis_visitor,
                                   node_var_type_map=var_name_to_type)
    need_to_transform = visitor.transform()
    return need_to_transform


def _delete_keywords_from(node):
    assert isinstance(node, gast.Call)
    func_src = astor.to_source(gast.gast_to_ast(node.func))
    import paddle.fluid as fluid
    full_args = eval("inspect.getargspec({})".format(func_src))
    full_args_name = full_args[0]

    node.keywords = [k for k in node.keywords if k.arg in full_args_name]
    return


def to_static_api(dygraph_class):
    if dygraph_class in dygraph_class_to_static_api:
        return dygraph_class_to_static_api[dygraph_class]
    else:
        raise NotImplementedError(
            "Paddle dygraph API {} cannot be converted "
            "to static graph at present.".format(dygraph_class))


def _add_keywords_to(node, dygraph_api_name):
    assert isinstance(node, gast.Call)
    if dygraph_api_name == "Linear":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"

        node.keywords.append(
            gast.keyword(arg="num_flatten_dims",
                         value=gast.Constant(value=-1, kind=None)))

    if dygraph_api_name == "BilinearTensorProduct":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"

    if dygraph_api_name == "PRelu":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "input":
                ast_keyword.arg = "x"
    return


def to_static_ast(node, class_node):
    assert isinstance(node, gast.Call)
    assert isinstance(class_node, gast.Call)
    static_api = to_static_api(class_node.func.attr)

    node.func = gast.Attribute(attr=static_api,
                               ctx=gast.Load(),
                               value=gast.Attribute(attr='layers',
                                                    ctx=gast.Load(),
                                                    value=gast.Name(
                                                        ctx=gast.Load(),
                                                        id='fluid',
                                                        annotation=None,
                                                        type_comment=None)))

    update_args_of_func(node, class_node, 'forward')

    node.args.extend(class_node.args)
    node.keywords.extend(class_node.keywords)
    _add_keywords_to(node, class_node.func.attr)
    _delete_keywords_from(node)

    gast.fix_missing_locations(node)

    return node


def update_args_of_func(node, dygraph_node, method_name):
    assert isinstance(node, gast.Call)
    if method_name not in ["__init__", "forward"]:
        raise ValueError(
            "The method name of class to update args should be '__init__' or 'forward'"
        )

    class_src = astor.to_source(gast.gast_to_ast(dygraph_node.func))
    import paddle.fluid as fluid
    if method_name == "__init__" or eval(
            "issubclass({}, fluid.dygraph.Layer)".format(class_src)):
        full_args = eval("inspect.getargspec({}.{})".format(
            class_src, method_name))
        full_args_name = [
            arg_name for arg_name in full_args[0] if arg_name != "self"
        ]
    else:
        full_args_name = []
    added_keywords = []
    for idx, arg in enumerate(node.args):
        added_keywords.append(gast.keyword(arg=full_args_name[idx], value=arg))

    node.args = []
    node.keywords = added_keywords + node.keywords


def create_api_shape_node(tensor_shape_node):
    assert isinstance(tensor_shape_node,
                      (gast.Name, gast.Attribute, gast.Subscript))

    if isinstance(tensor_shape_node, gast.Name):
        api_shape_node = gast.Call(
            func=gast.parse('paddle.shape').body[0].value,
            args=[tensor_shape_node],
            keywords=[])
        return api_shape_node

    if isinstance(tensor_shape_node, gast.Attribute):
        api_shape_node = gast.Call(
            func=gast.parse('paddle.shape').body[0].value,
            args=[tensor_shape_node.value],
            keywords=[])
        return api_shape_node

    if isinstance(tensor_shape_node, gast.Subscript):
        result_node = copy.deepcopy(tensor_shape_node)
        result_node.value = create_api_shape_node(result_node.value)
        return result_node


def get_constant_variable_node(name, value, shape=[1], dtype='int64'):
    return gast.parse('%s = paddle.full(%s, "%s", %s)' %
                      (name, str(shape), str(value), dtype))


def get_attribute_full_name(node):
    assert isinstance(
        node,
        gast.Attribute), "Input non-Attribute node to get attribute full name"
    return astor.to_source(gast.gast_to_ast(node)).strip()


def generate_name_node(name_ids, ctx=gast.Load(), gen_tuple_if_single=False):
    """
    If name_ids is list or tuple or set with multiple strings, this function
    generates gast.Tuple of gast.Name.
    If the name_ids is single string or contains only 1 string, this function
    returns gast.Name if gen_tuple_if_single==False else returns gast.Tuple
    with only one gast.Name

    This function is used at several gast.Return statements.
    """
    if isinstance(name_ids, six.string_types):
        name_ids = [name_ids]
    if not isinstance(name_ids, (list, tuple, set)):
        raise TypeError(
            'name_ids must be list or tuple or set, but received %s' %
            type(type(name_ids)))

    def create_node_for_name(name):
        if '.' not in name:
            return gast.Name(id=name,
                             ctx=ctx,
                             annotation=None,
                             type_comment=None)
        return gast.parse(name).body[0].value

    gast_names = [create_node_for_name(name_id) for name_id in name_ids]
    if len(gast_names) == 1 and not gen_tuple_if_single:
        name_node = gast_names[0]
    else:
        name_node = gast.Tuple(elts=gast_names, ctx=ctx)
    return name_node


def create_funcDef_node(nodes, name, input_args, return_name_ids):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    # add return statement
    if return_name_ids:
        nodes.append(gast.Return(value=generate_name_node(return_name_ids)))
    else:
        nodes.append(gast.Return(value=None))
    func_def_node = gast.FunctionDef(name=name,
                                     args=input_args,
                                     body=nodes,
                                     decorator_list=[],
                                     returns=None,
                                     type_comment=None)
    return func_def_node


def index_in_list(array_list, item):
    try:
        return array_list.index(item)
    except ValueError:
        # Item not in array_list
        return -1


def create_assign_node(name, node):
    """
    Creates a `gast.Assign` node by given name_id as target and node as value.
    """
    targets = generate_name_node(name, ctx=gast.Store())
    assign_node = gast.Assign(targets=[targets], value=node)
    return targets, assign_node


def ast_to_func(ast_root, dyfunc, delete_on_exit=True):
    """
    Transform modified AST of decorated function into python callable object.
    TODO: If only decorate one of inner function instead of decorating the main
    function, the other inner functions are invisible for the decorated function.
    """

    def remove_if_exit(filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    source = ast_to_source_code(ast_root)
    source = _inject_import_statements() + source

    f = tempfile.NamedTemporaryFile(mode='w',
                                    suffix='.py',
                                    delete=False,
                                    encoding='utf-8')
    with f:
        module_name = os.path.basename(f.name[:-3])
        f.write(source)

    if delete_on_exit:
        atexit.register(lambda: remove_if_exit(f.name))
        atexit.register(lambda: remove_if_exit(f.name[:-3] + ".pyc"))

    module = SourceFileLoader(module_name, f.name).load_module()
    func_name = dyfunc.__name__
    # The 'forward' or 'another_forward' of 'TranslatedLayer' cannot be obtained
    # through 'func_name'. So set the special function name '__i_m_p_l__'.
    if hasattr(module, '__i_m_p_l__'):
        callable_func = getattr(module, '__i_m_p_l__')
        callable_func.__name__ = func_name
    elif hasattr(module, func_name):
        callable_func = getattr(module, func_name)
    else:
        raise ValueError(
            'Function: %s doesn\'t exist in the Module transformed from AST.' %
            func_name)
    # After transform dygraph function into callable_func saved in tmp file,
    # it lost the global variables from imported statements or defined in source file.
    # Recovers the necessary variables by `__globals__`.
    recover_globals_attribute(dyfunc, callable_func)

    return callable_func, f.name


def _inject_import_statements():
    import_statements = [
        "import paddle", "from paddle import Tensor",
        "import paddle.fluid as fluid", "import paddle.jit.dy2static as _jst",
        "from typing import *", "import numpy as np"
    ]
    return '\n'.join(import_statements) + '\n'


def recover_globals_attribute(src_obj, dst_obj):
    attr_name = '__globals__'

    src_globals = getattr(src_obj, attr_name, {})
    dst_globals = getattr(dst_obj, attr_name, {})

    for k, v in six.iteritems(src_globals):
        # ignore builtin attribute.
        if not (k.startswith('__') and k.endswith('__')):
            dst_globals[k] = v


def func_to_source_code(function, dedent=True):
    """
    Transforms function into raw string of source code.
    """
    if not (inspect.isfunction(function) or inspect.ismethod(function)):
        raise TypeError(
            "The type of 'function' should be a function or method, but received {}."
            .format(type(function).__name__))
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


def ast_to_source_code(ast_node):
    """
    Transforms ast node into source code.
    """
    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s." %
            type(ast_node))
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)

    # Do not wrap lines even if they are too long
    def pretty_source(source):
        return ''.join(source)

    source_code = astor.to_source(ast_node, pretty_source=pretty_source)
    return source_code


def is_candidate_node(node):
    """
    Nodes with specified type will be dependent on tensor.
    """
    is_compare_node = isinstance(node, (gast.Compare, gast.BoolOp, gast.UnaryOp,
                                        gast.For, gast.If, gast.While))
    # TODO(Aurelius84): `.numpy()` may be an customized function,
    # and should consider a more elegant way to solve this problem.
    has_numpy_attr = ".numpy()" in ast_to_source_code(node)
    return is_compare_node or has_numpy_attr


def compare_with_none(node):
    """
    Whether the comparator of `gast.Compare` node is `None`.
    """
    if isinstance(node, gast.Compare):
        for child in [node.left, node.comparators]:
            # node.comparators is a list.
            if isinstance(child, list):
                child = child[0]
            if (isinstance(child, gast.Constant)
                    and child.value is None) or (isinstance(child, gast.Name)
                                                 and child.id == 'None'):
                return True
    return False


class IsControlFlowVisitor(gast.NodeVisitor):
    """
    Judge whether the ast_node of control flow from Dygraph code dependent on paddle Tensor.
    `ast_node` can be gast.If, gast.For, gast.While, gast.If.test(gast.Compare, gast.BoolOp, gast.UnaryOp).

    If returns True,
    gast.If.test must meet at least one of the following requirements:
        1. involves at least one var whose type is Tensor.
        2. the Tensor var calls `.numpy()[]` interface or Tensor.shape is [1].
        3. involves Tensor.shape[i] and the shape[i] is unknown in compile time.
    gast.While must meet at least one of the requirements 1 to 5:
        4. has `break` statement.
        5. has `continue` statement.
    gast.For must meet at least one of the requirements 4 to 8:
        6. calls `range` function in `for` statement and the argument of range is Tensor.
        7. calls `enumerate` function in `for` statement and the argument of enumerate is Tensor.
        8. the iterable varaible in `for` statement is Tensor.
        TODO: Support non-range case

    The following examples should not be considered as control_flow_if:
        1. `if Tensor_var` or `if Tensor_var is None`
        2. if Tensor.shape[i] is determined with fixed value (not -1 or None)

    Note: pred in ConditionalBlock require variable, which means all vars should be Tensor
          or transformed into Tensor, like fill_constant(shape=[1], dtype='int32', value=Tensor.shape[i]).

    TODO: 1. need to deal with `tensor.shape[i]` which need to eval the data of shape[i],
             because reshape_op may be called before this statement.
    """

    def __init__(self,
                 ast_node,
                 static_analysis_visitor=None,
                 node_var_type_map=None):
        assert isinstance(
            ast_node, gast.AST
        ), "Type of input node should be gast.AST, but received %s." % type(
            ast_node)
        self.ast_root = ast_node
        if static_analysis_visitor is None:
            from .static_analysis import StaticAnalysisVisitor
            static_analysis_visitor = StaticAnalysisVisitor(ast_node)
        self.static_analysis_visitor = static_analysis_visitor
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        self.node_var_type_map = node_var_type_map

        self.is_control_flow_num = 0
        self._compare_node_tenor_set = set()

    def transform(self):
        node = self.ast_root
        if isinstance(node, gast.If):
            self._visit_If(node)
        elif isinstance(node, gast.For):
            self._visit_For(node)
        elif isinstance(node, gast.While):
            self._visit_While(node)
        else:
            self.visit(node)
        return self.is_control_flow_num > 0

    def _visit_If(self, node):
        assert isinstance(node, gast.If)
        self.visit(node.test)
        return

    def _visit_For(self, node):
        assert isinstance(node, gast.For)
        if isinstance(node.iter, gast.Call):
            # for in range(var[0]|var.numpy()[0]) or for in enumerate(var|var.numpy())
            if isinstance(node.iter.func, gast.Name):
                if node.iter.func.id == "range" or node.iter.func.id == "enumerate":
                    for arg in node.iter.args:
                        self.visit(arg)
                else:
                    return
            # for in var.numpy()
            elif isinstance(node.iter.func, gast.Attribute):
                if node.iter.func.attr == 'numpy':
                    self._visit_Call(node.iter)
                else:
                    return
            else:
                return
        elif isinstance(node.iter, gast.Name):
            # for in var
            self.visit(node.iter)
        else:
            return

        for child_node in gast.walk(node):
            if isinstance(child_node, (gast.Continue, gast.Break)):
                self._visit_break_continue(child_node)
        return

    def _visit_While(self, node):
        assert isinstance(node, gast.While)
        test = node.test
        self.generic_visit(test)
        for child_node in gast.walk(node):
            if isinstance(child_node, (gast.Continue, gast.Break)):
                self._visit_break_continue(child_node)
        return

    def _visit_break_continue(self, node):
        assert isinstance(node, (gast.Break, gast.Continue))
        wrapper_node = self.node_to_wrapper_map.get(node)
        if not wrapper_node:
            # Transformed node is not in node_to_wrapper_map
            return

        while wrapper_node.parent:
            parent_node = wrapper_node.parent.node
            if isinstance(parent_node, (gast.For, gast.While)):
                if parent_node is self.ast_root:
                    self.is_control_flow_num += 1
                    return
                else:
                    return

            wrapper_node = wrapper_node.parent

        return

    def visit_BoolOp(self, node):
        for i, child in enumerate(node.values):
            self.visit(child)
        return node

    def visit_Compare(self, node):
        pre_control_flow_num = self.is_control_flow_num
        if not compare_with_none(node):
            self.generic_visit(node)
            for child in gast.walk(node):
                if isinstance(child, gast.Subscript):
                    self._visit_Subscript(child)
        if self.is_control_flow_num > pre_control_flow_num:
            self._compare_node_tenor_set.add(node)
        return node

    def _visit_Subscript(self, node):
        self.generic_visit(node)
        if hasattr(node, 'value') and isinstance(node.value, gast.Call):
            self._visit_Call(node.value)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.func, gast.Attribute):
            attr_node = node.func
            if attr_node.attr == 'numpy':
                self.is_control_flow_num += 1

    def visit_Call(self, node):
        self._visit_Call(node)
        if is_paddle_api(node):
            self.is_control_flow_num += 1
        return node

    def visit_Name(self, node):
        if self._is_node_with_tensor(node, node.id):
            self.is_control_flow_num += 1
        return node

    def visit_Constant(self, node):
        if self._is_node_with_tensor(node, node.value):
            self.is_control_flow_num += 1
        return node

    def _is_node_with_tensor(self, node, name_id):
        from paddle.fluid.dygraph.dygraph_to_static.static_analysis import NodeVarType

        # Look up the node_var_type_map by name_id.
        if self.node_var_type_map:
            if name_id and isinstance(name_id, six.string_types):
                var_type = self.node_var_type_map.get(name_id, None)
                if var_type and var_type & NodeVarType.TENSOR_TYPES:
                    return True
        # if not found, look up the node_to_wrapper_map by node.
        wrapper_node = self.node_to_wrapper_map.get(node, None)
        if wrapper_node is not None:
            if wrapper_node.node_var_type & NodeVarType.TENSOR_TYPES:
                return True

        return False

    def get_compare_nodes_with_tensor(self):
        return self._compare_node_tenor_set


# NOTE: inspect.unwrap() exits in PY3 but not in PY2.
def unwrap(func):
    """
    Returns the object wrapped by decorators.
    """

    def _is_wrapped(f):
        return hasattr(f, '__wrapped__')

    unwrapped_f = func
    while (_is_wrapped(unwrapped_f)):
        unwrapped_f = unwrapped_f.__wrapped__

    return unwrapped_f


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
        for (src_spec, desired_spec) in zip(src_input_specs,
                                            desired_input_specs):
            if isinstance(src_spec, paddle.static.InputSpec) or isinstance(
                    desired_spec, paddle.static.InputSpec):
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


def slice_is_num(slice_node):
    # A slice_node.slice can be a:
    # (1) ast.Index, which is a simple number such as [1], [-2]
    # (2) ast.Slice, which is represented by bounds such as [2:-1]
    # (3) ast.Tuple, which includes the above two cases such as [2:-1, 1]
    # If slice node is case (1), return True, Otherwise, return False.
    #
    # NOTE: In (1) case, when gast>=0.4.0, gast.Index is not used, which is replaced
    # other gast node such as gast.Constant, gast.Name, gast.UnaryOp and so on.
    # Considering the compatibility of gast, here use ast note to check whether the
    # node is a num. For more details, please visit https://github.com/serge-sans-paille/gast

    assert isinstance(slice_node, gast.Subscript)
    slice_node_str = ast_to_source_code(slice_node).strip()
    ast_node = ast.parse(slice_node_str).body[0].value

    if isinstance(ast_node.slice, (ast.Tuple, ast.Slice)):
        return False

    if isinstance(ast_node.slice, ast.Index):
        return True

    return False


class NameScope:

    def __init__(self):
        """ 
            A NameScope is a object which manager all the variable names. 
            only FunctionDef and Controlflow node will have a namescope property.

            type can be "function" and "controlflow"

            we don't analyze the read only variable because they don't affect the analysis.
        """
        self.globals = set()
        self.nonlocals = set()
        self.args = set()
        self.father = None  # point to the nearest function name scope.
        self.w_vars = set()  # all qualified + normal names been stored
        self.created = set(
        )  # useful for control flow compatibility. may be remove later

    def set_father(self, father):
        self.father = father

    def existed_vars(self):
        """ vars existing in current scope. 
            they must not contain qualified names.
        """
        local_vars = self.w_vars - self.globals - self.nonlocals - self.args
        return set(filter(lambda x: '.' not in x, local_vars))

    def created_vars(self):
        return self.created

    def modified_vars(self):
        # may be globals / non-locals / args / qualified names and created_vars
        return self.w_vars

    def control_flow_vars(self):
        valid_names = self.w_vars
        tmp = self.father.global_vars & valid_names,
        return {"global": tmp, "nonlocal": self.w_vars - tmp}

    def global_vars(self):
        return self.globals

    def merge_from(self, name_scope):
        self.globals |= name_scope.globals
        self.nonlocals |= name_scope.nonlocals
        self.args |= name_scope.args
        self.w_vars |= name_scope.w_vars


class FunctionNameLivenessAnalysis(gast.NodeVisitor):
    """ analyze the liveness of a function.

        every variables stored in this scope will be collected,
        in addition with global/nonlocal information.

        1. global variable is stored in node.var_globals.
        2. nonlocal variable is stored in node.var_nonlocals.
        3. arguments is stored in node.var_args.

        For example:

        def func(*args, **kargs):
            a = 12
            global i,j
            nonlocal x,y
            print(a)
            i = k
            for m in range(10):
                q = 12
        
        After this visitor we have: 
        # node is the FunctionDef node with name: "func"
        node.pd_scope = NameScope(
            globals = ['i', 'j'],
            nonlocals = ['x', 'y'],
            args = ['args', 'kargs'], 
            wr_vars = ['a', 'i', 'q', 'm']
        )
    """

    def __init__(self, root_node):
        self.scope_node_stack = []  # controlflow, functiondef node
        self.visit(root_node)

    def _reset_name_scope(self, node):
        # always reset the node as empty namescope.
        setattr(node, "pd_scope", NameScope())

    def _get_name_scope(self, node):
        if not hasattr(node, "pd_scope"):
            setattr(node, "pd_scope", NameScope())
        return node.pd_scope

    def _current_name_scope(self):
        return self._get_name_scope(self.scope_node_stack[-1])

    def _father_name_scope(self):
        if len(self.scope_node_stack) == 1: return None
        return self._get_name_scope(self.scope_node_stack[-2])

    def _nearest_function_scope(self):
        if len(self.scope_node_stack) == 1: return None
        for node in self.scope_node_stack[-2::-1]:
            if isinstance(node, gast.FunctionDef):
                return self._get_name_scope(node)

    def visit_ListComp(self, node):
        """ [ i for i in range(10) ]
            In this case, `i` will not created in FunctionScope. 
            We don't collect `i` by not calling generic_visit.
        """
        pass

    def visit_DictComp(self, node):
        """ the same as ListComp.
        """
        pass

    def visit_Name(self, node):
        self.generic_visit(node)
        write_context = (gast.Store, gast.AugStore, gast.Del)
        if isinstance(node.ctx, write_context):
            self._current_name_scope().w_vars.add(node.id)

    def visit_FunctionDef(self, node):

        def pre_func():
            self._current_name_scope().args |= set(
                self._get_argument_names(node))

        def post_func():
            """ NOTE: why we need merge w_vars here ? 
                because we do ifelse_transformer after loop_transformer. Loops will changed into functioons. but we know this function will be called in if. so we add w_vars to father function scope.
            """
            from paddle.fluid.dygraph.dygraph_to_static.loop_transformer import WHILE_CONDITION_PREFIX, WHILE_BODY_PREFIX, FOR_CONDITION_PREFIX, FOR_BODY_PREFIX
            from paddle.fluid.dygraph.dygraph_to_static.ifelse_transformer import TRUE_FUNC_PREFIX, FALSE_FUNC_PREFIX
            control_flow_function_def = [
                WHILE_BODY_PREFIX, WHILE_BODY_PREFIX, FOR_CONDITION_PREFIX,
                FOR_BODY_PREFIX, TRUE_FUNC_PREFIX, FALSE_FUNC_PREFIX
            ]

            def is_control_flow_def_node():
                for prefix in control_flow_function_def:
                    if node.name.startswith(prefix): return True
                return False

            if self._father_name_scope() and is_control_flow_def_node():
                self._father_name_scope().w_vars |= self._current_name_scope(
                ).w_vars

        self._visit_scope_node(node, pre_func, post_func)

    def _visit_scope_node(self, node, pre_func, post_func):
        """ scope node main visit logic.
            pre_func and post_func is callbacks
        """
        self._reset_name_scope(node)
        self.scope_node_stack.append(node)
        self._current_name_scope().father = self._nearest_function_scope()
        if pre_func: pre_func()
        self.generic_visit(node)
        if post_func: post_func()
        self.scope_node_stack.pop()

    def _visit_controlflow_node(self, node):

        def post_func():
            self._father_name_scope().merge_from(self._current_name_scope())
            self._nearest_function_scope().merge_from(
                self._current_name_scope())
            self._current_name_scope().created = self._nearest_function_scope(
            ).existed_vars() - node.before_created
            # gather created vars into father and used in CreateUndefinedVarTransform
            self._nearest_function_scope().created |= self._current_name_scope(
            ).created

        def pre_func():
            setattr(node, "before_created",
                    self._nearest_function_scope().existed_vars())

        self._visit_scope_node(node, pre_func, post_func)

    def visit_For(self, node):
        self._visit_controlflow_node(node)

    def visit_While(self, node):
        self._visit_controlflow_node(node)

    def visit_If(self, node):
        self._visit_controlflow_node(node)

    def visit_Global(self, node):
        self._current_name_scope().globals |= set(node.names)

    def visit_Nonlocal(self, node):
        self._current_name_scope().nonlocals |= set(node.names)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        write_context = (gast.Store, gast.AugStore, gast.Del)
        if isinstance(node.ctx, write_context):
            name = ast_to_source_code(node).strip()
            self._current_name_scope().w_vars.add(name)

    def _get_argument_names(self, node):
        """ get all arguments name in the functiondef node.
            this node is local to the function and shouldn't 
            be created.
        """
        assert isinstance(
            node, gast.FunctionDef), "Input node is not function define node"
        names = [a for a in node.args.args]
        names.append(node.args.vararg)
        names.append(node.args.kwarg)
        names = [i.id for i in names if i is not None]
        return names


def create_get_args_node(names):
    """
    Create get_args function as follows:

        def get_args_0():
            nonlocal x, y
            return x, y
    """

    def empty_node():
        func_def = """
        def {func_name}():
            return
        """.format(func_name=unique_name.generate(GET_ARGS_FUNC_PREFIX))
        return gast.parse(textwrap.dedent(func_def)).body[0]

    assert isinstance(names, (list, tuple))
    mapped = list(filter(lambda n: '.' not in n, names))
    nonlocal_names = sorted(
        mapped,
        key=mapped.index)  # to keep the order, we can't use set() to unique
    if not names:
        return empty_node()
    if not nonlocal_names:
        nonlocal_vars = "\n"
    else:
        nonlocal_vars = "nonlocal " + ",".join(nonlocal_names)
    template = """
    def {func_name}():
        {nonlocal_vars}
        return {vars},
    """
    func_def = template.format(
        func_name=unique_name.generate(GET_ARGS_FUNC_PREFIX),
        nonlocal_vars=nonlocal_vars,
        vars=",".join(names))
    return gast.parse(textwrap.dedent(func_def)).body[0]


def create_set_args_node(names):
    """
    Create set_args function as follows:

        def set_args_0(__args):
            nonlocal x, y
            x, y = __args
    """

    def empty_node():
        func_def = """
        def {func_name}({args}):
            pass
        """.format(func_name=unique_name.generate(SET_ARGS_FUNC_PREFIX),
                   args=ARGS_NAME)
        return gast.parse(textwrap.dedent(func_def)).body[0]

    assert isinstance(names, (list, tuple))
    mapped = list(filter(lambda n: '.' not in n, names))
    nonlocal_names = sorted(
        mapped,
        key=mapped.index)  # to keep the order, we can't use set() to unique
    if not names:
        return empty_node()
    if not nonlocal_names:
        nonlocal_vars = "\n"
    else:
        nonlocal_vars = "nonlocal " + ",".join(nonlocal_names)
    template = """
    def {func_name}({args}):
        {nonlocal_vars}
        {vars}, = {args}
    """
    func_def = template.format(
        func_name=unique_name.generate(SET_ARGS_FUNC_PREFIX),
        args=ARGS_NAME,
        nonlocal_vars=nonlocal_vars,
        vars=",".join(names))
    return gast.parse(textwrap.dedent(func_def)).body[0]


def create_nonlocal_stmt_nodes(names):
    assert isinstance(names, (list, tuple))

    mapped = list(filter(lambda n: '.' not in n, names))
    names = sorted(
        mapped,
        key=mapped.index)  # to keep the order, we can't use set() to unique
    if not names:
        return []
    func_code = "nonlocal {}".format(','.join(names))
    return [gast.parse(func_code).body[0]]
