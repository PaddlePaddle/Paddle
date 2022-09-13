#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
import multiprocessing
import sys
import warnings
import numpy as np
from .wrapped_decorator import signature_safe_contextmanager
import six
from .data_feeder import convert_dtype
from .framework import Program, default_main_program, Variable, Operator
from .framework import convert_np_dtype_to_dtype_, _apply_pass

from . import core
from . import unique_name
from . import compiler
from .. import compat as cpt
from .trainer_factory import TrainerFactory
from .trainer_factory import FetchHandlerMonitor
import copy
from . import framework
from .incubate.checkpoint import auto_checkpoint as acp
from .compiler import _prune_feed_ops

from functools import lru_cache

__all__ = ['Executor', 'global_scope', 'scope_guard']

g_scope = core.Scope()
InferNativeConfig = core.NativeConfig
InferAnalysisConfig = core.AnalysisConfig


def global_scope():
    """
    :api_attr: Static Graph

    Get the global/default scope instance. There are a lot of APIs use
    :code:`global_scope` as its default value, e.g., :code:`Executor.run`

    Returns:
        Scope: The global/default scope instance.

    Examples:
        .. code-block:: python

          import paddle
          import numpy

          paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), paddle.CPUPlace())
          numpy.array(paddle.static.global_scope().find_var("data").get_tensor())
    """
    return g_scope


def _switch_scope(scope):
    global g_scope
    ex = g_scope
    g_scope = scope
    return ex


@signature_safe_contextmanager
def scope_guard(scope):
    """
    
    This function switches scope through python `with` statement.
    Scope records the mapping between variable names and variables ( :ref:`api_guide_Variable` ),
    similar to brackets in programming languages.
    If this function is not invoked, all variables and variable names are recorded in the default global scope.
    When users need to create variables with the same name,
    they need to switch scopes through this function
    if they do not want the mapping of variables with the same name to be overwritten.
    After switching through the `with` statement,
    all variables created in the `with` block will be assigned to a new scope.

    Parameters:
        scope: The new scope.

    Returns:
        None

    Examples:
    
        .. code-block:: python

            import paddle
            import numpy
            paddle.enable_static()

            new_scope = paddle.static.Scope()
            with paddle.static.scope_guard(new_scope):
                 paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), paddle.CPUPlace())
            numpy.array(new_scope.find_var("data").get_tensor())
    """

    ex = _switch_scope(scope)
    try:
        yield
    finally:
        _switch_scope(ex)


def as_numpy(tensor, copy=False):
    """
    Convert a Tensor to a numpy.ndarray, its only support Tensor without LoD information.
    For higher dimensional sequence data, please use LoDTensor directly.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          new_scope = fluid.Scope()
          with fluid.scope_guard(new_scope):
              fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
          tensor = new_scope.find_var("data").get_tensor()
          fluid.executor.as_numpy(tensor) # or numpy.array(new_scope.find_var("data").get_tensor())

    Args:
       tensor(Variable): a instance of Tensor
       copy(bool, optional): Whether to use deep copy.

    Returns:
        numpy.ndarray
    """
    if isinstance(tensor, core.LoDTensorArray):
        return [as_numpy(t, copy) for t in tensor]
    if isinstance(tensor, list):
        return [as_numpy(t, copy) for t in tensor]
    assert isinstance(tensor, core.LoDTensor)
    lod = tensor.lod()
    if len(lod) > 0:
        raise RuntimeError("Some of your fetched tensors hold LoD information. \
            They can not be completely cast to Python ndarray. \
            Please set the parameter 'return_numpy' as 'False' to \
            return LoDTensor itself directly.")
    if tensor._is_initialized():
        if copy:
            return np.array(tensor)
        else:
            return np.asarray(tensor)
    else:
        return None


def dtype_is_compatible_with(first, second):
    """
    Returns True if the first dtype can be compatible the second one.
    Currently, we require the two dtype's have to be same.
      
    Args:
        dtype (np.dtype|VarType|str): The type of data: float32, int64, etc.
    
    Returns:
        True if the two types are same.
    """
    if not isinstance(first, core.VarDesc.VarType):
        first = convert_np_dtype_to_dtype_(first)
    if not isinstance(second, core.VarDesc.VarType):
        second = convert_np_dtype_to_dtype_(second)
    return first == second


def dimension_is_compatible_with(first, second):
    """
    Returns True if the two dimensions are compatible.

    A dimension is compatible with the other if:
    1. The length of the dimensions are same.
    2. Each non-negative number of the two dimensions are same.
    3. For negative number or 'None' in a dimension, it means unknown so it
       is compatible with any number.

    Args:
        first (list/tuple): integers representing shape. "None" or negative
            number means unknown.
        second (list/tuple): integers representing shape. "None" or negative
            number means unknown.

    Returns:
        True if the two dimensions are compatible.
    """

    dim_len = len(first)
    if dim_len != len(second):
        return False

    for i in range(dim_len):
        if first[i] is None or first[i] < 0:
            continue
        if second[i] is None or second[i] < 0:
            continue
        if first[i] != second[i]:
            return False

    return True


def check_feed_shape_type(var, feed, num_places=1):
    """
    Returns True if the variable doesn't require feed check or it is compatible
    with the shape and have same dtype as the fed value.

    A dimension is compatible with the other if:
    1. The length of the dimensions are same.
    2. Each non-negative number of the two dimensions are same.
    3. For negative number or 'None' in a dimension, it means unknown so it
       is compatible with any number.
    
    Args:
        var (Variable): the Variable object
        feed (LoDTensor): the fed value, which must be a LoDTensor
        num_places: an integer value indicating the number of places.
            ParallelExecutor will divide data into devices (CPU/GPU) evenly.
    Returns:
        True if the shape and dtype of variable is compatible with the feed value
    Raises:
        ValueError: if the shape or dtype of the variable is not compatible with
            the feed value
    """
    if var.desc.need_check_feed():
        diff_shape = core.diff_tensor_shape(feed, var.desc, num_places)
        if diff_shape is not None:
            raise ValueError(
                'The fed Variable %r should have dimensions = %d, shape = '
                '%r, but received fed shape %r on each device' %
                (var.name, len(var.shape), var.shape, diff_shape))
        if not dtype_is_compatible_with(feed._dtype(), var.dtype):
            var_dtype_format = convert_dtype(var.dtype) if isinstance(
                var.dtype, core.VarDesc.VarType) else var.dtype
            feed_dtype_format = convert_dtype(feed._dtype()) if isinstance(
                feed._dtype(), core.VarDesc.VarType) else feed._dtype()
            raise ValueError(
                'The data type of fed Variable %r must be %r, but received %r' %
                (var.name, var_dtype_format, feed_dtype_format))
    return True


def has_feed_operators(block, feed_targets, feed_holder_name):
    """ Check whether the block already has feed operators.

    Return false if the block does not have any feed operators.
    If some feed operators have been prepended to the block, check that
    the info contained in these feed operators matches the feed_targets
    and feed_holder_name. Raise exception when any mismatch is found.
    Return true when the block has feed operators with matching info.

    Args:
        block: a block instance (typically global block of a program)
        feed_targets: a dictionary of {feed_target_name: feed_target_data}
        feed_holder_name: the name of the variable that holds the data of
            all feed targets. The type of this feed_holder variable is
            FEED_MINIBATCH, which is essentially vector<LoDTensor>.

    Returns:
        A boolean value that indicates whether a block has feed operators
        that match the info contained in feed_targets and feed_holder_name.
    """

    feed_count = 0
    for op in block.ops:
        if op.desc.type() == 'feed':
            feed_count += 1
            assert op.desc.input('X')[0] == feed_holder_name
            feed_target_name = op.desc.output('Out')[0]
            if feed_target_name not in feed_targets:
                raise Exception(
                    "'feed_targets' does not have {} variable".format(
                        feed_target_name))
        else:
            break
    if feed_count > 0 and feed_count != len(feed_targets):
        raise Exception(
            "Feed operators in program desc do not match 'feed_targets'")
    return feed_count > 0


def has_fetch_operators(block,
                        fetch_targets,
                        fetch_holder_name,
                        fetch_op='fetch'):
    """ Check whether the block already has fetch operators.

    Return false if the block does not have any fetch operators.
    If some fetch operators have been appended to the block, check that
    the info contained in these fetch operators matches the fetch_targets
    and fetch_holder_name. Raise exception when any mismatch is found.
    Return true when the block has fetch operators with matching info.

    Args:
        block: a block instance (typically global block of a program)
        fetch_targets: a dictionary of {fetch_target_name: fetch_target_data}
        fetch_holder_name: the name of the variable that holds the data of
            all fetch targets. The type of this fetch_holder variable is
            FETCH_LIST, which is essentially vector<LoDTensor>.
        fetch_op: the operator name of fetch

    Return:
        A boolean value that indicates whether a block has fetch operators
        that match the info contained in fetch_targets and fetch_holder_name.
    """

    fetch_count = 0
    for op in block.ops:
        if op.desc.type() == fetch_op:
            fetch_count += 1
            assert op.desc.output('Out')[0] == fetch_holder_name
            fetch_target_name = op.desc.input('X')[0]
            if fetch_target_name not in [
                    var.desc.name() for var in fetch_targets
            ]:
                raise Exception(
                    "'fetch_targets' does not have {} variable".format(
                        fetch_target_name))
            idx = op.desc.attr('col')
            assert fetch_target_name == fetch_targets[idx].desc.name()
    if fetch_count > 0 and fetch_count != len(fetch_targets):
        raise Exception(
            "Fetch operators in program desc do not match 'fetch_targets'")
    return fetch_count > 0


def _add_feed_fetch_ops(program,
                        feed,
                        fetch_list,
                        feed_var_name,
                        fetch_var_name,
                        use_fetch_v2=False):
    tmp_program = program.clone()

    global_block = tmp_program.global_block()

    if feed_var_name in global_block.vars:
        feed_var = global_block.var(feed_var_name)
    else:
        feed_var = global_block.create_var(
            name=feed_var_name,
            type=core.VarDesc.VarType.FEED_MINIBATCH,
            persistable=True)

    if fetch_var_name in global_block.vars:
        fetch_var = global_block.var(fetch_var_name)
    else:
        fetch_var = global_block.create_var(
            name=fetch_var_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True)

    # prepend feed operators
    if not has_feed_operators(global_block, feed, feed_var_name):
        for i, name in enumerate(feed):
            if global_block.has_var(name):
                out = global_block.var(name)
                global_block._prepend_op(type='feed',
                                         inputs={'X': [feed_var]},
                                         outputs={'Out': [out]},
                                         attrs={'col': i})
            else:
                warnings.warn(
                    "The variable %s is not found in program. It is not declared or is pruned."
                    % name)

    if use_fetch_v2:
        fetch_op = 'fetch_v2'
    else:
        fetch_op = 'fetch'

    # append fetch_operators
    if not has_fetch_operators(global_block, fetch_list, fetch_var_name,
                               fetch_op):
        for i, var in enumerate(fetch_list):
            assert isinstance(var, Variable) or isinstance(
                var, six.string_types), ("Wrong type for fetch_list[%s]: %s" %
                                         (i, type(var)))
            global_block.append_op(type=fetch_op,
                                   inputs={'X': [var]},
                                   outputs={'Out': [fetch_var]},
                                   attrs={'col': i})

    return tmp_program


def _apply_inplace_addto_pass(program, enable_inplace, enable_addto,
                              skip_var_names):
    use_cuda = True if core.is_compiled_with_cuda() else False

    attrs = {"use_cuda": use_cuda, "mem_opt_skip_vars": skip_var_names}
    attr_types = {"use_cuda": "bool", "mem_opt_skip_vars": "list[str]"}

    empty_startup_program = Program()
    if enable_inplace:
        pass_name = "buffer_shared_inplace_pass"
        _apply_pass(program, empty_startup_program, pass_name, attrs,
                    attr_types)
    if enable_addto and use_cuda:
        pass_name = "inplace_addto_op_pass"
        _apply_pass(program, empty_startup_program, pass_name, attrs,
                    attr_types)


def _fetch_var(name, scope=None, return_numpy=True):
    """
    Fetch the value of the variable with the given name from the
    given scope.

    Args:
        name(str): name of the variable. Typically, only persistable variables
            can be found in the scope used for running the program.
        scope(core.Scope|None): scope object. It should be the scope where
            you pass to Executor.run() when running your program.
            If None, global_scope() will be used. Default None.
        return_numpy(bool): whether convert the tensor to numpy.ndarray.
            Default True.

    Returns:
       LodTensor|numpy.ndarray
    """
    assert isinstance(name, six.string_types)
    if scope is None:
        scope = global_scope()
    assert isinstance(scope, core._Scope)

    var = scope.find_var(_to_name_str(name))
    assert var is not None, (
        "Cannot find " + name + " in scope. Perhaps you need to make the"
        " variable persistable by using var.persistable = True in your"
        " program.")
    tensor = var.get_tensor()
    if return_numpy:
        tensor = as_numpy(tensor, copy=True)
    return tensor


def _to_name_str(var):

    def _to_str(var):
        if isinstance(var, Variable):
            return var.desc.name()
        elif isinstance(var, str):
            return var
        elif isinstance(var, six.string_types):
            return str(var)
        elif isinstance(var, Operator):
            return str(id(var))
        else:
            raise TypeError(str(var) + " should be Variable, Operator or str")

    # NOTEz(zhiqiu): The item in fetch_list may be tuple returned by Optimizer.minimize(),
    # see comments in _split_optimize_ops_in_fetch_list for more details.
    if isinstance(var, tuple):
        var = var[0]
    if isinstance(var, list):
        s = [_to_str(item) for item in var]
        return ','.join(s)
    else:
        return _to_str(var)


def _is_enable_standalone_executor():
    return framework._enable_standalone_executor_ is None or framework._enable_standalone_executor_ in [
        1, '1', True, 'True', 'true'
    ]


def _is_dy2st_enable_standalone_executor():
    return framework._dy2st_enable_standalone_executor_ in [
        1, '1', True, 'True', 'true'
    ]


def _prepare_fleet_executor():
    from ..distributed.fleet.proto import fleet_executor_desc_pb2
    trainer_endpoints_str = os.getenv("PADDLE_TRAINER_ENDPOINTS", "")
    trainer_endpoints = trainer_endpoints_str.split(',')
    fleet_exe_desc = fleet_executor_desc_pb2.FleetExecutorDesc()
    cur_rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
    fleet_exe_desc.cur_rank = cur_rank
    nrank = len(trainer_endpoints)
    for rank, endpoint in enumerate(trainer_endpoints):
        rank_info = fleet_executor_desc_pb2.RankInfo()
        rank_info.rank = rank
        rank_info.ip_port = endpoint
        fleet_exe_desc.cluster_info.append(rank_info)
    fleet_exe = core.FleetExecutor(fleet_exe_desc.SerializeToString())
    return fleet_exe


def _get_strong_program_cache_key_for_new_exe(program, feed, fetch_list):
    return program.desc.cached_hash_str() + _get_program_cache_key(
        feed, fetch_list)


def _get_strong_program_cache_key(program, feed, fetch_list):
    # TODO(zhiqiu): use hash_str to generate cache key as above
    def _get_varname_from_block(block):
        block_str = []
        for var_name in list(block.vars.keys()):
            block_str.append(var_name)
        return "\n".join(block_str)

    inner_program = program._program if isinstance(
        program, compiler.CompiledProgram) else program
    return _get_varname_from_block(inner_program.blocks[0]) + str(
        id(program)) + _get_program_cache_key(feed, fetch_list)


def _get_program_cache_key(feed, fetch_list):
    feed_var_names = []
    if isinstance(feed, dict):
        feed_var_names = list(feed.keys())
    elif isinstance(feed, list) or isinstance(feed, tuple):
        for i, each in enumerate(feed):
            feed_var_names += list(each.keys())
    fetch_var_names = list(map(_to_name_str, fetch_list))
    return str(feed_var_names + fetch_var_names)


def _as_lodtensor(data, place, dtype=None):
    """
        Convert numpy.ndarray to Tensor, its only support Tensor without LoD information.
        For higher dimensional sequence data, please use LoDTensor directly.

        Examples:
            >>> import paddle.fluid as fluid
            >>> place = fluid.CPUPlace()
            >>> exe = fluid.executor(place)
            >>> data = np.array(size=(100, 200, 300))
            >>> np_outs = map(lambda x: fluid.executor._as_lodtensor(x, place), data)
            >>>     ...

        Args:
            data(numpy.ndarray|list|tuple|scalar): a instance of array, scalar, list or tuple
            data(core.Place): the place of created tensor
            dtype(core.VarDesc.VarType|str): the expected data type of created tensor

        Returns:
            LoDTensor
        """
    #NOTE(zhiqiu): convert python builtin, like float, int, and list, to numpy ndarray
    if not isinstance(data, np.ndarray):
        assert dtype is not None, 'The dtype should be given when feed data is not np.ndarray'
        dtype = convert_dtype(dtype) if isinstance(
            dtype, core.VarDesc.VarType) else dtype
        if np.isscalar(data):
            data = np.array([data]).astype(dtype)
        elif isinstance(data, (list, tuple)):
            data = np.array(data)
            if data.dtype == np.object_:
                raise TypeError(
                    "\n\tFaild to convert input data to a regular ndarray :\n\t* Usually "
                    "this means the input data contains nested lists with different lengths. "
                    "Please consider using 'fluid.create_lod_tensor' to convert it to a LoD-Tensor."
                )
            data = data.astype(dtype)
        else:
            raise TypeError(
                "Convert data of type {} to Tensor is not supported".format(
                    type(data)))

    # convert numpy.ndarray to tensor
    tensor = core.LoDTensor()
    tensor.set(data, place)
    return tensor


class FetchHandler(object):

    def __init__(self, var_dict=None, period_secs=60):
        assert var_dict != None
        self.var_dict = var_dict
        self.period_secs = period_secs

    def handler(self, res_dict):
        for key in res_dict:
            if type(res_dict[key]) is np.ndarray:
                sys.stdout.write("{}[0]: {} ".format(key, res_dict[key][0]))
        sys.stdout.write("\n")

    @staticmethod
    def help():
        print("""
class FetchHandlerExample(FetchHandler):
    def handler(self, res_dict):
        print(res_dict["auc"])
        print("auc: {}, {}".format(res_dict["auc"], time.ctime()))

auc = Variable()
var_dict = {"auc": auc}
handler = FetchHandlerExample(var_dict=var_dict)
""")


class _StandaloneExecutor(object):

    def __init__(self, place, main_program, scope):
        self._place = core.Place()
        self._place.set_place(place)
        self._main_program = main_program
        self._scope = scope
        self._new_exe = self._create_new_executor()

    def run(self, scope, feed_names, fetch_list, return_numpy=True):
        """
        Args:
            feed_names(list): This parameter represents the input names of the model.
            fetch_list(list): This parameter represents the Tensors that need to be returned
                after the model runs. The default is None. 
            return_numpy(bool): This parameter indicates whether convert the fetched Tensors
                (the Tensor specified in the fetch list) to numpy.ndarray. if it is False,
                the type of the return value is a list of :code:`LoDTensor`. The default is True.
        """
        fetch_list = self._check_fetch(fetch_list)

        tensors = self._new_exe.run(scope, feed_names,
                                    fetch_list)._move_to_list()
        if return_numpy:
            return as_numpy(tensors, copy=True)
        else:
            return tensors

    def _create_new_executor(self):
        new_exe = core.StandaloneExecutor(self._place, self._main_program.desc)

        return new_exe

    def _update_feed(self, feed):
        """
        Update the feed dict, remove the feed item which is pruned in program.  

        Notes: This is a very low level API. Users should not use this API
        directly. 

        Args:
            feed(list|dict): feed dict or list.

        Returns:
            feed:(list|dict)  updated feed.
        """
        if feed is None:
            feed = {}
        elif isinstance(feed, (list, tuple)):
            assert len(feed) == 1, "Not compiled with data parallel"
            feed = feed[0]

        if not isinstance(feed, dict):
            raise TypeError(
                "feed requires dict as its Parameter. But you passed in %s" %
                (type(feed)))

        global_block = self._main_program.global_block()
        for feed_name in list(feed.keys()):
            if not global_block.has_var(feed_name):
                feed.pop(feed_name)
                warnings.warn(
                    "The variable %s is not found in program. It is not declared or is pruned."
                    % feed_name)

        return feed

    def _check_fetch(self, fetch_list):
        if fetch_list is None:
            fetch_list = []

        res = []
        for fetch_var in fetch_list:
            if isinstance(fetch_var, Variable):
                fetch_var = fetch_var.name
            elif not isinstance(fetch_var, str):
                raise TypeError(
                    "Required fetch_var shall be str|Variable, but received {}".
                    format(type(fetch_var).__name__))

            res.append(fetch_var)
        return res


class _ExecutorCache(object):

    class _CachedData(object):

        def __init__(self, program, feed, fetch_list, feed_var_name,
                     fetch_var_name, place, scope):
            self.program = program
            self.feed = feed
            self.fetch_list = fetch_list
            self.feed_var_name = feed_var_name
            self.fetch_var_name = fetch_var_name
            self.place = place
            self.scope = scope

            # NOTE(Ruibiao): Not all changeable item is considered for key at present,
            # ONLY: program, feed, and fetch_list
            if isinstance(self.program, compiler.CompiledProgram):
                if not self.program._program:
                    # The program holds no _program, maybe it is constructed by graph.
                    # Convert graph to program in order to generate key.
                    self.program._program = framework.IrGraph(
                        self.program._graph).to_program()
                self.key = hash(
                    _get_strong_program_cache_key_for_new_exe(
                        self.program._program, feed, fetch_list))
            else:
                self.key = hash(
                    _get_strong_program_cache_key_for_new_exe(
                        self.program, feed, fetch_list))

        def __eq__(self, other):
            return isinstance(
                other, _ExecutorCache._CachedData) and self.key == other.key

        def __hash__(self):
            return self.key

    def __init__(self):
        # NOTE(Ruibiao): Wrap the lru_cache in constructor so that the cache is local to
        # the _ExecutorCache instance, otherwise a global cache may not be released after
        # the Executor instance deleted
        self._get_cached_program_and_executor = lru_cache(maxsize=8)(
            self._get_program_and_executor)

    def clear(self):
        self._get_cached_program_and_executor.cache_clear()

    def get_program_and_executor(self, program, feed, fetch_list, feed_var_name,
                                 fetch_var_name, place, scope):
        return self._get_cached_program_and_executor(
            self._CachedData(program, feed, fetch_list, feed_var_name,
                             fetch_var_name, place, scope))

    def _get_program_and_executor(self, cached_data):
        program = cached_data.program
        inner_program = program._program if isinstance(
            program, compiler.CompiledProgram) else program
        feed = cached_data.feed
        fetch_list = cached_data.fetch_list
        feed_var_name = cached_data.feed_var_name
        fetch_var_name = cached_data.fetch_var_name
        place = cached_data.place
        scope = cached_data.scope

        # To apply IR pass, compile the Program to IrGraph and convert it back to Program
        if isinstance(program, compiler.CompiledProgram) or isinstance(
                program._graph, compiler.CompiledProgram):
            compiled_program = program if isinstance(
                program, compiler.CompiledProgram) else program._graph
            build_strategy = compiled_program._build_strategy
            # print(f"Program before convert:\n {inner_program}", flush=True)
            compiled_program._compile(scope, place)
            ir_graph = framework.IrGraph(compiled_program._graph)
            converted_program = ir_graph.to_program()

            if hasattr(inner_program, 'lr_sheduler'):
                converted_program.lr_sheduler = inner_program.lr_sheduler

            inner_program = converted_program
            # print(f"Program after convert:\n {inner_program}", flush=True)
        else:
            build_strategy = None
            from paddle.incubate.autograd import prim_enabled, prim2orig
            if prim_enabled() and program == default_main_program():
                prim2orig()

            inner_program = program

        program = _add_feed_fetch_ops(program=inner_program,
                                      feed=feed,
                                      fetch_list=fetch_list,
                                      feed_var_name=feed_var_name,
                                      fetch_var_name=fetch_var_name,
                                      use_fetch_v2=True)

        if os.environ.get('FLAGS_CONVERT_GRAPH_TO_PROGRAM', None) in [
                1, '1', True, 'True', 'true'
        ] and not program._is_start_up_program_:
            if program.num_blocks > 1:
                # If there are multiple blocks in the program, subblock will not be executed with the new executor in temporary
                logging.warning("There are more than 1 block in program.")
            elif program.num_blocks == 1:
                logging.warning("There are 1 block in program.")
            else:
                logging.warning("There are no block in program.")

        # standalone executor will apply buffer_shared_inplace_pass and
        # inplace_addto_op_pass to program according to build_strategy
        enable_inplace = True if build_strategy is None or build_strategy.enable_inplace else False
        enable_addto = True if build_strategy is not None and build_strategy.enable_addto else False
        if enable_inplace or enable_addto:
            # inplace should skip feed and fetch var
            skip_var_names = eval(_get_program_cache_key(feed, fetch_list))
            _apply_inplace_addto_pass(program, enable_inplace, enable_addto,
                                      skip_var_names)

        new_program = program.clone()
        new_exe = _StandaloneExecutor(place, new_program, scope)
        return new_program, new_exe


class Executor(object):
    """
    :api_attr: Static Graph

    An Executor in Python, supports single/multiple-GPU running,
    and single/multiple-CPU running.

    Args:
        place(paddle.CPUPlace()|paddle.CUDAPlace(n)|str|None): This parameter represents
            which device the executor runs on. When this parameter is None, PaddlePaddle
            will set the default device according to its installation version. If Paddle
            is CPU version, the default device would be set to `CPUPlace()` . If Paddle is
            GPU version, the default device would be set to `CUDAPlace(0)` . Default is None.
            If ``place`` is string, it can be ``cpu``, and ``gpu:x``, where ``x`` 
            is the index of the GPUs. Note: users only pass one Place or None to initialize
            Executor when using multiple-cards. Other APIs will override the cards. See
            `document for multiple-cards <https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/01_paddle2.0_introduction/update_en.html#stand-alone-multi-card-launch>`_ 

    Returns:
        Executor

    Examples:
        .. code-block:: python

            import paddle
            import numpy
            import os

            # Executor is only used in static graph mode
            paddle.enable_static()

            # Set place explicitly.
            # use_cuda = True
            # place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            # exe = paddle.static.Executor(place)

            # If you don't set place, PaddlePaddle sets the default device.
            exe = paddle.static.Executor()

            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(train_program, startup_program):
                data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
                hidden = paddle.static.nn.fc(data, 10)
                loss = paddle.mean(hidden)
                paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

            # Run the startup program once and only once.
            # Not need to optimize/compile the startup program.
            exe.run(startup_program)

            # Run the main program directly without compile.
            x = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data, = exe.run(train_program, feed={"X": x}, fetch_list=[loss.name])

            # Or, compiled the program and run. See `CompiledProgram`
            # for more details.
            # NOTE: If you use CPU to run the program or Paddle is
            # CPU version, you need to specify the CPU_NUM, otherwise,
            # PaddlePaddle will use all the number of the logic core as
            # the CPU_NUM, in that case, the batch size of the input
            # should be greater than CPU_NUM, if not, the process will be
            # failed by an exception.

            # Set place explicitly.
            # if not use_cuda:
            #     os.environ['CPU_NUM'] = str(2)

            # If you don't set place and PaddlePaddle is CPU version
            os.environ['CPU_NUM'] = str(2)

            compiled_prog = paddle.static.CompiledProgram(
                train_program).with_data_parallel(loss_name=loss.name)
            loss_data, = exe.run(compiled_prog, feed={"X": x}, fetch_list=[loss.name])

    """

    def __init__(self, place=None):
        if place is None:
            expected_place = framework._current_expected_place()
            self.place = expected_place
        else:
            self.place = framework._get_paddle_place(place)
        self.program_caches = dict()
        self.ctx_caches = dict()
        self.trainer_caches = dict()
        self.scope_caches = dict()
        self.var_caches = dict()
        self.pruned_program_caches = dict()
        p = core.Place()
        p.set_place(self.place)
        self._default_executor = core.Executor(p)
        self._closed = False
        self.pruned_program_scope_caches = dict()
        self._prepare_to_run_called = False

        self._auto_checkpoint_name = unique_name.generate(
            "__auto_checkpoint_executor__")

        # NOTE: Whether to use experimental executor `StandaloneExecutor`.
        self._enable_interpreter_core = _is_enable_standalone_executor()
        self._executor_cache = _ExecutorCache()

        self._fleet_executor = None
        # TODO(liyurui): This option will be removed and always true when the functionality
        # of fleet executor with standalone executor is ready.
        self._fleet_executor_with_standalone = False

    def __del__(self):
        # NOTE(Ruibiao): The manually call of clear is required. Because in Python, executor_cache
        # may not immediately destructed after Executor instance deleted (so does not the _StandaloneExecutor),
        # that brings errors to mkl-dnn unit tests (see ClearMKLDNNCache in interpretercore.cc for why).
        self._executor_cache.clear()

    def _get_scope_cache(self, program_cache_key):
        return self.scope_caches.get(program_cache_key, None)

    def _get_ctx_cache(self, program_cache_key):
        return self.ctx_caches.get(program_cache_key, None)

    def _get_trainer_cache(self, program_cache_key):
        return self.trainer_caches.get(program_cache_key, None)

    def _get_program_cache(self, program_cache_key):
        return self.program_caches.get(program_cache_key, None)

    def _add_program_cache(self, program_cache_key, program):
        self.program_caches[program_cache_key] = program

    def _get_pruned_program_cache(self, program_cache_key):
        return self.pruned_program_caches.get(program_cache_key, None)

    def _add_pruned_program_cache(self, program_cache_key, program):
        self.pruned_program_caches[program_cache_key] = program

    def _get_pruned_program_scope_cache(self, program_cache_key):
        return self.pruned_program_scope_caches.get(program_cache_key, None)

    def _add_pruned_program_scope_cache(self, program_cache_key, program):
        self.pruned_program_scope_caches[program_cache_key] = program

    def _add_ctx_cache(self, ctx_cache_key, ctx):
        self.ctx_caches[ctx_cache_key] = ctx

    def _add_trainer_cache(self, trainer_cache_key, ctx):
        self.trainer_caches[trainer_cache_key] = ctx

    def _add_scope_cache(self, scope_cache_key, scope):
        self.scope_caches[scope_cache_key] = scope

    # just for testing, will be removed later
    @lru_cache()
    def _log_force_set_program_cache(self, use_program_cache):
        logging.warning(
            f"use_program_cache is force set to {use_program_cache} by FLAGS_FORCE_USE_PROGRAM_CACHE"
        )

    def _feed_data(self, program, feed, feed_var_name, scope):
        # feed var to framework
        global_block = program.global_block()
        for op in global_block.ops:
            if op.desc.type() == 'feed':
                feed_target_name = op.desc.output('Out')[0]
                cur_feed = feed[feed_target_name]
                var = global_block.var(feed_target_name)
                if var.dtype != core.VarDesc.VarType.STRINGS:
                    if not isinstance(cur_feed, core.LoDTensor):
                        cur_feed = _as_lodtensor(cur_feed, self.place,
                                                 var.dtype)
                    check_feed_shape_type(var, cur_feed)
                idx = op.desc.attr('col')
                core.set_feed_variable(scope, cur_feed, feed_var_name, idx)
            else:
                break

    def _fetch_data(self, fetch_list, fetch_var_name, scope):
        outs = [
            core.get_fetch_variable(scope, fetch_var_name, i)
            for i in six.moves.range(len(fetch_list))
        ]
        return outs

    @classmethod
    def _split_optimize_ops_in_fetch_list(cls, fetch_list):
        """
        Split optimize_ops from fetch_list, which provided to specify program prunning.
        Args:
            fetch_list(list): The original fetch_list.
            Possible types of fetch_list are:
                fetch_list = ['loss']
                fetch_list = [[sgd, sgd], 'loss']
                fetch_list = [([sgd, sgd], [(param, grad)]), 'loss']

        Returns:
            optimize_ops(list): The optimize operators splited from fetch_list.
            fetch_list(list):  The updated fetch_list which does not contain optimize operators.  
        """
        _optimize_ops = []
        _fetch_list = []

        def _get_targets(_optimize_ops, _fetch_list, item):
            if isinstance(item, Operator):
                if item._is_optimize_op():
                    _optimize_ops.append(item)
                else:
                    raise TypeError(
                        "The operator in fetch_list is not an optimize_op")
            elif isinstance(item, Variable) or isinstance(
                    item, str) or isinstance(item, six.string_types):
                _fetch_list.append(item)
            else:
                raise TypeError(
                    "The item in fetch_list should be str, variable or optimize_op, but received %s.",
                    type(item))

        for index, item in enumerate(fetch_list):
            # NOTE(zhiqiu): to support (optimizer_ops, param_and_grads) and optimizer_ops in fetch_list
            # we should handle tuple and list in fetch_list.
            # TODO(zhiqiu): find a better way to handle that.
            if isinstance(item, list):
                for i in item:
                    _get_targets(_optimize_ops, _fetch_list, i)
            elif isinstance(item, tuple):
                if not isinstance(item[0], (list, tuple)):
                    raise TypeError(
                        "Requires fetch_list[{}][0] shall be one of (list, tuple) when type(fetch_list[{}]) is `tuple`, but received fetch_list[{}][0]'s type is `{}`."
                        .format(index, index, index,
                                type(item[0]).__name__))
                for i in item[0]:
                    _get_targets(_optimize_ops, _fetch_list, i)
            else:
                _get_targets(_optimize_ops, _fetch_list, item)

        return _fetch_list, _optimize_ops

    @classmethod
    def _prune_program(cls,
                       program,
                       feed=None,
                       fetch_list=None,
                       optimize_ops=None):
        """
        Prune operators and variables which are not needed to generate
        :code:`fetch_list` and optimize operators. 
        Prune operators and variables which are needed 
        to generate variables to be feeded.  

        Notes: This is a very low level API. Users should not use this API
        directly. 

        Args:
            program(Program): the origin program
            feed(list|dict): feed dict or list.
            fetch_list(list|Variable): A list of variables need to be fetched
            optimize_ops(list[Operator]): A list of optimizer operators

        Returns:
            Program:  A new, pruned program.
        """
        compiled = isinstance(program, compiler.CompiledProgram)
        if compiled:
            if program._program:
                origin_program = program._program
            else:
                warnings.warn(
                    "The program holds no _program, maybe it is constructed by graph, which can't be pruned yet."
                )
                return
        else:
            origin_program = program

        feed_names = []
        if isinstance(feed, dict):
            feed_names = list(feed.keys())
        elif isinstance(feed, list) or isinstance(feed, tuple):
            for i, each in enumerate(feed):
                feed_names += list(each.keys())

        # if optimize_ops is [], all optimize ops in the program is used.
        if not optimize_ops:
            for block in origin_program.blocks:
                for op in block.ops:
                    if op._is_optimize_op():
                        optimize_ops.append(op)

        targets = fetch_list + optimize_ops
        pruned_program = origin_program._prune_with_input(feed_names, targets)

        if compiled:
            # for compiled program, update the underlying program, re-generate graph,
            # and reset the flag so it can be compiled again.
            program._program = pruned_program
            program._graph = core.Graph(pruned_program.desc)
            program._compiled = False
        else:
            program = pruned_program

        return program

    @classmethod
    def _update_feed(cls, program, feed):
        """
        Update the feed dict, remove the feed item which is pruned in program.  

        Notes: This is a very low level API. Users should not use this API
        directly. 

        Args:
            program(Program): the pruned program.
            feed(list|dict): feed dict or list.

        Returns:
            feed:(list|dict)  updated feed.
        """
        compiled = isinstance(program, compiler.CompiledProgram)
        if compiled:
            if program._program:
                global_block = program._program.global_block()
            else:
                warnings.warn(
                    "The program holds no _program, maybe it is constructed by graph."
                )
                return feed
        else:
            global_block = program.global_block()

        if isinstance(feed, dict):
            for feed_name in list(feed.keys()):
                if not global_block.has_var(feed_name):
                    feed.pop(feed_name)
                    warnings.warn(
                        "The variable %s is not found in program. It is not declared or is pruned."
                        % feed_name)

        elif isinstance(feed, list) or isinstance(feed, tuple):
            for i, each in enumerate(feed):
                for feed_name in list(each.keys()):
                    if not global_block.has_var(feed_name):
                        each.pop(feed_name)
                        warnings.warn(
                            "The variable %s is not found in program. It is not declared or is pruned."
                            % feed_name)
        return feed

    '''
    TODO(typhoonzero): Define "no longer use" meaning? Can user create
    a new Executor for the same program and run?
    TODO(panyx0718): Why ParallelExecutor doesn't have close?
    '''

    def close(self):
        """
        Close the executor. This interface is used for distributed training (PServers mode).
        This executor can not be used after calling the interface, because
        this interface releases resources associated with the current Trainer.

        Returns:
            None

        Examples:
            .. code-block:: python

              import paddle

              cpu = paddle.CPUPlace()
              exe = paddle.static.Executor(cpu)
              # execute training or testing
              exe.close()
        """
        if not self._closed:
            self._closed = True
            for k, trainer_instance in self.trainer_caches.items():
                self._default_executor.release_trainer(trainer_instance)
                del trainer_instance
            self._default_executor.close()

    def _run_parallel(self, program, scope, feed, fetch_list, fetch_var_name,
                      return_numpy, return_merged):
        from paddle.optimizer.lr import LRScheduler
        exe = program._executor
        # TODO(zhenghuihuang): quantization uses Graph in CompiledProgram
        # instead of program. We will add support for checking Vars in Graph
        need_check_feed = program._program is not None
        if need_check_feed:
            global_block = program._program.global_block()
        if isinstance(feed, dict):
            feed_tensor_dict = dict()
            for feed_name in feed:
                feed_tensor = feed[feed_name]
                var = global_block.var(feed_name) if need_check_feed else None
                if not isinstance(feed_tensor, core.LoDTensor):
                    # always set to CPU place, since the tensor need to be split
                    # it is fast in CPU
                    feed_tensor = _as_lodtensor(feed[feed_name],
                                                core.CPUPlace(),
                                                var.dtype if var else None)
                if need_check_feed:
                    check_feed_shape_type(var, feed_tensor, exe.device_count())
                feed_tensor_dict[feed_name] = feed_tensor
            exe.feed_and_split_tensor_into_local_scopes(feed_tensor_dict)

        elif isinstance(feed, list) or isinstance(feed, tuple):
            res = list()
            for i, each in enumerate(feed):
                if not isinstance(each, dict):
                    raise TypeError(
                        "Each element of feed list should be a dict")
                res_dict = dict()
                for feed_name in each:
                    tensor = each[feed_name]
                    var = global_block.var(
                        feed_name) if need_check_feed else None
                    if not isinstance(tensor, core.LoDTensor):
                        tensor = _as_lodtensor(each[feed_name],
                                               program._places[i],
                                               var.dtype if var else None)
                    if need_check_feed:
                        check_feed_shape_type(var, tensor)
                    res_dict[feed_name] = tensor
                res.append(res_dict)

            exe.feed_tensors_into_local_scopes(res)

        if hasattr(program._program, 'lr_sheduler'):
            lr_sheduler = program._program.lr_sheduler
            assert isinstance(lr_sheduler, LRScheduler), "must be LRScheduler"
            lr_value = lr_sheduler()
            lr_var = program._program.global_block().vars[lr_sheduler._var_name]
            lr_tensor = _as_lodtensor(lr_value, core.CPUPlace(), lr_var.dtype)
            if core.is_cuda_graph_capturing():
                warnings.warn(
                    "Caution!!! When capturing CUDA Graph, the learning rate scheduler would not "
                    "take any effect! Please set the learning rate manually before each batch!"
                )
            else:
                exe.feed_and_split_tensor_into_local_scopes(
                    {lr_sheduler._var_name: lr_tensor})

        fetch_var_names = list(map(_to_name_str, fetch_list))
        tensors = exe.run(fetch_var_names, return_merged)._move_to_list()
        return as_numpy(tensors) if return_numpy else tensors

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True,
            use_program_cache=False,
            return_merged=True,
            use_prune=False):
        """
        Run the specified :code:`Program` or :code:`CompiledProgram`. It should be noted that the executor
        will execute all the operators in :code:`Program` or :code:`CompiledProgram` without pruning some
        operators of the :code:`Program` or :code:`CompiledProgram` according to fetch_list. And you could
        specify the scope to store the :code:`Tensor` during the executor running if the scope
        is not set, the executor will use the global scope, i.e. :code:`paddle.static.global_scope()`.

        Args:
            program(Program|CompiledProgram): This parameter represents the :code:`Program` or
                :code:`CompiledProgram` to be executed. If this parameter is not provided, that
                parameter is None, the program will be set to :code:`paddle.static.default_main_program()`.
                The default is None.
            feed(list|dict): This parameter represents the input Tensors of the model.
                If it is single card training, the feed is dict type, and if it is multi-card
                training, the parameter feed can be dict or list of Tensors. If the
                parameter type is dict, the data in the feed will be split and sent to
                multiple devices (CPU/GPU), that is to say, the input data will be evenly
                sent to different devices, so you should make sure the number of samples of
                the current mini-batch must be greater than the number of places;
                if the parameter type is list, those data are copied directly to each device,
                so the length of this list should be equal to the number of places.
                The default is None.
            fetch_list(list): This parameter represents the Tensors that need to be returned
                after the model runs. The default is None. 
            feed_var_name(str): This parameter represents the name of the input Tensor of
                the feed operator. The default is "feed".
            fetch_var_name(str): This parameter represents the name of the output Tensor of
                the fetch operator. The default is "fetch".
            scope(Scope): the scope used to run this program, you can switch 
                it to different scope. default is :code:`paddle.static.global_scope()`
            return_numpy(bool): This parameter indicates whether convert the fetched Tensors
                (the Tensor specified in the fetch list) to numpy.ndarray. if it is False,
                the type of the return value is a list of :code:`LoDTensor`. The default is True.
            use_program_cache(bool): This parameter indicates whether the input :code:`Program` is cached.
                If the parameter is True, the model may run faster in the following cases:
                the input program is :code:`paddle.static.Program`, and the parameters(program, feed Tensor name
                and fetch_list Tensor) of this interface remains unchanged during running.
                The default is False.
            return_merged(bool): This parameter indicates whether fetched Tensors (the Tensors
                specified in the fetch list) should be merged according to the execution device dimension.
                If :code:`return_merged` is False, the type of the return value is a two-dimensional list
                of :code:`Tensor` / :code:`LoDTensorArray` ( :code:`return_numpy` is False) or a two-dimensional
                list of :code:`numpy.ndarray` ( :code:`return_numpy` is True). If :code:`return_merged` is True,
                the type of the return value is an one-dimensional list of :code:`Tensor` / :code:`LoDTensorArray`
                ( :code:`return_numpy` is False) or an one-dimensional list of :code:`numpy.ndarray`
                ( :code:`return_numpy` is True). Please see Examples 2 for more details. If the lengths of fetched
                results are variant, please set :code:`return_merged` as False, which denotes that the fetched
                results will not be merged. The default is True, but it is just for the compatibility, and may
                use False as default value in the future version.
            use_prune(bool): This parameter indicates whether the input :code:`Program` will be pruned. 
                If the parameter is True, the program will be pruned accroding to the given feed and fetch_list,
                which means the operators and variables in program that generate :code:`feed` and are not 
                needed to generate :code:`fetch_list` will be pruned. The default is False, which means the 
                program will not pruned and all the operators and variables will be executed during running.
                Note that if the tuple returned from :code:`Optimizer.minimize()` is passed to :code:`fetch_list`, 
                :code:`use_prune` will be overrided to True, and the program will be pruned.
                
        Returns:

            List: The fetched result list.

        NOTES:
            1. If it is multi-card running and the feed parameter is dict type, the input data
               will be evenly sent to different cards. For example, using two GPUs to run the model,
               the input sample number is 3, that is, [0, 1, 2], the sample number on GPU0 is 1,
               that is, [0], and the sample number on GPU1 is 2, that is, [1, 2].
               If the number of samples is less than the number of devices, the program will
               throw an exception, so when running the model, you should make sure that the
               number of samples of the last batch of the data set should be greater than the
               number of CPU cores or GPU cards, if it is less than, it is recommended that
               the batch be discarded.
            2. If the number of CPU cores or GPU cards available is greater than 1, the fetch
               results are spliced together in dimension 0 for the same Tensor values
               (Tensors in fetch_list) on different devices.

        Examples:
            .. code-block:: python
                :name: code-example-1

                import paddle
                import numpy

                # First create the Executor.
                paddle.enable_static()
                place = paddle.CPUPlace()  # paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)

                data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
                hidden = paddle.static.nn.fc(data, 10)
                loss = paddle.mean(hidden)
                adam = paddle.optimizer.Adam()
                adam.minimize(loss)
                i = paddle.zeros(shape=[1], dtype='int64')
                array = paddle.fluid.layers.array_write(x=loss, i=i)

                # Run the startup program once and only once.
                exe.run(paddle.static.default_startup_program())

                x = numpy.random.random(size=(10, 1)).astype('float32')
                loss_val, array_val = exe.run(feed={'X': x},
                                              fetch_list=[loss.name, array.name])
                print(array_val)
                # [array([0.02153828], dtype=float32)]

            .. code-block:: python
                :name: code-example-2

                # required: gpu
                import paddle
                import numpy as np

                # First create the Executor.
                paddle.enable_static()
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)

                data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
                class_dim = 2
                prediction = paddle.static.nn.fc(data, class_dim)
                loss = paddle.mean(prediction)
                adam = paddle.optimizer.Adam()
                adam.minimize(loss)

                # Run the startup program once and only once.
                exe.run(paddle.static.default_startup_program())
                build_strategy = paddle.static.BuildStrategy()
                binary = paddle.static.CompiledProgram(
                    paddle.static.default_main_program()).with_data_parallel(
                        loss_name=loss.name, build_strategy=build_strategy)
                batch_size = 6
                x = np.random.random(size=(batch_size, 1)).astype('float32')

                # Set return_merged as False to fetch unmerged results:
                unmerged_prediction, = exe.run(binary,
                                               feed={'X': x},
                                               fetch_list=[prediction.name],
                                               return_merged=False)
                # If the user uses two GPU cards to run this python code, the printed result will be
                # (2, 3, class_dim). The first dimension value of the printed result is the number of used
                # GPU cards, and the second dimension value is the quotient of batch_size and the
                # number of used GPU cards.
                print("The unmerged prediction shape: {}".format(
                    np.array(unmerged_prediction).shape))
                print(unmerged_prediction)

                # Set return_merged as True to fetch merged results:
                merged_prediction, = exe.run(binary,
                                             feed={'X': x},
                                             fetch_list=[prediction.name],
                                             return_merged=True)
                # If the user uses two GPU cards to run this python code, the printed result will be
                # (6, class_dim). The first dimension value of the printed result is the batch_size.
                print("The merged prediction shape: {}".format(
                    np.array(merged_prediction).shape))
                print(merged_prediction)

                # Out:
                # The unmerged prediction shape: (2, 3, 2)
                # [array([[-0.37620035, -0.19752218],
                #        [-0.3561043 , -0.18697084],
                #        [-0.24129935, -0.12669306]], dtype=float32), array([[-0.24489994, -0.12858354],
                #        [-0.49041364, -0.25748932],
                #        [-0.44331917, -0.23276259]], dtype=float32)]
                # The merged prediction shape: (6, 2)
                # [[-0.37789783 -0.19921964]
                #  [-0.3577645  -0.18863106]
                #  [-0.24274671 -0.12814042]
                #  [-0.24635398 -0.13003758]
                #  [-0.49232286 -0.25939852]
                #  [-0.44514108 -0.2345845 ]]

        """
        # Temporary FLAGS, just for testing the performance of program cache
        force_use_program_cache = os.environ.get(
            'FLAGS_FORCE_USE_PROGRAM_CACHE', None)
        if force_use_program_cache is not None:
            use_program_cache = force_use_program_cache in [
                1, '1', True, 'True', 'true'
            ]
            self._log_force_set_program_cache(use_program_cache)

        try:
            res = self._run_impl(program=program,
                                 feed=feed,
                                 fetch_list=fetch_list,
                                 feed_var_name=feed_var_name,
                                 fetch_var_name=fetch_var_name,
                                 scope=scope,
                                 return_numpy=return_numpy,
                                 use_program_cache=use_program_cache,
                                 use_prune=use_prune,
                                 return_merged=return_merged)
            core.update_autotune_status()
            return res
        except Exception as e:
            six.reraise(*sys.exc_info())

    def _run_impl(self, program, feed, fetch_list, feed_var_name,
                  fetch_var_name, scope, return_numpy, use_program_cache,
                  return_merged, use_prune):
        if self._closed:
            raise RuntimeError("Attempted to use a closed Executor")

        use_default_main_program = program is None
        if program is None:
            program = default_main_program()

        fetch_list = self._check_fetch_list(fetch_list)

        if isinstance(program, Program) and program._pipeline_opt:
            if "fleet_opt" in program._pipeline_opt:
                # Move prepare here for port conflict with nccl in startup program
                if self._fleet_executor is None:
                    self._fleet_executor = _prepare_fleet_executor()
                return self._run_using_fleet_executor(
                    program=program,
                    feed=feed,
                    fetch_list=fetch_list,
                    with_standalone_executor=self.
                    _fleet_executor_with_standalone)
            if "startup_program" in program._pipeline_opt:
                program = program._pipeline_opt["startup_program"]
            else:
                return self._run_pipeline(program,
                                          fetch_list=fetch_list,
                                          use_program_cache=use_program_cache)

        if isinstance(program, Program) and program._heter_pipeline_opt:
            #print("program._heter_pipeline_opt: {}".format(
            #    program._heter_pipeline_opt))
            ## change default executor
            heter_place = program._heter_pipeline_opt["heter_place"]
            heter_place = framework._get_paddle_place(heter_place)
            p = core.Place()
            p.set_place(heter_place)
            self._default_executor = core.Executor(p)
            # TODO(zhangminxu): support heterps pipeline training using exe.run
            if "startup_program" in program._heter_pipeline_opt:
                #print("get startup_program from _pipeline_opt")
                program = program._heter_pipeline_opt["startup_program"]

        if isinstance(program, Program) and \
                        len(program.global_block().ops) == 0:
            if use_default_main_program:
                error_info = "Now you are using default_main_program, "\
                    "but there are no operators in the program to be executed. "\
                    "Please ensure you create model correctly or you can pass "\
                    "the Program or the CompiledProgram manually."
            else:
                error_info = "There are no operators in the program to be executed. "\
                    "If you pass Program manually, please use fluid.program_guard "\
                    "to ensure the current Program is being used."
            warnings.warn(error_info)

        if scope is None:
            scope = global_scope()

        # use_prune can be overrided by putting optimize_ops in fetch_list
        _origin_fetch_list = fetch_list
        _origin_program = program
        fetch_list, optimize_ops = self._split_optimize_ops_in_fetch_list(
            fetch_list)
        if optimize_ops:
            use_prune = True
        if use_prune:
            cache_key = _get_strong_program_cache_key(program, feed,
                                                      _origin_fetch_list)
            cached_pruned_program = self._get_pruned_program_cache(cache_key)
            if cached_pruned_program is None:
                if isinstance(program, compiler.CompiledProgram):
                    program_scope_cache = self._get_pruned_program_scope_cache(
                        str(id(_origin_program)))
                    # copy the original program, so it can be cached.
                    program = copy.copy(program)
                    # share the local scopes for same original CompiledProgram.
                    program._share_vars_from = program_scope_cache
                    if self._get_pruned_program_scope_cache(
                            str(id(_origin_program))) is None:
                        self._add_pruned_program_scope_cache(
                            str(id(_origin_program)), program)
                pruned_program = self._prune_program(program, feed, fetch_list,
                                                     optimize_ops)
                self._add_pruned_program_cache(cache_key, pruned_program)
            else:
                pruned_program = cached_pruned_program

            feed = self._update_feed(pruned_program, feed)
            program = pruned_program

        def _can_use_interpreter_core(program, place):
            if core.is_compiled_with_mlu() or isinstance(
                    place, core.CustomPlace):
                return False

            use_standalone_executor_for_distribution = os.environ.get(
                'FLAGS_CONVERT_GRAPH_TO_PROGRAM',
                None) in [1, '1', True, 'True', 'true']

            compiled = isinstance(program,
                                  compiler.CompiledProgram) or isinstance(
                                      program._graph, compiler.CompiledProgram)
            if compiled:
                compiled_program = program if isinstance(
                    program, compiler.CompiledProgram) else program._graph

                # delete this code after supporting compiled_program._graph
                if compiled_program._program is None:
                    warnings.warn("Standalone executor is not used for Graph",
                                  UserWarning)
                    return use_standalone_executor_for_distribution

                # delete this code after supporting distribution
                if compiled_program._build_strategy is not None and (
                        compiled_program._build_strategy.is_distribution
                        or compiled_program._build_strategy.num_trainers > 1):
                    warnings.warn(
                        "Standalone executor is not used for distribution",
                        UserWarning)
                    return use_standalone_executor_for_distribution

                # Unsupported case 1: data parallel
                if compiled_program._is_data_parallel and len(
                        compiled_program._get_places(
                            place, compiled_program._places)) != 1:
                    warnings.warn(
                        "Standalone executor is not used for data parallel",
                        UserWarning)
                    return False

                # Unsupported case 2: parallel graph
                if core.globals()['FLAGS_enable_parallel_graph'] in [
                        1, '1', True, 'True', 'true'
                ]:
                    warnings.warn(
                        "Standalone executor is not used for parallel graph",
                        UserWarning)
                    return False

                # Unsupported case 3: inference
                if compiled_program._is_inference:
                    warnings.warn(
                        "Standalone executor is not used for inference",
                        UserWarning)
                    return False

                # Unsupported case 4: CUDA Graph
                if compiled_program._build_strategy is not None and compiled_program._build_strategy.allow_cuda_graph_capture:
                    warnings.warn(
                        "Standalone executor is not used for CUDA Graph",
                        UserWarning)
                    return False

                # Unsupported case 5: async mode
                if compiled_program._build_strategy is not None and compiled_program._build_strategy.async_mode:
                    warnings.warn(
                        "Standalone executor is not used for async mode",
                        UserWarning)
                    return False

            # delete this code after supporting fleet
            from paddle.distributed.fleet import fleet
            if fleet._role_maker is not None:
                warnings.warn("Standalone executor is not used for fleet",
                              UserWarning)
                return use_standalone_executor_for_distribution

            return True

        # NOTE: This is an experimental feature. If `export FLAGS_USE_STANDALONE_EXECUTOR=1 `,
        # use StandaloneExecutor to run the program.
        if return_merged and self._enable_interpreter_core and _can_use_interpreter_core(
                program, self.place):

            if feed is None:
                feed = {}
            elif isinstance(feed, (list, tuple)):
                assert len(feed) == 1, "Not compiled with data parallel"
                feed = feed[0]
            if not isinstance(feed, dict):
                raise TypeError(
                    "feed requires dict as its Parameter. But you passed in %s"
                    % (type(feed)))
            feed = self._update_feed(program, feed)

            program, new_exe = self._executor_cache.get_program_and_executor(
                program, feed, fetch_list, feed_var_name, fetch_var_name,
                self.place, scope)

            self._feed_data(program, feed, feed_var_name, scope)
            if hasattr(program, 'lr_sheduler'):
                from paddle.optimizer.lr import LRScheduler
                assert isinstance(program.lr_sheduler,
                                  LRScheduler), "must be LRScheduler"
                lr_sheduler = program.lr_sheduler
                lr_value = lr_sheduler()
                lr_var = program.global_block().vars[lr_sheduler._var_name]
                data = np.array([lr_value]).astype(convert_dtype(lr_var.dtype))
                tensor = core.get_variable_tensor(scope, lr_sheduler._var_name)
                # NOTE(dev): `tensor.set(data, self.place)` always call TensorCopySync that is a blocking behavior. So we use `_copy_from` to replace it.
                cpu_tensor = _as_lodtensor(data, core.CPUPlace())
                # for ipu, tensor is allocated on cpu
                if core.is_compiled_with_ipu():
                    tensor._copy_from(cpu_tensor, tensor._place())
                else:
                    tensor._copy_from(cpu_tensor, self.place)

            return new_exe.run(scope, list(feed.keys()), fetch_list,
                               return_numpy)

        compiled = isinstance(program, compiler.CompiledProgram)

        # Check if fluid.data() variable no feed data
        if use_prune:
            if compiled:
                global_block = program._program.global_block()
            else:
                global_block = program.global_block()
            for varname in global_block.vars:
                vardesc = global_block.desc.find_var(cpt.to_bytes(varname))
                varobj = global_block.vars[varname]

                # Can not check var build by fluid.layers.data(), bucause fluid.layers.data() had not set need_check_feed
                if vardesc.persistable() == False and \
                    vardesc.type() == core.VarDesc.VarType.LOD_TENSOR and \
                    vardesc.need_check_feed() == True and \
                    varobj.stop_gradient == True and \
                    varobj.is_data == True and \
                    varobj.belong_to_optimizer == False and \
                    varname not in feed:
                    raise ValueError('Need feed data for variable %s' % varname)

        acp._auto_checkpoint(self, program)

        # For backward compatibility, run directly.
        if not compiled:
            # In distributed training, the compiled program is saved in Program._graph
            has_compiled_graph = isinstance(program._graph,
                                            compiler.CompiledProgram)

            if has_compiled_graph:
                program._graph._compile(scope, self.place)
                # _graph in program does not support inference since the _graph is optimized
                # through optimizer.minimize function and should not be used as inference graph
                # assert not program._graph._is_inference
                return self._run_parallel(program._graph,
                                          scope=scope,
                                          feed=feed,
                                          fetch_list=fetch_list,
                                          fetch_var_name=fetch_var_name,
                                          return_numpy=return_numpy,
                                          return_merged=return_merged)

            return self._run_program(program,
                                     feed=feed,
                                     fetch_list=fetch_list,
                                     feed_var_name=feed_var_name,
                                     fetch_var_name=fetch_var_name,
                                     scope=scope,
                                     return_numpy=return_numpy,
                                     use_program_cache=use_program_cache)

        program._compile(scope, self.place)
        if program._is_inference:
            return self._run_inference(program._executor, feed)
        else:
            return self._run_parallel(program,
                                      scope=scope,
                                      feed=feed,
                                      fetch_list=fetch_list,
                                      fetch_var_name=fetch_var_name,
                                      return_numpy=return_numpy,
                                      return_merged=return_merged)

    def _run_program(self, program, feed, fetch_list, feed_var_name,
                     fetch_var_name, scope, return_numpy, use_program_cache):
        from paddle.optimizer.lr import LRScheduler
        if feed is None:
            feed = {}
        elif isinstance(feed, (list, tuple)):
            assert len(feed) == 1, "Not compiled with data parallel"
            feed = feed[0]

        if not isinstance(feed, dict):
            raise TypeError(
                "feed requires dict as its Parameter. But you passed in %s" %
                (type(feed)))

        assert program is not None, "The program should not be Empty"
        if not isinstance(program, Program):
            raise TypeError(
                "Executor requires Program as its Parameter. But you passed in %s"
                % (type(program)))

        if not isinstance(fetch_var_name, str):
            raise TypeError(
                "The name of fetch variable requires string as its Parameter. But you passed in %s"
                % (type(fetch_var_name)))

        if use_program_cache:
            cache_key = _get_strong_program_cache_key(program, feed, fetch_list)
            cached_program = self._get_program_cache(cache_key)
            cached_ctx = self._get_ctx_cache(cache_key)
            cached_scope = self._get_scope_cache(cache_key)
            if cached_program is None:
                cached_program = _add_feed_fetch_ops(
                    program=program,
                    feed=feed,
                    fetch_list=fetch_list,
                    feed_var_name=feed_var_name,
                    fetch_var_name=fetch_var_name)
                self._add_program_cache(cache_key, cached_program)
                fetch_list_str = list(map(_to_name_str, fetch_list))
                cached_ctx = self._default_executor.prepare(
                    cached_program.desc, 0, fetch_list_str, False)
                # currently, we cache program, vars, sub_scope here
                # we suppose that in a life cycle of training, a user
                # will not create many programs. So, here the basic
                # rule of caching is to cache all unseen (program, var, scope)
                # when a user use use_program_cache.
                cached_scope = scope.new_scope()
                self._default_executor.create_variables(cached_program.desc,
                                                        cached_scope, 0)
                self._add_ctx_cache(cache_key, cached_ctx)
                self._add_scope_cache(cache_key, cached_scope)
            program = cached_program
            ctx = cached_ctx
            scope = cached_scope
        else:
            program = _add_feed_fetch_ops(program=program,
                                          feed=feed,
                                          fetch_list=fetch_list,
                                          feed_var_name=feed_var_name,
                                          fetch_var_name=fetch_var_name)

        self._feed_data(program, feed, feed_var_name, scope)
        if hasattr(program, 'lr_sheduler'):
            assert isinstance(program.lr_sheduler,
                              LRScheduler), "must be LRScheduler"
            lr_sheduler = program.lr_sheduler
            lr_value = lr_sheduler()
            lr_var = program.global_block().vars[lr_sheduler._var_name]
            data = np.array([lr_value]).astype(convert_dtype(lr_var.dtype))
            tensor = core.get_variable_tensor(scope, lr_sheduler._var_name)
            tensor.set(data, self.place)

        if not use_program_cache:
            self._default_executor.run(program.desc, scope, 0, True, True,
                                       [fetch_var_name])
        else:
            self._default_executor.run_prepared_ctx(ctx, scope, False, False,
                                                    False)
        arr = scope.find_var(fetch_var_name).get_fetch_list()
        tensors = arr._move_to_list()
        if return_numpy:
            return as_numpy(tensors)
        else:
            return tensors

    def _run_inference(self, exe, feed):
        return exe.run(feed)

    def _check_fetch_list(self, fetch_list):
        is_fetch_var = lambda var: isinstance(var,
                                              (Variable, str, six.string_types))
        is_tuple_list = lambda var: isinstance(var, (tuple, list))

        if fetch_list is None: return []
        if is_fetch_var(fetch_list): return [fetch_list]

        assert is_tuple_list(fetch_list), \
            "Currently , The fetch_list type only should be list or tuple, \n"\
            "but the input type is {}. For more information please refer to \n"\
            "the executor.run(...).".format(type(fetch_list))

        res = []
        for i, var in enumerate(fetch_list):
            if is_fetch_var(var):
                res.append(var)
            # such as [x, 'mean_out', loss]
            elif is_tuple_list(var):
                if all(is_fetch_var(v) for v in var):
                    res.extend(list(var))
                else:
                    res.append(var)
            else:
                raise TypeError(
                    "Require fetch_list[{}] 's type shall be one of (Variable, str), but received {}."
                    .format(i,
                            type(var).__name__))

        return res

    def _dump_debug_info(self, program=None, trainer=None):
        with open(str(id(program)) + "_train_desc.prototxt", "w") as fout:
            fout.write(str(trainer))
        if program._fleet_opt and "fleet_desc" in program._fleet_opt:
            with open("fleet_desc.prototxt", "w") as fout:
                fout.write(str(program._fleet_opt["fleet_desc"]))

    def _adjust_pipeline_resource(self, pipeline_opt, dataset, pipeline_num):
        filelist_length = len(dataset.dataset.get_filelist())
        if filelist_length < pipeline_num:
            pipeline_num = filelist_length
            print(
                "Pipeline training: setting the pipeline num to %d is enough because there are only %d files"
                % (filelist_length, filelist_length))
        if filelist_length < pipeline_num * pipeline_opt["concurrency_list"][0]:
            print(
                "Pipeline training: setting the 1st element in concurrency_list to %d is enough because there are only %d files"
                % (filelist_length // pipeline_num, filelist_length))
            pipeline_opt["concurrency_list"][
                0] = filelist_length // pipeline_num
        dataset.set_thread(pipeline_opt["concurrency_list"][0] * pipeline_num)
        return pipeline_num

    def _prepare_trainer(self,
                         program=None,
                         dataset=None,
                         scope=None,
                         thread=0,
                         debug=False,
                         fetch_list=None,
                         fetch_info=None,
                         print_period=100):
        is_heter = 0
        use_ps_gpu = 0
        if not program._fleet_opt is None:
            if program._fleet_opt.get("worker_class", "") == "HeterCpuWorker":
                is_heter = 1
            if program._fleet_opt.get("trainer", "") == "HeterXpuTrainer":
                is_heter = 1
            if program._fleet_opt.get("use_ps_gpu", False):
                use_ps_gpu = True
        if scope is None:
            scope = global_scope()
        if fetch_list is None:
            fetch_list = []
        if fetch_info is None:
            fetch_info = []
        assert len(fetch_list) == len(fetch_info)
        compiled = isinstance(program, compiler.CompiledProgram)
        if is_heter:
            from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
            from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
            fu = FleetUtil()
            ret = fu.split_program_by_device(program)
        if not compiled:
            # TODO: Need a better way to distinguish and specify different execution mode
            if program._pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program._pipeline_opt)
            elif program._heter_pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program._heter_pipeline_opt)
            else:
                trainer = TrainerFactory()._create_trainer(program._fleet_opt)
                trainer._set_thread_barrier(program._is_distributed)
            trainer._set_program(program)
            if is_heter:
                trainer._set_heter_info(ret)
        else:
            if program._pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program.program._pipeline_opt)
            elif program._heter_pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program.program._heter_pipeline_opt)
            else:
                trainer = TrainerFactory()._create_trainer(
                    program.program._fleet_opt)
            trainer._set_program(program.program)

        if thread <= 0:
            if use_ps_gpu:
                trainer._set_thread(len(program._fleet_opt["worker_places"]))
            elif dataset.thread_num <= 0:
                raise RuntimeError(
                    "You should set thread num first, either in Dataset"
                    "or in Executor.train_from_dataset")
            else:
                trainer._set_thread(dataset.thread_num)
        else:
            trainer._set_thread(thread)

        trainer._set_debug(debug)
        trainer._set_fetch_var_and_info(fetch_list, fetch_info, print_period)
        return scope, trainer

    def _run_from_dataset(self,
                          program=None,
                          dataset=None,
                          scope=None,
                          thread=0,
                          is_infer=False,
                          debug=False,
                          fetch_list=None,
                          fetch_info=None,
                          print_period=100,
                          fetch_handler=None):
        if program._pipeline_opt is not None:
            import paddle
            if dataset is not None:
                raise RuntimeError("dataset should be None for pipeline mode")
            # The following fake dataset is created to call
            # the _prepare_trainer api, and it is meaningless.
            data_vars = []
            for var in program.global_block().vars.values():
                if var.is_data:
                    data_vars.append(var)
            if core.is_compiled_with_npu():
                dataset = paddle.fluid.DatasetFactory().create_dataset(
                    'InMemoryDataset')
            else:
                dataset = paddle.fluid.DatasetFactory().create_dataset(
                    'FileInstantDataset')
            dataset.set_batch_size(1)
            dataset.set_thread(1)
            dataset.set_filelist(['None'])
            dataset.set_use_var(data_vars)
        elif program._heter_pipeline_opt is not None:
            stage_id = program._heter_pipeline_opt["pipeline_stage"]
            #print("test_fl_stage_id: {}".format(stage_id))
            heter_place = program._heter_pipeline_opt["heter_place"]
            if stage_id != 0:
                if "is_fl_mode" not in program._heter_pipeline_opt:
                    import paddle
                    if dataset is not None:
                        raise RuntimeError(
                            "dataset should be None for heter pipeline mode")
                    # The following fake dataset is created to call
                    # the _prepare_trainer api, and it is meaningless.
                    data_vars = []
                    for var in program.global_block().vars.values():
                        if var.is_data:
                            data_vars.append(var)
                    dataset = paddle.fluid.DatasetFactory().create_dataset(
                        'InMemoryDataset')
                    dataset.set_batch_size(1)
                    dataset.set_thread(1)
                    dataset.set_filelist(['None'])
                    dataset.set_use_var(data_vars)
            else:
                if dataset is None:
                    raise RuntimeError(
                        "dataset is need and should be initialized")
            ## change default executor
            heter_place = framework._get_paddle_place(heter_place)
            p = core.Place()
            p.set_place(heter_place)
            self._default_executor = core.Executor(p)
        else:
            if dataset is None:
                raise RuntimeError("dataset is need and should be initialized")

        dataset._prepare_to_run()
        real_fetch_list = []
        if program._pipeline_opt:
            real_program = program._pipeline_opt["section_program"]
            for fetch_var in fetch_list:
                if isinstance(fetch_var, Variable):
                    fetch_var_name = fetch_var.name
                else:
                    fetch_var_name = fetch_var
                if fetch_var_name in real_program.global_block().vars:
                    real_fetch_list.append(fetch_var)

            program._pipeline_opt["section_program"] = _add_feed_fetch_ops(
                program=program._pipeline_opt["section_program"],
                feed=[],
                fetch_list=real_fetch_list,
                feed_var_name='feed',
                fetch_var_name='fetch')
            main_block = program._pipeline_opt["section_program"].block(0)
            for op in main_block.ops:
                # set the op_role of fetch op to Optimize to avoid
                # erase the fetched vars by gc for pipeline
                if op.type == 'fetch':
                    op._set_attr(
                        'op_role',
                        core.op_proto_and_checker_maker.OpRole.Optimize)
            fetch_list = None
        scope, trainer = self._prepare_trainer(program=program,
                                               dataset=dataset,
                                               scope=scope,
                                               thread=thread,
                                               debug=debug,
                                               fetch_list=fetch_list,
                                               fetch_info=fetch_info,
                                               print_period=print_period)

        trainer._set_infer(is_infer)
        trainer._gen_trainer_desc()

        if program._pipeline_opt is None:
            if program._heter_pipeline_opt is None:
                self._dump_debug_info(program=program, trainer=trainer)
        # warning if dataset not set psgpu in psgpu mode
        if dataset.use_ps_gpu is False and trainer.proto_desc.use_ps_gpu:
            logging.warning("dataset should call set_use_ps_gpu in PsGpu mode")

        dataset._dynamic_adjust_before_train(trainer.proto_desc.thread_num)

        if program._heter_pipeline_opt is None:
            trainer_instance = self._default_executor.init_for_dataset(  # -->InitForDataset
                program.desc, trainer._desc(), scope, dataset.dataset)
        else:
            # cache trainer instance for heterps pipeline training
            if fetch_list == None:
                fetch_list = []
            cache_key = _get_strong_program_cache_key(program, None, fetch_list)
            trainer_instance = self._get_trainer_cache(cache_key)
            if trainer_instance is None:
                trainer_instance = self._default_executor.init_for_dataset(
                    program.desc, trainer._desc(), scope, dataset.dataset)
                #print("test_fl_ps - trainer_desc: {}\n".format(trainer))
                self._add_trainer_cache(cache_key, trainer_instance)
            else:
                trainer_instance.ResetDataset(dataset.dataset)

        if fetch_handler is not None:
            scope0 = trainer_instance.get_worker_scope(0)
            fetch_monitor = FetchHandlerMonitor(scope0, fetch_handler)
            fetch_monitor.start()
            self._default_executor.run_from_dataset(trainer_instance)
            fetch_monitor.stop()
            if program._heter_pipeline_opt is None:
                self._default_executor.release_trainer(trainer_instance)
        else:
            self._default_executor.run_from_dataset(trainer_instance)
            if program._heter_pipeline_opt is None:
                self._default_executor.release_trainer(trainer_instance)

        dataset._dynamic_adjust_after_train()
        dataset._finish_to_run()
        if real_fetch_list:
            arr = scope.find_var('fetch').get_fetch_list()
            tensors = arr._move_to_list()
            return as_numpy(tensors)

        return None

    def _prepare_pipeline_ctx(self,
                              program=None,
                              dataset=None,
                              scope=None,
                              thread=0,
                              is_infer=False,
                              debug=False,
                              fetch_list=None,
                              fetch_info=None,
                              print_period=100,
                              fetch_handler=None,
                              use_program_cache=False):
        assert program._pipeline_opt is not None
        assert dataset is None, "dataset should be None for pipeline mode"

        cache_key = _get_strong_program_cache_key(program, None, fetch_list)
        ctx = self._get_ctx_cache(cache_key)
        if use_program_cache and ctx is not None:
            return ctx

        import paddle

        # The following fake dataset is created to call
        # the _prepare_trainer api, and it is meaningless.
        def _get_dataset():
            data_vars = []
            for var in program.global_block().vars.values():
                if var.is_data:
                    data_vars.append(var)
            if core.is_compiled_with_npu():
                dataset = paddle.fluid.DatasetFactory().create_dataset(
                    'InMemoryDataset')
            else:
                dataset = paddle.fluid.DatasetFactory().create_dataset(
                    'FileInstantDataset')
            dataset.set_batch_size(1)
            dataset.set_thread(1)
            dataset.set_filelist(['None'])
            dataset.set_use_var(data_vars)
            dataset._prepare_to_run()
            return dataset

        dataset = _get_dataset()

        def _get_real_program_fetch_list():
            real_program = program._pipeline_opt["section_program"]
            real_fetch_list = []
            for fetch_var in fetch_list:
                if isinstance(fetch_var, Variable):
                    fetch_var_name = fetch_var.name
                else:
                    fetch_var_name = fetch_var
                if fetch_var_name in real_program.global_block().vars:
                    real_fetch_list.append(fetch_var)

            real_program = _add_feed_fetch_ops(program=real_program,
                                               feed=[],
                                               fetch_list=real_fetch_list,
                                               feed_var_name='feed',
                                               fetch_var_name='fetch')
            main_block = real_program.block(0)
            for op in main_block.ops:
                # set the op_role of fetch op to Optimize to avoid
                # erase the fetched vars by gc for pipeline
                if op.type == 'fetch':
                    op._set_attr(
                        'op_role',
                        core.op_proto_and_checker_maker.OpRole.Optimize)
            return real_program, real_fetch_list

        real_program, real_fetch_list = _get_real_program_fetch_list()

        program._pipeline_opt["section_program"] = real_program
        fetch_list = None

        scope, trainer = self._prepare_trainer(program=program,
                                               dataset=dataset,
                                               scope=scope,
                                               thread=thread,
                                               debug=debug,
                                               fetch_list=fetch_list,
                                               fetch_info=fetch_info,
                                               print_period=print_period)

        trainer._set_infer(is_infer)
        trainer._gen_trainer_desc()

        # NOTE: only for debug, very slow
        # self._dump_debug_info(program=program, trainer=trainer)

        # warning if dataset not set psgpu in psgpu mode
        if dataset.use_ps_gpu is False and trainer.proto_desc.use_ps_gpu:
            logging.warning("dataset should call set_use_ps_gpu in PsGpu mode")
        dataset._dynamic_adjust_before_train(trainer.proto_desc.thread_num)

        trainer_desc = trainer._desc()  # slow, cache
        trainer_instance = self._default_executor.init_for_dataset(
            program.desc, trainer_desc, scope, dataset.dataset)

        ctx = [scope, real_fetch_list, trainer_instance]
        if use_program_cache: self._add_ctx_cache(cache_key, ctx)

        return ctx

    def _prepare_fleet_executor_carrier(self,
                                        carrier_id="",
                                        program=None,
                                        scope=None,
                                        fleet_opt=None,
                                        with_standalone_executor=False):
        num_micro_batches = fleet_opt[
            "num_micro_batches"] if "num_micro_batches" in fleet_opt else 1
        cur_rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "").split(',')
        nrank = len(trainer_endpoints)

        assert 'scheduler' in fleet_opt or 'tasks' in fleet_opt, \
            "Fleet executor need configuration for scheduler, you can choose from 1F1B or Origin. " \
            "Or you can provide a list of task nodes to init fleet executor directly."
        if 'tasks' in fleet_opt:
            assert 'task_id_to_rank' in fleet_opt, "If you provide tasks to init fleet executor," \
                                                   " task_id_to_rank should also be provided."
            print('fleet executor will use user defined task nodes')
            tasks = [task.task_node() for task in fleet_opt['tasks']]
            task_id_to_rank = fleet_opt['task_id_to_rank']
        else:
            scheduler = fleet_opt['scheduler']
            if scheduler == '1F1B':
                from paddle.distributed.fleet.fleet_executor_utils import run1f1b
                if "dist_strategy" not in fleet_opt or \
                   "pp_degree" not in fleet_opt["dist_strategy"] or \
                   fleet_opt["dist_strategy"]["pp_degree"] == 1:
                    warnings.warn("Using 1F1B scheduler with pp_degree == 1.")
                tasks, task_id_to_rank = run1f1b(
                    program, cur_rank, fleet_opt.get('num_micro_batches', 1),
                    fleet_opt.get('dist_strategy', {}), nrank,
                    with_standalone_executor)
            elif scheduler == 'Origin':
                from paddle.distributed.fleet.fleet_executor_utils import origin
                if "dist_strategy" in fleet_opt and \
                   "pp_degree" in fleet_opt["dist_strategy"]:
                    assert fleet_opt["dist_strategy"]["pp_degree"] == 1, \
                        "For pipeline mode, the scheduler should be 1F1B instead of Origin."
                if "num_micro_batches" in fleet_opt:
                    assert fleet_opt["num_micro_batches"] == 1, \
                        "For origin scheduler mode, the num micro batches should be 1."
                tasks, task_id_to_rank = origin(program, cur_rank)
            else:
                raise "Fleet_executor only supports 1F1B and Origin scheduler, " \
                      "but received " + str(scheduler) + "."
            # NOTE: have to hold these vars, otherwise will be destructed
            fleet_opt['tasks'] = tasks
            fleet_opt['task_id_to_rank'] = task_id_to_rank
        place = core.Place()
        place.set_place(self.place)
        # NOTE: the last argument is used to force create some vars in root scope,
        # won't be used during train.
        self._fleet_executor.init(carrier_id, program.desc, scope, place,
                                  num_micro_batches, tasks, task_id_to_rank, [])

    def _run_using_fleet_executor(self,
                                  program=None,
                                  feed=None,
                                  feed_var_name="feed",
                                  fetch_var_name="fetch",
                                  fetch_list=None,
                                  with_standalone_executor=False):
        cache_key = _get_strong_program_cache_key(program, feed, fetch_list)
        cached_program = self._get_program_cache(cache_key)
        cached_scope = self._get_scope_cache(cache_key)
        if cached_scope is None:
            cached_scope = global_scope()
            self._add_scope_cache(cache_key, cached_scope)
        if cached_program is None:
            assert program._pipeline_opt, "program should have _pipeline_opt to start carrier"
            real_feed = [] if feed is None else feed
            real_program = program
            if "section_program" in program._pipeline_opt:
                real_program = program._pipeline_opt["section_program"]
            cached_program = _add_feed_fetch_ops(program=real_program,
                                                 feed=real_feed,
                                                 fetch_list=fetch_list,
                                                 feed_var_name=feed_var_name,
                                                 fetch_var_name=fetch_var_name)
            main_block = cached_program.block(0)
            for op in main_block.ops:
                # set the op_role of fetch op to Optimize to avoid
                # erase the fetched vars by gc for pipeline
                if op.type == 'fetch':
                    op._set_attr(
                        'op_role',
                        core.op_proto_and_checker_maker.OpRole.Optimize)
            self._add_program_cache(cache_key, cached_program)
            fleet_opt = program._pipeline_opt["fleet_opt"]
            if 'tasks' in fleet_opt:
                # Insert feed/fetch op for cloned program in each task node,
                # these ops has already been inserted into the origin program.
                # To avoid every task nodes all have feed/fetch ops,
                # only insert feed ops into the first task node,
                # then insert fetch ops into the last task node.

                # Insert feed ops
                feed_task = fleet_opt['tasks'][0]
                print("Inserting feed ops for task", feed_task.task_id())
                feed_program = feed_task.get_program()
                feed_program = self._add_feed_ops(program=feed_program,
                                                  feed=real_feed,
                                                  feed_var_name=feed_var_name)
                feed_task.set_program(feed_program)

                # Insert fetch ops
                fetch_task = fleet_opt['tasks'][-1]
                print("Inserting fetch ops for task", fetch_task.task_id())
                fetch_program = fetch_task.get_program()
                fetch_program = self._add_fetch_ops(
                    program=fetch_program,
                    fetch_list=fetch_list,
                    fetch_var_name=fetch_var_name)
                main_block = fetch_program.block(0)
                for op in main_block.ops:
                    # set the op_role of fetch op to Optimize to avoid
                    # erase the fetched vars by gc for pipeline
                    if op.type == 'fetch':
                        op._set_attr(
                            'op_role',
                            core.op_proto_and_checker_maker.OpRole.Optimize)
                fetch_task.set_program(fetch_program)

            self._prepare_fleet_executor_carrier(
                cache_key,
                program=cached_program,
                scope=cached_scope,
                fleet_opt=fleet_opt,
                with_standalone_executor=with_standalone_executor)

        if feed:
            # NOTE: don't have to traverse programs in task nodes,
            # since they all sub program of cached program and
            # cached program is also added feed fetch var
            self._feed_data(cached_program, feed, feed_var_name, cached_scope)

        from paddle.optimizer.lr import LRScheduler
        if hasattr(program, 'lr_sheduler'):
            lr_sheduler = program.lr_sheduler
            assert isinstance(lr_sheduler, LRScheduler), "must be LRScheduler"
            lr_value = lr_sheduler()
            lr_var = program.global_block().vars[lr_sheduler._var_name]
            data = np.array([lr_value]).astype(convert_dtype(lr_var.dtype))
            tensor = core.get_variable_tensor(cached_scope,
                                              lr_sheduler._var_name)
            tensor.set(data, self.place)

        self._fleet_executor.run(cache_key)

        if fetch_list:
            arr = cached_scope.find_var(fetch_var_name).get_fetch_list()
            tensors = arr._move_to_list()
            return as_numpy(tensors)
        return None

    def _add_feed_ops(self, program, feed, feed_var_name):
        tmp_program = program.clone()

        global_block = tmp_program.global_block()

        if feed_var_name in global_block.vars:
            feed_var = global_block.var(feed_var_name)
        else:
            feed_var = global_block.create_var(
                name=feed_var_name,
                type=core.VarDesc.VarType.FEED_MINIBATCH,
                persistable=True)

        # prepend feed operators
        if not has_feed_operators(global_block, feed, feed_var_name):
            for i, name in enumerate(feed):
                if global_block.has_var(name):
                    out = global_block.var(name)
                    global_block._prepend_op(type='feed',
                                             inputs={'X': [feed_var]},
                                             outputs={'Out': [out]},
                                             attrs={'col': i})
                else:
                    warnings.warn(
                        "The variable %s is not found in program. It is not declared or is pruned."
                        % name)

        return tmp_program

    @classmethod
    def _add_fetch_ops(cls,
                       program,
                       fetch_list,
                       fetch_var_name,
                       use_fetch_v2=False):
        tmp_program = program.clone()

        global_block = tmp_program.global_block()

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True)

        if use_fetch_v2:
            fetch_op = 'fetch_v2'
        else:
            fetch_op = 'fetch'

        # append fetch_operators
        if not has_fetch_operators(global_block, fetch_list, fetch_var_name,
                                   fetch_op):
            for i, var in enumerate(fetch_list):
                assert isinstance(var, Variable) or isinstance(
                    var,
                    six.string_types), ("Wrong type for fetch_list[%s]: %s" %
                                        (i, type(var)))
                global_block.append_op(type=fetch_op,
                                       inputs={'X': [var]},
                                       outputs={'Out': [fetch_var]},
                                       attrs={'col': i})

        return tmp_program

    @classmethod
    def _remove_fetch_ops(cls, program, fetch_op_name='fetch'):
        tmp_program = program.clone()
        global_block = tmp_program.global_block()
        op_num = len(global_block.ops)
        for idx in reversed(range(op_num)):
            if global_block.ops[idx].type == fetch_op_name:
                global_block._remove_op(idx)

        return tmp_program

    def _run_pipeline(self,
                      program=None,
                      dataset=None,
                      scope=None,
                      thread=0,
                      is_infer=False,
                      debug=False,
                      fetch_list=None,
                      fetch_info=None,
                      print_period=100,
                      fetch_handler=None,
                      use_program_cache=False):
        scope, real_fetch_list, trainer_instance = \
            self._prepare_pipeline_ctx(program, dataset, scope, thread,
                                       is_infer, debug, fetch_list, fetch_info,
                                       print_period, fetch_handler,
                                       use_program_cache)

        from paddle.optimizer.lr import LRScheduler
        if hasattr(program, 'lr_sheduler'):
            lr_sheduler = program.lr_sheduler
            assert isinstance(lr_sheduler, LRScheduler), "must be LRScheduler"
            lr_value = lr_sheduler()
            lr_var = program.global_block().vars[lr_sheduler._var_name]
            data = np.array([lr_value]).astype(convert_dtype(lr_var.dtype))
            tensor = core.get_variable_tensor(scope, lr_sheduler._var_name)
            tensor.set(data, self.place)

        self._default_executor.run_from_dataset(trainer_instance)

        if not use_program_cache:
            self._default_executor.release_trainer(trainer_instance)

        if real_fetch_list:
            arr = scope.find_var('fetch').get_fetch_list()
            tensors = arr._move_to_list()
            return as_numpy(tensors)

        return None

    def infer_from_dataset(self,
                           program=None,
                           dataset=None,
                           scope=None,
                           thread=0,
                           debug=False,
                           fetch_list=None,
                           fetch_info=None,
                           print_period=100,
                           fetch_handler=None):
        """
        Infer from a pre-defined Dataset. Dataset is defined in paddle.fluid.dataset.
        Given a program, either a program or compiled program, infer_from_dataset will
        consume all data samples in dataset. Input scope can be given by users. By default,
        scope is global_scope(). The total number of thread run in training is `thread`.
        Thread number used in training will be minimum value of threadnum in Dataset and
        the value of thread in this interface. Debug can be set so that executor will display
        Run-Time for all operators and the throughputs of current infer task.

        The document of infer_from_dataset is almost the same as train_from_dataset,
        except that in distributed training, push gradients will be disabled in infer_from_dataset.
        infer_from_dataset() can be used for evaluation in multi-threadvery easily.

        Args:
            program(Program|CompiledProgram): the program that needs to be run,
                if not provided, then default_main_program (not compiled) will be used.
            dataset(paddle.fluid.Dataset): dataset created outside this function,
                a user should provide a well-defined dataset before calling this function.
                Please check the document of Dataset if needed. default is None
            scope(Scope): the scope used to run this program, you can switch it to different scope
                for each run. default is global_scope
            thread(int): number of thread a user wants to run in this function. Default is 0, which
                means using thread num of dataset
            debug(bool): whether a user wants to run infer_from_dataset, default is False
            fetch_list(Tensor List): fetch Tensor list, each Tensor will be printed during
                training, default is None
            fetch_info(String List): print information for each Tensor, default is None
            print_period(int): the number of mini-batches for each print, default is 100
            fetch_handler(FetchHandler): a user define class for fetch output.

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle

                paddle.enable_static()
                place = paddle.CPUPlace()  # you can set place = paddle.CUDAPlace(0) to use gpu
                exe = paddle.static.Executor(place)
                x = paddle.static.data(name="x", shape=[None, 10, 10], dtype="int64")
                y = paddle.static.data(name="y", shape=[None, 1], dtype="int64", lod_level=1)
                dataset = paddle.fluid.DatasetFactory().create_dataset()
                dataset.set_use_var([x, y])
                dataset.set_thread(1)
                # you should set your own filelist, e.g. filelist = ["dataA.txt"]
                filelist = []
                dataset.set_filelist(filelist)
                exe.run(paddle.static.default_startup_program())
                exe.infer_from_dataset(program=paddle.static.default_main_program(),
                                       dataset=dataset)

        """
        return self._run_from_dataset(program, dataset, scope, thread, True,
                                      debug, fetch_list, fetch_info,
                                      print_period, fetch_handler)

    def start_heter_trainer(self,
                            program=None,
                            scope=None,
                            debug=False,
                            fetch_list=None,
                            fetch_info=None,
                            print_period=100,
                            fetch_handler=None):
        scope, trainer = self._prepare_trainer(program=program,
                                               dataset=None,
                                               scope=scope,
                                               thread=1,
                                               debug=debug,
                                               fetch_list=fetch_list,
                                               fetch_info=fetch_info,
                                               print_period=print_period)

        trainer._set_infer(False)
        trainer._gen_trainer_desc()

        self._dump_debug_info(program=program, trainer=trainer)

        trainer_instance = self._default_executor.init_for_dataset(
            program.desc, trainer._desc(), scope, None)

        #if fetch_handler is not None:
        #    scope0 = trainer_instance.get_worker_scope(0)
        #    fetch_monitor = FetchHandlerMonitor(scope0, fetch_handler)
        #    fetch_monitor.start()
        #    self._default_executor.run_from_dataset(trainer_instance)
        #    fetch_monitor.stop()
        #    self._default_executor.release_trainer(trainer_instance)
        #else:

        self._default_executor.run_from_dataset(trainer_instance)
        #self._default_executor.release_trainer(trainer_instance)

        return trainer_instance

    def train_from_dataset(self,
                           program=None,
                           dataset=None,
                           scope=None,
                           thread=0,
                           debug=False,
                           fetch_list=None,
                           fetch_info=None,
                           print_period=100,
                           fetch_handler=None):
        """
        Train from a pre-defined Dataset. Dataset is defined in paddle.fluid.dataset.
        Given a program, either a program or compiled program, train_from_dataset will
        consume all data samples in dataset. Input scope can be given by users. By default,
        scope is global_scope(). The total number of thread run in training is `thread`.
        Thread number used in training will be minimum value of threadnum in Dataset and
        the value of thread in this interface. Debug can be set so that executor will display
        Run-Time for all operators and the throughputs of current training task.

        Note: train_from_dataset will destroy all resources created within executor for each run.

        Args:
            program(Program|CompiledProgram): the program that needs to be run,
                if not provided, then default_main_program (not compiled) will be used.
            dataset(paddle.fluid.Dataset): dataset created outside this function,
                a user should provide a well-defined dataset before calling this function.
                Please check the document of Dataset if needed.
            scope(Scope): the scope used to run this program, you can switch it to different scope
                for each run. default is global_scope
            thread(int): number of thread a user wants to run in this function. Default is 0, which
                means using thread num of dataset
            debug(bool): whether a user wants to run train_from_dataset 
            fetch_list(Tensor List): fetch Tensor list, each variable will be printed
                during training
            fetch_info(String List): print information for each Tensor, its length should be equal
                to fetch_list
            print_period(int): the number of mini-batches for each print, default is 100
            fetch_handler(FetchHandler): a user define class for fetch output.

        Returns:
            None
        
        Examples:
        
            .. code-block:: python

              import paddle

              paddle.enable_static()
              place = paddle.CPUPlace() # you can set place = paddle.CUDAPlace(0) to use gpu
              exe = paddle.static.Executor(place)
              x = paddle.static.data(name="x", shape=[None, 10, 10], dtype="int64")
              y = paddle.static.data(name="y", shape=[None, 1], dtype="int64", lod_level=1)
              dataset = paddle.fluid.DatasetFactory().create_dataset()
              dataset.set_use_var([x, y])
              dataset.set_thread(1)
              # you should set your own filelist, e.g. filelist = ["dataA.txt"]
              filelist = []
              dataset.set_filelist(filelist)
              exe.run(paddle.static.default_startup_program())
              exe.train_from_dataset(program=paddle.static.default_main_program(),
                                     dataset=dataset)

        """
        return self._run_from_dataset(program, dataset, scope, thread, False,
                                      debug, fetch_list, fetch_info,
                                      print_period, fetch_handler)
