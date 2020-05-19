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
from .framework import Program, default_main_program, Variable, Operator, convert_np_dtype_to_dtype_
from . import core
from . import compiler
from .. import compat as cpt
from .trainer_factory import TrainerFactory
from .trainer_factory import FetchHandlerMonitor
import copy

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

          import paddle.fluid as fluid
          import numpy

          fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
          numpy.array(fluid.global_scope().find_var("data").get_tensor())
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
    :api_attr: Static Graph
    
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

            import paddle.fluid as fluid
            import numpy

            new_scope = fluid.Scope()
            with fluid.scope_guard(new_scope):
                 fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
            numpy.array(new_scope.find_var("data").get_tensor())
    """

    ex = _switch_scope(scope)
    try:
        yield
    finally:
        _switch_scope(ex)


def as_numpy(tensor):
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

    Returns:
        numpy.ndarray
    """
    if isinstance(tensor, core.LoDTensorArray):
        return [as_numpy(t) for t in tensor]
    if isinstance(tensor, list):
        return [as_numpy(t) for t in tensor]
    assert isinstance(tensor, core.LoDTensor)
    lod = tensor.lod()
    if len(lod) > 0:
        raise RuntimeError("Some of your fetched tensors hold LoD information. \
            They can not be completely cast to Python ndarray. \
            Please set the parameter 'return_numpy' as 'False' to \
            return LoDTensor itself directly.")
    if tensor._is_initialized():
        return np.array(tensor)
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
                raise Exception("'feed_targets' does not have {} variable".
                                format(feed_target_name))
        else:
            break
    if feed_count > 0 and feed_count != len(feed_targets):
        raise Exception(
            "Feed operators in program desc do not match 'feed_targets'")
    return feed_count > 0


def has_fetch_operators(block, fetch_targets, fetch_holder_name):
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

    Return:
        A boolean value that indicates whether a block has fetch operators
        that match the info contained in fetch_targets and fetch_holder_name.
    """

    fetch_count = 0
    for op in block.ops:
        if op.desc.type() == 'fetch':
            fetch_count += 1
            assert op.desc.output('Out')[0] == fetch_holder_name
            fetch_target_name = op.desc.input('X')[0]
            if fetch_target_name not in [
                    var.desc.name() for var in fetch_targets
            ]:
                raise Exception("'fetch_targets' does not have {} variable".
                                format(fetch_target_name))
            idx = op.desc.attr('col')
            assert fetch_target_name == fetch_targets[idx].desc.name()
    if fetch_count > 0 and fetch_count != len(fetch_targets):
        raise Exception(
            "Fetch operators in program desc do not match 'fetch_targets'")
    return fetch_count > 0


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
        tensor = as_numpy(tensor)
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


def _get_strong_program_cache_key(program, feed, fetch_list):
    return str(id(program)) + _get_program_cache_key(feed, fetch_list)


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
            data(numpy.ndarray): a instance of array
            data(core.Place): the place of created tensor
            dtype(core.VarDesc.VarType): the expected data type of created tensor

        Returns:
            LoDTensor
        """
    if isinstance(data, list):
        raise RuntimeError("Some of your feed data hold LoD information. \
                They can not be completely cast from a list of Python \
                ndarray to LoDTensor. Please convert data to LoDTensor \
                directly before feeding the data.\
                ")

    #NOTE(zhiqiu): convert python builtin ,like float and int, to numpy array
    if not isinstance(data, np.ndarray):
        if np.isscalar(data):
            assert dtype is not None, 'dtype should be given when casting python scalar to tensor'
            dtype = convert_dtype(dtype) if isinstance(
                dtype, core.VarDesc.VarType) else dtype
            data = np.array([data]).astype(dtype)

    # single tensor case
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


class Executor(object):
    """
    :api_attr: Static Graph

    An Executor in Python, supports single/multiple-GPU running,
    and single/multiple-CPU running.

    Args:
        place(fluid.CPUPlace()|fluid.CUDAPlace(n)|None): This parameter represents
            which device the executor runs on. When this parameter is None, PaddlePaddle
            will set the default device according to its installation version. If Paddle
            is CPU version, the default device would be set to `CPUPlace()` . If Paddle is
            GPU version, the default device would be set to `CUDAPlace(0)` . Default is None.

    Returns:
        Executor

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.fluid.compiler as compiler
          import numpy
          import os

          # Set place explicitly.
          # use_cuda = True
          # place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
          # exe = fluid.Executor(place)

          # If you don't set place, PaddlePaddle sets the default device.
          exe = fluid.Executor()

          train_program = fluid.Program()
          startup_program = fluid.Program()
          with fluid.program_guard(train_program, startup_program):
              data = fluid.data(name='X', shape=[None, 1], dtype='float32')
              hidden = fluid.layers.fc(input=data, size=10)
              loss = fluid.layers.mean(hidden)
              fluid.optimizer.SGD(learning_rate=0.01).minimize(loss)

          # Run the startup program once and only once.
          # Not need to optimize/compile the startup program.
          startup_program.random_seed=1
          exe.run(startup_program)

          # Run the main program directly without compile.
          x = numpy.random.random(size=(10, 1)).astype('float32')
          loss_data, = exe.run(train_program,
                               feed={"X": x},
                               fetch_list=[loss.name])

          # Or, compiled the program and run. See `CompiledProgram`
          # for more detail.
          # NOTE: If you use CPU to run the program or Paddle is
          # CPU version, you need to specify the CPU_NUM, otherwise,
          # fluid will use all the number of the logic core as
          # the CPU_NUM, in that case, the batch size of the input
          # should be greater than CPU_NUM, if not, the process will be
          # failed by an exception.

          # Set place explicitly.
          # if not use_cuda:
          #     os.environ['CPU_NUM'] = str(2)

          # If you don't set place and PaddlePaddle is CPU version
          os.environ['CPU_NUM'] = str(2)

          compiled_prog = compiler.CompiledProgram(
              train_program).with_data_parallel(
              loss_name=loss.name)
          loss_data, = exe.run(compiled_prog,
                               feed={"X": x},
                               fetch_list=[loss.name])
    """

    def __init__(self, place=None):
        if place is None:
            if core.is_compiled_with_cuda():
                self.place = core.CUDAPlace(0)
            else:
                self.place = core.CPUPlace()
        else:
            self.place = place
        self.program_caches = dict()
        self.ctx_caches = dict()
        self.scope_caches = dict()
        self.var_caches = dict()
        self.pruned_program_caches = dict()
        p = core.Place()
        p.set_place(self.place)
        self._default_executor = core.Executor(p)
        self._closed = False
        self.pruned_program_scope_caches = dict()

    def _get_scope_cache(self, program_cache_key):
        return self.scope_caches.get(program_cache_key, None)

    def _get_ctx_cache(self, program_cache_key):
        return self.ctx_caches.get(program_cache_key, None)

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

    def _add_scope_cache(self, scope_cache_key, scope):
        self.scope_caches[scope_cache_key] = scope

    def _add_feed_fetch_ops(self, program, feed, fetch_list, feed_var_name,
                            fetch_var_name):
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
                    global_block._prepend_op(
                        type='feed',
                        inputs={'X': [feed_var]},
                        outputs={'Out': [out]},
                        attrs={'col': i})
                else:
                    warnings.warn(
                        "The variable %s is not found in program. It is not declared or is pruned."
                        % name)
        # append fetch_operators
        if not has_fetch_operators(global_block, fetch_list, fetch_var_name):
            for i, var in enumerate(fetch_list):
                assert isinstance(var, Variable) or isinstance(
                    var, six.string_types), (
                        "Wrong type for fetch_list[%s]: %s" % (i, type(var)))
                global_block.append_op(
                    type='fetch',
                    inputs={'X': [var]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i})

        return tmp_program

    def _feed_data(self, program, feed, feed_var_name, scope):
        # feed var to framework
        global_block = program.global_block()
        for op in global_block.ops:
            if op.desc.type() == 'feed':
                feed_target_name = op.desc.output('Out')[0]
                cur_feed = feed[feed_target_name]
                var = global_block.var(feed_target_name)
                if not isinstance(cur_feed, core.LoDTensor):
                    cur_feed = _as_lodtensor(cur_feed, self.place, var.dtype)
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

    def _split_optimize_ops_in_fetch_list(self, fetch_list):
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
                    "The item in fetch_list should be str, variable or optimize_op, but recieved %s.",
                    type(item))

        for item in fetch_list:
            # NOTE(zhiqiu): to support (optimizer_ops, param_and_grads) and optimizer_ops in fetch_list
            # we should handle tuple and list in fetch_list.
            # TODO(zhiqiu): find a better way to handle that.
            if isinstance(item, list):
                for i in item:
                    _get_targets(_optimize_ops, _fetch_list, i)
            elif isinstance(item, tuple):
                for i in item[0]:
                    _get_targets(_optimize_ops, _fetch_list, i)
            else:
                _get_targets(_optimize_ops, _fetch_list, item)

        return _fetch_list, _optimize_ops

    def _prune_program(self,
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

    def _update_feed(self, program, feed):
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

              import paddle.fluid as fluid

              cpu = fluid.CPUPlace()
              exe = fluid.Executor(cpu)
              # execute training or testing
              exe.close()
        """
        if not self._closed:
            self._default_executor.close()
            self._closed = True

    def _run_parallel(self, program, scope, feed, fetch_list, fetch_var_name,
                      return_numpy, return_merged):
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
                                                core.CPUPlace(), var.dtype
                                                if var else None)
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
                                               program._places[i], var.dtype
                                               if var else None)
                    if need_check_feed:
                        check_feed_shape_type(var, tensor)
                    res_dict[feed_name] = tensor
                res.append(res_dict)
            exe.feed_tensors_into_local_scopes(res)

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
        specify the scope to store the :code:`Variables` during the executor running if the scope
        is not set, the executor will use the global scope, i.e. :code:`fluid.global_scope()`.

        Args:
            program(Program|CompiledProgram): This parameter represents the :code:`Program` or
                :code:`CompiledProgram` to be executed. If this parameter is not provided, that
                parameter is None, the program will be set to :code:`fluid.default_main_program()`.
                The default is None.
            feed(list|dict): This parameter represents the input variables of the model.
                If it is single card training, the feed is dict type, and if it is multi-card
                training, the parameter feed can be dict or list type variable. If the
                parameter type is dict, the data in the feed will be split and sent to
                multiple devices (CPU/GPU), that is to say, the input data will be evenly
                sent to different devices, so you should make sure the number of samples of
                the current mini-batch must be greater than the number of places;
                if the parameter type is list, those data are copied directly to each device,
                so the length of this list should be equal to the number of places.
                The default is None.
            fetch_list(list): This parameter represents the variables that need to be returned
                after the model runs. The default is None. 
            feed_var_name(str): This parameter represents the name of the input variable of
                the feed operator. The default is "feed".
            fetch_var_name(str): This parameter represents the name of the output variable of
                the fetch operator. The default is "fetch".
            scope(Scope): the scope used to run this program, you can switch 
                it to different scope. default is :code:`fluid.global_scope()`
            return_numpy(bool): This parameter indicates whether convert the fetched variables
                (the variable specified in the fetch list) to numpy.ndarray. if it is False,
                the type of the return value is a list of :code:`LoDTensor`. The default is True.
            use_program_cache(bool): This parameter indicates whether the input :code:`Program` is cached.
                If the parameter is True, the model may run faster in the following cases:
                the input program is :code:`fluid.Program`, and the parameters(program, feed variable name
                and fetch_list variable) of this interface remains unchanged during running.
                The default is False.
            return_merged(bool): This parameter indicates whether fetched variables (the variables
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
               results are spliced together in dimension 0 for the same variable values
               (variables in fetch_list) on different devices.

        Examples 1:
            .. code-block:: python

              import paddle.fluid as fluid
              import numpy

              # First create the Executor.
              place = fluid.CPUPlace() # fluid.CUDAPlace(0)
              exe = fluid.Executor(place)

              data = fluid.data(name='X', shape=[None, 1], dtype='float32')
              hidden = fluid.layers.fc(input=data, size=10)
              loss = fluid.layers.mean(hidden)
              adam = fluid.optimizer.Adam()
              adam.minimize(loss)
              i = fluid.layers.zeros(shape=[1], dtype='int64')
              array = fluid.layers.array_write(x=loss, i=i)

              # Run the startup program once and only once.
              exe.run(fluid.default_startup_program())

              x = numpy.random.random(size=(10, 1)).astype('float32')
              loss_val, array_val = exe.run(feed={'X': x},
                                            fetch_list=[loss.name, array.name])
              print(array_val)
              # [array([0.02153828], dtype=float32)]

        Examples 2:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                # First create the Executor.
                place = fluid.CUDAPlace(0)
                exe = fluid.Executor(place)

                data = fluid.data(name='X', shape=[None, 1], dtype='float32')
                class_dim = 2
                prediction = fluid.layers.fc(input=data, size=class_dim)
                loss = fluid.layers.mean(prediction)
                adam = fluid.optimizer.Adam()
                adam.minimize(loss)

                # Run the startup program once and only once.
                exe.run(fluid.default_startup_program())
                build_strategy = fluid.BuildStrategy()
                binary = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(
                    loss_name=loss.name, build_strategy=build_strategy)
                batch_size = 6
                x = np.random.random(size=(batch_size, 1)).astype('float32')

                # Set return_merged as False to fetch unmerged results:
                unmerged_prediction, = exe.run(binary, feed={'X': x},
                    fetch_list=[prediction.name],
                    return_merged=False)
                # If the user uses two GPU cards to run this python code, the printed result will be
                # (2, 3, class_dim). The first dimension value of the printed result is the number of used
                # GPU cards, and the second dimension value is the quotient of batch_size and the
                # number of used GPU cards.
                print("The unmerged prediction shape: {}".format(np.array(unmerged_prediction).shape))
                print(unmerged_prediction)

                # Set return_merged as True to fetch merged results:
                merged_prediction, = exe.run(binary, feed={'X': x},
                    fetch_list=[prediction.name],
                    return_merged=True)
                # If the user uses two GPU cards to run this python code, the printed result will be
                # (6, class_dim). The first dimension value of the printed result is the batch_size.
                print("The merged prediction shape: {}".format(np.array(merged_prediction).shape))
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
        try:
            return self._run_impl(
                program=program,
                feed=feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name,
                scope=scope,
                return_numpy=return_numpy,
                use_program_cache=use_program_cache,
                use_prune=use_prune,
                return_merged=return_merged)
        except Exception as e:
            if not isinstance(e, core.EOFException):
                warnings.warn(
                    "The following exception is not an EOF exception.")
            six.reraise(*sys.exc_info())

    def _run_impl(self, program, feed, fetch_list, feed_var_name,
                  fetch_var_name, scope, return_numpy, use_program_cache,
                  return_merged, use_prune):
        if self._closed:
            raise RuntimeError("Attempted to use a closed Executor")

        use_default_main_program = program is None
        if program is None:
            program = default_main_program()
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

        if fetch_list is not None:
            if isinstance(fetch_list, Variable) or isinstance(
                    fetch_list, str) or isinstance(fetch_list,
                                                   six.string_types):
                fetch_list = [fetch_list]
            assert isinstance(fetch_list, tuple) or isinstance(fetch_list, list), \
                "Currently , The fetch_list type only should be list or tuple, \n"\
                "but the input type is {}. For more information please refer to \n"\
                "the executor.run(...).".format(type(fetch_list))
        else:
            fetch_list = []

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

        compiled = isinstance(program, compiler.CompiledProgram)

        # For backward compatibility, run directly.
        if not compiled:
            return self._run_program(
                program,
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
            return self._run_parallel(
                program,
                scope=scope,
                feed=feed,
                fetch_list=fetch_list,
                fetch_var_name=fetch_var_name,
                return_numpy=return_numpy,
                return_merged=return_merged)

    def _run_program(self, program, feed, fetch_list, feed_var_name,
                     fetch_var_name, scope, return_numpy, use_program_cache):

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

        if use_program_cache:
            cache_key = _get_strong_program_cache_key(program, feed, fetch_list)
            cached_program = self._get_program_cache(cache_key)
            cached_ctx = self._get_ctx_cache(cache_key)
            cached_scope = self._get_scope_cache(cache_key)
            if cached_program is None:
                cached_program = self._add_feed_fetch_ops(
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
            program = self._add_feed_fetch_ops(
                program=program,
                feed=feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name)

        self._feed_data(program, feed, feed_var_name, scope)
        if not use_program_cache:
            self._default_executor.run(program.desc, scope, 0, True, True,
                                       fetch_var_name)
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
        if scope is None:
            scope = global_scope()
        if fetch_list is None:
            fetch_list = []
        if fetch_info is None:
            fetch_info = []
        assert len(fetch_list) == len(fetch_info)
        compiled = isinstance(program, compiler.CompiledProgram)
        if not compiled:
            # TODO: Need a better way to distinguish and specify different execution mode
            if program._pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program._pipeline_opt)
            else:
                trainer = TrainerFactory()._create_trainer(program._fleet_opt)
                trainer._set_thread_barrier(program._is_distributed)
            trainer._set_program(program)
        else:
            if program._pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program.program._pipeline_opt)
            else:
                trainer = TrainerFactory()._create_trainer(
                    program.program._fleet_opt)
            trainer._set_program(program.program)

        if thread <= 0:
            if dataset.thread_num <= 0:
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
        if dataset is None:
            raise RuntimeError("dataset is need and should be initialized")

        if program._pipeline_opt is not None and program._pipeline_opt[
                "sync_steps"] != -1:
            # hack for paddlebox: sync_steps(-1) denotes paddlebox
            thread = self._adjust_pipeline_resource(program._pipeline_opt,
                                                    dataset, thread)

        dataset._prepare_to_run()

        scope, trainer = self._prepare_trainer(
            program=program,
            dataset=dataset,
            scope=scope,
            thread=thread,
            debug=debug,
            fetch_list=fetch_list,
            fetch_info=fetch_info,
            print_period=print_period)

        trainer._set_infer(is_infer)
        trainer._gen_trainer_desc()

        self._dump_debug_info(program=program, trainer=trainer)
        dataset._dynamic_adjust_before_train(trainer.proto_desc.thread_num)

        trainer_instance = self._default_executor.init_for_dataset(
            program.desc, trainer._desc(), scope, dataset.dataset)

        if fetch_handler is not None:
            scope0 = trainer_instance.get_worker_scope(0)
            fetch_monitor = FetchHandlerMonitor(scope0, fetch_handler)
            fetch_monitor.start()
            self._default_executor.run_from_dataset(trainer_instance)
            fetch_monitor.stop()
            self._default_executor.release_trainer(trainer_instance)
        else:

            self._default_executor.run_from_dataset(trainer_instance)
            self._default_executor.release_trainer(trainer_instance)

        dataset._dynamic_adjust_after_train()
        dataset._finish_to_run()

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
            fetch_list(Variable List): fetch variable list, each variable will be printed during
                training, default is None
            fetch_info(String List): print information for each variable, default is None
            print_period(int): the number of mini-batches for each print, default is 100
            fetch_handler(FetchHandler): a user define class for fetch output.

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle.fluid as fluid

                place = fluid.CPUPlace() # you can set place = fluid.CUDAPlace(0) to use gpu
                exe = fluid.Executor(place)
                x = fluid.layers.data(name="x", shape=[10, 10], dtype="int64")
                y = fluid.layers.data(name="y", shape=[1], dtype="int64", lod_level=1)
                dataset = fluid.DatasetFactory().create_dataset()
                dataset.set_use_var([x, y])
                dataset.set_thread(1)
                filelist = [] # you should set your own filelist, e.g. filelist = ["dataA.txt"]
                dataset.set_filelist(filelist)
                exe.run(fluid.default_startup_program())
                exe.infer_from_dataset(program=fluid.default_main_program(),
                                       dataset=dataset)        

        """
        return self._run_from_dataset(program, dataset, scope, thread, True,
                                      debug, fetch_list, fetch_info,
                                      print_period, fetch_handler)

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
            fetch_list(Variable List): fetch variable list, each variable will be printed
                during training
            fetch_info(String List): print information for each variable, its length should be equal
                to fetch_list
            print_period(int): the number of mini-batches for each print, default is 100
            fetch_handler(FetchHandler): a user define class for fetch output.

        Returns:
            None
        
        Examples:
        
            .. code-block:: python

              import paddle.fluid as fluid

              place = fluid.CPUPlace() # you can set place = fluid.CUDAPlace(0) to use gpu
              exe = fluid.Executor(place)
              x = fluid.layers.data(name="x", shape=[10, 10], dtype="int64")
              y = fluid.layers.data(name="y", shape=[1], dtype="int64", lod_level=1)
              dataset = fluid.DatasetFactory().create_dataset()
              dataset.set_use_var([x, y])
              dataset.set_thread(1)
              filelist = [] # you should set your own filelist, e.g. filelist = ["dataA.txt"]
              dataset.set_filelist(filelist)
              exe.run(fluid.default_startup_program())
              exe.train_from_dataset(program=fluid.default_main_program(),
                                     dataset=dataset)

        """
        return self._run_from_dataset(program, dataset, scope, thread, False,
                                      debug, fetch_list, fetch_info,
                                      print_period, fetch_handler)
