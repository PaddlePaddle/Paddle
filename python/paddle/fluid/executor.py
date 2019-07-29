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
import numpy as np
from .wrapped_decorator import signature_safe_contextmanager
import six
from .framework import Program, default_main_program, Variable
from . import core
from . import compiler
from .. import compat as cpt
from .trainer_factory import TrainerFactory

__all__ = ['Executor', 'global_scope', 'scope_guard']

g_scope = core.Scope()
InferNativeConfig = core.NativeConfig
InferAnalysisConfig = core.AnalysisConfig


def global_scope():
    """
    Get the global/default scope instance. There are a lot of APIs use
    :code:`global_scope` as its default value, e.g., :code:`Executor.run`

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
          numpy.array(fluid.global_scope().find_var("data").get_tensor())

    Returns:
        Scope: The global/default scope instance.
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
    Change the global/default scope instance by Python `with` statement. All
    variable in runtime will assigned to the new scope.

    Args:
        scope: The new global/default scope.

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
    yield
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
    assert isinstance(name, str)
    if scope is None:
        scope = global_scope()
    assert isinstance(scope, core._Scope)

    var = scope.find_var(name)
    assert var is not None, (
        "Cannot find " + name + " in scope. Perhaps you need to make the"
        " variable persistable by using var.persistable = True in your"
        " program.")
    tensor = var.get_tensor()
    if return_numpy:
        tensor = as_numpy(tensor)
    return tensor


def _to_name_str(var):
    if isinstance(var, Variable):
        return var.desc.name()
    elif isinstance(var, str):
        return var
    elif isinstance(var, six.string_types):
        return str(var)
    else:
        raise TypeError(str(var) + " should be Variable or str")


def _get_strong_program_cache_key(program, feed, fetch_list):
    return str(id(program)) + _get_program_cache_key(feed, fetch_list)


def _get_program_cache_key(feed, fetch_list):
    feed_var_names = list(feed.keys())
    fetch_var_names = list(map(_to_name_str, fetch_list))

    return str(feed_var_names + fetch_var_names)


def _as_lodtensor(data, place):
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

        Returns:
            LoDTensor
        """
    if isinstance(data, list):
        raise RuntimeError("Some of your feed data hold LoD information. \
                They can not be completely cast from a list of Python \
                ndarray to LoDTensor. Please convert data to LoDTensor \
                directly before feeding the data.\
                ")
    # single tensor case
    tensor = core.LoDTensor()
    tensor.set(data, place)
    return tensor


class Executor(object):
    """
    An Executor in Python, supports single/multiple-GPU running,
    and single/multiple-CPU running. Python executor takes a program,
    adds feed operators and fetch operators to this program according
    to feed map and fetch_list. Feed map provides input data for the
    program. fetch_list provides the variables(or names) that user wants
    to get after program runs. Note: the executor will run all operators
    in the program but not only the operators dependent by the fetch_list.
    It stores the global variables into the global scope, and creates a
    local scope for the temporary variables. The contents in local scope
    may be discarded after every minibatch forward/backward finished.
    But the global scope variables will be persistent through different runs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.fluid.compiler as compiler
          import numpy
          import os

          use_cuda = True
          place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
          exe = fluid.Executor(place)

          train_program = fluid.Program()
          startup_program = fluid.Program()
          with fluid.program_guard(train_program, startup_program):
              data = fluid.layers.data(name='X', shape=[1], dtype='float32')
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
          # NOTE: If you use CPU to run the program, you need
          # to specify the CPU_NUM, otherwise, fluid will use
          # all the number of the logic core as the CPU_NUM,
          # in that case, the batch size of the input should be
          # greater than CPU_NUM, if not, the process will be
          # failed by an exception.
          if not use_cuda:
              os.environ['CPU_NUM'] = str(2)

          compiled_prog = compiler.CompiledProgram(
              train_program).with_data_parallel(
              loss_name=loss.name)
          loss_data, = exe.run(compiled_prog,
                               feed={"X": x},
                               fetch_list=[loss.name])

    Args:
        place(fluid.CPUPlace|fluid.CUDAPlace(n)): indicate the executor run on which device.

    """

    def __init__(self, place):
        self.place = place
        self.program_caches = dict()
        self.ctx_caches = dict()
        self.scope_caches = dict()
        self.var_caches = dict()
        p = core.Place()
        p.set_place(self.place)
        self._default_executor = core.Executor(p)
        self._closed = False

    def _get_var_cache(self, program_cache_key):
        return self.var_caches.get(program_cache_key, None)

    def _get_scope_cache(self, program_cache_key):
        return self.scope_caches.get(program_cache_key, None)

    def _get_ctx_cache(self, program_cache_key):
        return self.ctx_caches.get(program_cache_key, None)

    def _get_program_cache(self, program_cache_key):
        return self.program_caches.get(program_cache_key, None)

    def _add_program_cache(self, program_cache_key, program):
        self.program_caches[program_cache_key] = program

    def _add_ctx_cache(self, ctx_cache_key, ctx):
        self.ctx_caches[ctx_cache_key] = ctx

    def _add_scope_cache(self, scope_cache_key, scope):
        self.scope_caches[scope_cache_key] = scope

    def _add_var_cache(self, var_cache_key, var):
        self.var_caches[var_cache_key] = var

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
                out = global_block.var(name)
                global_block._prepend_op(
                    type='feed',
                    inputs={'X': [feed_var]},
                    outputs={'Out': [out]},
                    attrs={'col': i})

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
        for op in program.global_block().ops:
            if op.desc.type() == 'feed':
                feed_target_name = op.desc.output('Out')[0]
                cur_feed = feed[feed_target_name]
                if not isinstance(cur_feed, core.LoDTensor):
                    cur_feed = _as_lodtensor(cur_feed, self.place)
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

    '''
    TODO(typhoonzero): Define "no longer use" meaning? Can user create
    a new Executor for the same program and run?
    TODO(panyx0718): Why ParallelExecutor doesn't have close?
    '''

    def close(self):
        """
        Close this executor.

        You can no longer use this executor after calling this method.
        For the distributed training, this method would free the resource
        on PServers related to the current Trainer.

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
                      return_numpy):
        exe = program._executor
        if isinstance(feed, dict):
            feed_tensor_dict = dict()
            for feed_name in feed:
                feed_tensor = feed[feed_name]
                if not isinstance(feed_tensor, core.LoDTensor):
                    feed_tensor = core.LoDTensor()
                    # always set to CPU place, since the tensor need to be splitted
                    # it is fast in CPU
                    feed_tensor.set(feed[feed_name], core.CPUPlace())
                feed_tensor_dict[feed_name] = feed_tensor

            exe.feed_and_split_tensor_into_local_scopes(feed_tensor_dict)
        elif isinstance(feed, list) or isinstance(feed, tuple):
            if len(feed) != len(program._places):
                raise ValueError(
                    "Feed a list of tensor, the list should be the same size as places"
                )

            res = list()
            for i, each in enumerate(feed):
                if not isinstance(each, dict):
                    raise TypeError(
                        "Each element of feed list should be a dict")
                res_dict = dict()
                for feed_name in each:
                    tensor = each[feed_name]
                    if not isinstance(tensor, core.LoDTensor):
                        tmp = core.LoDTensor()
                        tmp.set(tensor, program._places[i])
                        tensor = tmp
                    res_dict[feed_name] = tensor
                res.append(res_dict)
            exe.feed_tensors_into_local_scopes(res)

        fetch_var_names = list(map(_to_name_str, fetch_list))
        exe.run(fetch_var_names, fetch_var_name)
        arr = scope.find_var(fetch_var_name).get_lod_tensor_array()

        if return_numpy:
            return as_numpy(arr)
        return [arr[i] for i in range(len(arr))]

    def _check_fetch_vars_persistable(self, program, fetch_list):
        for var in fetch_list:
            if isinstance(var, Variable):
                persistable = var.persistable
            else:
                block_num = program.desc.num_blocks()
                persistable = None
                var_name = cpt.to_bytes(var)
                for i in six.moves.range(block_num):
                    var_desc = program.desc.block(i).find_var(var_name)
                    if var_desc:
                        persistable = var_desc.persistable()
                        break
                assert persistable is not None, "Variable {} is not found".format(
                    var)

            if not persistable:
                logging.warn("""
     Detect that build_strategy.memory_optimize = True, but the some variables in the fetch
     list is not persistable, you may get wrong fetched value, or an exeception may be thrown
     about cannot find variable of the fetch list. 

     TO FIX this:
         # Sample
         conv1 = fluid.layers.conv2d(data, 4, 5, 1, act=None) 
         # if you need to fetch conv1, then:
         conv1.persistable = True

                 """)

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True,
            use_program_cache=False):
        """
        Run program by this Executor. Feed data by feed map, fetch result by
        fetch_list. Python executor takes a program, add feed operators and
        fetch operators to this program according to feed map and fetch_list.
        Feed map provides input data for the program. fetch_list provides
        the variables(or names) that user want to get after program run.

        Note: the executor will run all operators in the program but not
        only the operators dependent by the fetch_list.

        Examples:
            .. code-block:: python

              import paddle.fluid as fluid
              import numpy

              # First create the Executor.
              place = fluid.CPUPlace() # fluid.CUDAPlace(0)
              exe = fluid.Executor(place)

              data = fluid.layers.data(name='X', shape=[1], dtype='float32')
              hidden = fluid.layers.fc(input=data, size=10)
              loss = fluid.layers.mean(hidden)
              adam = fluid.optimizer.Adam()
              adam.minimize(loss)

              # Run the startup program once and only once.
              exe.run(fluid.default_startup_program())

              x = numpy.random.random(size=(10, 1)).astype('float32')
              outs = exe.run(feed={'X': x},
                             fetch_list=[loss.name])

        Args:
            program(Program|CompiledProgram): the program that need to run,
                if not provided, then default_main_program (not compiled) will be used.
            feed(dict): feed variable map, e.g. {"image": ImageData, "label": LabelData}
            fetch_list(list): a list of variable or variable names that user 
                wants to get, this method will return them according to this list.
            feed_var_name(str): the name for the input variable of 
                feed Operator.
            fetch_var_name(str): the name for the output variable of 
                fetch Operator.
            scope(Scope): the scope used to run this program, you can switch 
                it to different scope. default is global_scope
            return_numpy(bool): if convert the fetched tensor to numpy
            use_program_cache(bool): whether to use the cached program 
                settings across batches. Setting it be true would be faster 
                only when (1) the program is not compiled with data parallel, 
                and (2) program, feed variable names and fetch_list variable 
                names do not changed compared to the last step. 
                
        Returns:

            list(numpy.array): fetch result according to fetch_list.
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
                use_program_cache=use_program_cache)
        except Exception as e:
            if not isinstance(e, core.EOFException):
                print("An exception was thrown!\n {}".format(str(e)))
            raise e

    def _run_impl(self, program, feed, fetch_list, feed_var_name,
                  fetch_var_name, scope, return_numpy, use_program_cache):

        if self._closed:
            raise RuntimeError("Attempted to use a closed Executor")

        if scope is None:
            scope = global_scope()
        if fetch_list is None:
            fetch_list = []

        compiled = isinstance(program, compiler.CompiledProgram)
        # For backward compatibility, run directly.
        if not compiled:
            return self._run_program(
                program,
                self._default_executor,
                feed=feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name,
                scope=scope,
                return_numpy=return_numpy,
                use_program_cache=use_program_cache)
        else:
            if fetch_list and program._is_data_parallel and program._program and    \
                    program._build_strategy._use_legacy_memory_optimize_strategy:
                self._check_fetch_vars_persistable(program._program, fetch_list)

        program._compile(scope, self.place)
        if program._is_data_parallel:
            return self._run_parallel(
                program,
                scope=scope,
                feed=feed,
                fetch_list=fetch_list,
                fetch_var_name=fetch_var_name,
                return_numpy=return_numpy)
        elif program._is_inference:
            return self._run_inference(program._executor, feed)
        else:
            # TODO(panyx0718): Can compile program to optimize executor
            # performance.
            # TODO(panyx0718): executor should be able to run graph.
            assert program._program, "CompiledProgram is compiled from graph, can only run with_data_parallel."
            # use_program_cache is not valid with CompiledProgram
            return self._run_program(
                program._program,
                self._default_executor,
                feed=feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name,
                scope=scope,
                return_numpy=return_numpy,
                use_program_cache=False)

    def _run_program(self, program, exe, feed, fetch_list, feed_var_name,
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
        if program is None:
            program = default_main_program()

        if not isinstance(program, Program):
            raise TypeError(
                "Executor requires Program as its Parameter. But you passed in %s"
                % (type(program)))

        if use_program_cache:
            cache_key = _get_strong_program_cache_key(program, feed, fetch_list)
            cached_program = self._get_program_cache(cache_key)
            cached_ctx = self._get_ctx_cache(cache_key)
            cached_scope = self._get_scope_cache(cache_key)
            cached_var = self._get_var_cache(cache_key)
            if cached_program is None:
                cached_program = self._add_feed_fetch_ops(
                    program=program,
                    feed=feed,
                    fetch_list=fetch_list,
                    feed_var_name=feed_var_name,
                    fetch_var_name=fetch_var_name)
                self._add_program_cache(cache_key, cached_program)
                fetch_list_str = list(map(_to_name_str, fetch_list))
                cached_ctx = self._default_executor.prepare_ctx_cache(
                    cached_program.desc, 0, fetch_list_str, False)
                cached_var = self._default_executor.create_variables(
                    cached_program.desc, scope, 0)
                # currently, we cache program, vars, sub_scope here
                # we suppose that in a life cycle of training, a user
                # will not create many programs. So, here the basic
                # rule of caching is to cache all unseen (program, var, scope)
                # when a user use use_program_cache.
                cached_scope = scope.new_scope()
                self._add_ctx_cache(cache_key, cached_ctx)
                self._add_var_cache(cache_key, cached_var)
                self._add_scope_cache(cache_key, cached_scope)
            program = cached_program
            ctx = cached_ctx
            scope = cached_scope
            var = cached_var
        else:
            program = self._add_feed_fetch_ops(
                program=program,
                feed=feed,
                fetch_list=fetch_list,
                feed_var_name=feed_var_name,
                fetch_var_name=fetch_var_name)

        self._feed_data(program, feed, feed_var_name, scope)
        if not use_program_cache:
            exe.run(program.desc, scope, 0, True, True, fetch_var_name)
        else:
            exe.run_cached_prepared_ctx(ctx, scope, False, False, False)
        outs = self._fetch_data(fetch_list, fetch_var_name, scope)
        if return_numpy:
            outs = as_numpy(outs)
        return outs

    def _run_inference(self, exe, feed):
        return exe.run(feed)

    def _dump_debug_info(self, program=None, trainer=None):
        with open(str(id(program)) + "_train_desc.prototxt", "w") as fout:
            fout.write(str(trainer))
        if program._fleet_opt:
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
            trainer._set_program(program)
        else:
            if program._pipeline_opt:
                trainer = TrainerFactory()._create_trainer(
                    program.program._pipeline_opt)
            else:
                trainer = TrainerFactory()._create_trainer(
                    program.program._fleet_opt)
            trainer._set_program(program.program)

        # The following thread_num-determined logic will be deprecated
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

    def infer_from_dataset(self,
                           program=None,
                           dataset=None,
                           scope=None,
                           thread=0,
                           debug=False,
                           fetch_list=None,
                           fetch_info=None,
                           print_period=100):
        """
        The document of infer_from_dataset is almost the same as
        train_from_dataset, except that in distributed training,
        push gradients will be disabled in infer_from_dataset.
        infer_from_dataset() can be used for evaluation in multi-thread
        very easily.

        Args:
            program(Program|CompiledProgram): the program that needs to be run,
               if not provided, then default_main_program (not compiled) will be used.
            dataset(paddle.fluid.Dataset): dataset created outside this function,
               a user should provide a well-defined dataset before calling this function.
               Please check the document of Dataset if needed. default is None
            scope(Scope): the scope used to run this program, you can switch it to different scope
               for each run. default is global_scope
            thread(int): number of thread a user wants to run in this function. The actual number
               of thread will be min(Dataset.thread_num, thread) if thread > 0, default is 0
            debug(bool): whether a user wants to run infer_from_dataset, default is False
            fetch_list(Variable List): fetch variable list, each variable
                                       will be printed during training, default is None
            fetch_info(String List): print information for each variable, default is None
            print_period(int): the number of mini-batches for each print, default is 100

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
        if dataset == None:
            raise RuntimeError("dataset is needed and should be initialized")

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
        trainer._set_infer(True)
        trainer._gen_trainer_desc()
        self._dump_debug_info(program=program, trainer=trainer)
        self._default_executor.run_from_dataset(program.desc, scope,
                                                dataset.dataset,
                                                trainer._desc())
        dataset._finish_to_run()
        return None

    def train_from_dataset(self,
                           program=None,
                           dataset=None,
                           scope=None,
                           thread=0,
                           debug=False,
                           fetch_list=None,
                           fetch_info=None,
                           print_period=100):
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
            thread(int): number of thread a user wants to run in this function. The actual number
               of thread will be min(Dataset.thread_num, thread)
            debug(bool): whether a user wants to run train_from_dataset 
            fetch_list(Variable List): fetch variable list, each variable
                                       will be printed during training
            fetch_info(String List): print information for each variable
            print_period(int): the number of mini-batches for each print

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
        if dataset == None:
            raise RuntimeError("dataset is need and should be initialized")

        if program._pipeline_opt:
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
        trainer._gen_trainer_desc()
        self._dump_debug_info(program=program, trainer=trainer)
        self._default_executor.run_from_dataset(program.desc, scope,
                                                dataset.dataset,
                                                trainer._desc())
        dataset._finish_to_run()
        return None
