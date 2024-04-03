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

import logging
import warnings

import numpy as np

import paddle
from paddle.base import core
from paddle.base.data_feeder import convert_dtype
from paddle.base.executor import (
    _as_lodtensor,
    _StandaloneExecutor,
    check_feed_shape_type,
)
from paddle.base.framework import Operator, Program
from paddle.distributed.auto_parallel.static.utils import get_logger, is_comm_op


def check_if_op_supports_runtime_profiling(op):
    return not is_comm_op(op)


def _measure_program_real_op_cost_multipass(program, place, run_iters, verbose):
    '''
    Run op profiling for a single pass. Internal function, do not call this directly.
    '''

    # clone the program to avoid accidental change made to the vanilla program.
    cloned_program = program.clone()
    cloned_main_block = cloned_program.global_block()

    # We will run the executor in a newly created scope, so that our
    # executor will not pollute the global scope when running. Since
    # we created a brand new scope, we need to manually create input
    # tensors and network parameters and feed fake data into them.
    scope = core.Scope()

    logger = get_logger(log_level=logging.INFO)

    def _analyze_graph_and_collect_all_vars_with_zero_in_degree():
        var_in_degree = {}

        def _collect_op_input_var_names(op: Operator):
            input_var_names = []
            for input_name in op.input_names:
                input_var_names += op.input(input_name)
            return input_var_names

        def _collect_op_output_var_names(op: Operator):
            output_var_names = []
            for output_name in op.output_names:
                output_var_names += op.output(output_name)
            return output_var_names

        def _record_op_output_vars_in_degree(in_var_names, out_var_names):
            for out_var_name in out_var_names:
                if out_var_name in in_var_names:
                    # NOTE (liuchenghao): if an op's input var is its output var,
                    # this means this var forms an in-place connection to itself,
                    # in this situation we need to ignore this variable, this way
                    # we can ensure that vars with zero in-degree are dangling vars
                    # and they should be created manually before program executes.
                    continue
                var_in_degree[out_var_name] += 1

        def _filter_vars_with_zero_in_degree_and_ignore_feed_fetch_vars():
            filtered_vars = []
            for var_name in var_in_degree:
                if var_name in ['feed', 'fetch']:
                    continue
                if var_in_degree[var_name] == 0:
                    filtered_vars.append(var_name)
            return filtered_vars

        for op in cloned_main_block.ops:
            op: Operator
            if is_comm_op(op):
                # ignore communication op from graph, because sometimes we want to profile a sub-graph
                # and these dangling operators will not work (no graph to communicate to/from)
                continue
            input_var_names, output_var_names = _collect_op_input_var_names(
                op
            ), _collect_op_output_var_names(op)
            for var_name in input_var_names + output_var_names:
                if var_name not in var_in_degree:
                    var_in_degree[var_name] = 0
            _record_op_output_vars_in_degree(input_var_names, output_var_names)
        return _filter_vars_with_zero_in_degree_and_ignore_feed_fetch_vars()

    def _alloc_and_fill_var(var_name):
        supported_var_dtypes = [
            "paddle.float16",
            "paddle.float32",
            "paddle.float64",
            "paddle.int8",
            "paddle.int16",
            "paddle.int32",
            "paddle.int64",
            "paddle.bool",
        ]
        var = cloned_main_block.var(var_name)
        var_shape = var.shape
        var_dtype = var.dtype
        assert str(var_dtype) in supported_var_dtypes, (
            "Found unsupported variable dtype: \"{}\", current supported "
            "dtype(s) is/are: [{}]. ".format(
                str(var_dtype), ", ".join(supported_var_dtypes)
            )
        )
        logger.info(
            f'[+] var: "{var_name}", shape={str(var_shape)}, dtype="{str(var_dtype)}".\n'
        ) if verbose else None
        np_dtype = (
            convert_dtype(var_dtype)
            if isinstance(var_dtype, core.VarDesc.VarType)
            else var_dtype
        )
        if str(var_dtype).find('int') != -1:
            # target variable's type is int* (uint*, int*), it is highly possible that
            # the target variable contains indices (such as lookup_table op's input var)
            # for safety we need to fill it with all one instead of random numbers
            # NOTE (liuchenghao): filling with zero will generate "division by zero" error
            # in mod ops, so filling with one seems to be the simplest way to make it work,
            # although it is possible that for array with only one element, index "1" is
            # invalid, that situation is very rare and we don't need to care about it now.
            new_tensor = np.array(np.ones(var_shape)).astype(np_dtype)
        else:
            # target variable's type is float*, we treat it as an ordinary tensor, fill it
            # with random gaussian numbers
            new_tensor = np.array(np.random.randn(*var_shape)).astype(np_dtype)
        new_tensor = _as_lodtensor(new_tensor, place, var_dtype)
        check_feed_shape_type(var, new_tensor)
        core.set_variable(scope, new_tensor, var_name)
        return new_tensor

    def _configure_feed_ops_and_return_feed_names():
        """
        configure feed op,
        1. alloc feed op output var storage
        2. fill feed op's input var
        return feed var names
        """

        feed_names = []
        has_feed_op = False
        for op in cloned_main_block.ops:
            if op.type == "feed":
                has_feed_op = True
                out_var_name = op.desc.output('Out')[0]
                in_var_name = op.desc.input('X')[0]  # this is usually "feed"
                input_index = op.desc.attr('col')
                new_tensor = _alloc_and_fill_var(out_var_name)
                core.set_feed_variable(
                    scope, new_tensor, in_var_name, input_index
                )
                feed_names.append(out_var_name)
        if not has_feed_op:
            logger.info(
                "WARNING: program does not have any feed op.\n"
            ) if verbose else None
        return feed_names

    for var_name in _analyze_graph_and_collect_all_vars_with_zero_in_degree():
        _alloc_and_fill_var(var_name)
    feed_names = _configure_feed_ops_and_return_feed_names()

    # build a simple plan from program and run profiling
    plan = core.Plan([core.Job("default")], {"default": cloned_program.desc})
    exe = _StandaloneExecutor(place, plan, scope)

    num_ops = len(cloned_main_block.ops)
    prof_results = [[None for _ in range(run_iters)] for _ in range(num_ops)]

    for iter_id in range(run_iters):
        # for each iteration, run profiling and retrieve modified version of program desc
        program_desc = exe.run_profile(feed_names)

        # rebuild program object from the new program desc
        temp_program = cloned_program.clone()
        temp_program._rebuild_from_desc(program_desc)
        temp_main_block = temp_program.global_block()

        # collect profiling result
        for op_id, temp_op in zip(
            range(len(temp_main_block.ops)), temp_main_block.ops
        ):
            run_time_us = temp_op.dist_attr.run_time_us
            prof_results[op_id][iter_id] = (
                run_time_us
                if check_if_op_supports_runtime_profiling(temp_op)
                and run_time_us >= 0.0
                else None
            )
    return prof_results


def measure_program_real_op_cost(
    program: paddle.static.Program,
    run_iters: int = 8,
    place=paddle.base.framework._current_expected_place(),
    verbose_level: int = 0,
):
    '''
    Description
    -----------
    Measuring real op run time (us) with respect to the given "program" and "place".

    Parameters
    -----------
    @param program: paddle.static.Program
        The program object waiting to be executed.
    @param run_iters: int
        Specify how many iterations will be run during profiling. Larger value tends
        to give more accurate profiling result but requires more time.
    @param place: paddle.CPUPlace | paddle.CUDAPlace
        Where the program is going to be executed.
    @param verbose_level: int
        Set up verbose level during profiling. Can be set to one of the following:
        0 = turn off, don't output anything,
        1 = output profiling messages only,
        2 = output profiling and debug messages.

    Returns
    -----------
    Nothing to return. This API will write op run time directly into program object.
    For example, to retrieve the run time for the first op in program, use:
    >>> program.global_block().ops[0].dist_attr.run_time_us

    Note
    -----------
    Not all ops support runtime profiling. Currently communication ops do not support
    runtime profiling feature since their execution times rely on other ops. To check
    if an op supports runtime profiling, use:
    >>> check_if_op_supports_runtime_profiling(op)
    where "op" is an instance of "paddle.base.framework.Operator".

    Example
    -----------
    * Profiling a simple program from scratch:
    >>> from paddle.distributed.auto_parallel.static.utils import measure_program_real_op_cost
    >>> program = ... # build your own program object here.
    >>> measure_program_real_op_cost(
    >>>     program, verbose_level=1
    >>> )
    * Profiling a program which is already embedded into an Executor or some other class instance:
    >>> import paddle
    >>> from paddle.distributed.auto_parallel.static.utils import measure_program_real_op_cost
    >>> place: str = paddle.device.get_device() # here we assume place = "cuda:x"
    >>> place = paddle.CUDAPlace(int(place.split(':')[1]))
    >>> # here "program" is an inner object that has already been built before
    >>> measure_program_real_op_cost(program, verbose_level=1)
    '''

    assert isinstance(program, Program), (
        '"program" should be a instance of "paddle.base.framework.Program" but got type "%s".'
        % type(program).__name__
    )
    supported_places = [
        paddle.CUDAPlace,
    ]
    assert any(
        isinstance(place, supported_place)
        for supported_place in supported_places
    ), f'Current place ({str(place)}) does not support runtime profiling. "place" should be one of the following: {str(supported_places)}.'
    assert isinstance(run_iters, int) and run_iters >= 1, (
        'Invalid parameter run_iters set. run_iters '
        'should be an integer >= 1.'
    )
    if run_iters == 1:
        warnings.warn(
            'run_iters was set to 1, profiling results might be inaccurate due to outliers.'
        )

    logger = get_logger(log_level=logging.INFO)

    # run profiling multiple times and record op run time of each run
    prof_results = _measure_program_real_op_cost_multipass(
        program, place, run_iters, verbose=(verbose_level >= 2)
    )
    op_num = len(prof_results)
    for op_id, op in zip(range(op_num), program.global_block().ops):
        op_runtime_us_final = None
        if prof_results[op_id][0] is not None:
            op_runtime_us_final = np.median(prof_results[op_id])
        if (
            op_runtime_us_final is not None
            and check_if_op_supports_runtime_profiling(op)
        ):
            op.dist_attr.run_time_us = op_runtime_us_final
        logger.info(
            "%4s %32s  %.1f us"
            % (str(op_id), str(op.type), op_runtime_us_final)
        ) if verbose_level >= 1 else None
