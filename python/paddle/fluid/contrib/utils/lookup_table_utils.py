# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""lookup_table_utils.py will move to fluid/incubate/fleet/utils/lookup_table.py"""

from __future__ import print_function

import os
import time
import logging

import paddle
from paddle.fluid import core
from paddle.fluid import io
from paddle.fluid import Program
from paddle.fluid.log_helper import get_logger

__all__ = [
    "load_persistables_for_increment", "load_persistables_for_inference",
    "convert_dist_to_sparse_program"
]

_logger = get_logger(
    'lookup_table_utils',
    logging.INFO,
    fmt='%(asctime)s-%(levelname)s: %(message)s')

model_filename = "__model__"
lookup_table_dir = "__lookup_table__"


def __insert_lookup_sparse_table_op(main_program, idx, ids, w, out):
    main_program.global_block()._insert_op(
        index=idx,
        type="lookup_sparse_table",
        inputs={"Ids": [ids],
                "W": [w]},
        outputs={"Out": [out]},
        attrs={
            "is_distributed": False,
            "is_sparse": True,
            "grad_inplace": False
        })


def __get_prefetch_op_tuples(main_program):
    # current lookup tables op is split_ids->prefetch->merge_ids
    prefetch_op_tuples = None
    op_types = [op.type for op in main_program.global_block().ops]

    for i in range(len(op_types)):
        if op_types[i] == "prefetch":
            if op_types[i - 1] == "split_ids" and op_types[i +
                                                           1] == "merge_ids":
                split_ids_op_id = i - 1
                split_ids_inputs = main_program.global_block().ops[i - 1].input(
                    "Ids")
                prefetch_op_inputs = main_program.global_block().ops[i].input(
                    "X")
                prefetch_op_outputs = main_program.global_block().ops[i].output(
                    "Out")
                merge_ids_outputs = main_program.global_block().ops[
                    i + 1].output("Out")

                need_delete_vars = []
                need_delete_vars.extend(prefetch_op_inputs)
                need_delete_vars.extend(prefetch_op_outputs)

                prefetch_op_tuples = (split_ids_op_id, split_ids_inputs,
                                      merge_ids_outputs, need_delete_vars)
                break
    return prefetch_op_tuples


def convert_dist_to_sparse_program(program):
    """
    WARNING: this function will only be used for distributed training with distributed lookup table.
    when we train model with distributed lookup table but want to do the local inference, we can use
    this function to convert the train program with distributed lookup table to sparse lookup table.

    Args:
        program(Program): the program must be the trainer program, which will be get by the distribute transpiler.
    Returns:
        program: The `program` is a Program, it's the program replace distributed lookup table to sparse lookup table.
    """
    if not program._distributed_lookup_table:
        _logger.warn(
            "There are no distributed lookup tables need to be converted")
        return

    # create table param and grad var in pserver program
    origin_emb_var = "{}.origin".format(program._distributed_lookup_table)
    emb_var = program._distributed_lookup_table
    program.global_block()._rename_var(emb_var, origin_emb_var)
    origin_param_var = program.global_block().vars[origin_emb_var]

    param_var = program.global_block().create_var(
        name=emb_var,
        shape=origin_param_var.shape,
        dtype=origin_param_var.dtype,
        type=core.VarDesc.VarType.SELECTED_ROWS,
        persistable=True)
    # parameter must be selected rows
    param_var.desc.set_type(core.VarDesc.VarType.SELECTED_ROWS)
    program._sync_with_cpp()

    prefetch_op_tuples = __get_prefetch_op_tuples(program)

    split_ids_id = prefetch_op_tuples[0]

    for idx in range(split_ids_id + 2, split_ids_id - 1, -1):
        program.global_block()._remove_op(idx)
    program.desc.flush()

    in_out_pairs = zip(prefetch_op_tuples[1], prefetch_op_tuples[2])

    for in_out_pair in in_out_pairs:
        idx = split_ids_id
        ids = program.global_block().vars[in_out_pair[0]]
        out = program.global_block().vars[in_out_pair[1]]
        __insert_lookup_sparse_table_op(program, idx, ids, param_var, out)
        program.desc.flush()
    return program


def load_persistables_for_increment(dirname, executor, program,
                                    lookup_table_var, lookup_table_var_path):
    """
    WARNING: this function will only be used for distributed training with distributed lookup table.
    for increment training, the pserver will not only load dense variables,
    but also load the suitable lookup table var. Because of sliced lookup table
    var with HASH, we must load the correct sliced var.

    Args:
        dirname(str): The directory path
        executor(Executor): The executor to run for loading inference model.
        program(Program): The parameter server program, which will run on Pserver.
        lookup_table_var: the distributed lookup tables var name.
        lookup_table_var_path: the the distributed lookup tables var location.

    Returns:
        None
    """

    def _load_persistable_vars(executor, dirname, need_load_vars):
        load_prog = Program()
        load_block = load_prog.global_block()
        need_delete_vars = []

        for param in need_load_vars:
            origin_var = param.origin
            slice_var = param.slice
            is_slice = param.is_slice
            offset = param.offset

            if is_slice:
                origin = load_block.create_var(
                    name="{}.load".format(origin_var.name),
                    type=origin_var.type,
                    shape=origin_var.shape,
                    dtype=origin_var.dtype,
                    persistable=True)

                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [origin]},
                    attrs={
                        'file_path': os.path.join(dirname, origin_var.name)
                    })

                slice = load_block.create_var(
                    name=slice_var.name,
                    type=slice_var.type,
                    shape=slice_var.shape,
                    dtype=slice_var.dtype,
                    persistable=True)

                dim1_flatten = reduce(lambda x, y: x * y, slice.shape[1:])
                start = int(offset / dim1_flatten)
                end = int(offset / dim1_flatten + slice.shape[0])

                load_block.append_op(
                    type="slice",
                    inputs={'Input': origin},
                    outputs={'Out': slice},
                    attrs={'axes': [0],
                           'starts': [start],
                           'ends': [end]})

                need_delete_vars.append(origin)
            else:
                origin = load_block.create_var(
                    name="{}".format(origin_var.name),
                    type=origin_var.type,
                    shape=origin_var.shape,
                    dtype=origin_var.dtype,
                    persistable=True)
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [origin]},
                    attrs={
                        'file_path': os.path.join(dirname, origin_var.name)
                    })

        load_block.append_op(
            type='delete_var',
            inputs={'X': need_delete_vars}, )

        executor.run(load_prog)

    def __load_lookup_table_vars(executor, main_program, lookup_table_var,
                                 lookup_table_var_path):
        emb_var = main_program.global_block().var(lookup_table_var)

        load_program = Program()
        load_block = load_program.global_block()
        load_block.append_op(
            type='load',
            inputs={},
            outputs={'Out': [emb_var]},
            attrs={'file_path': lookup_table_var_path})
        executor.run(load_program)

    if not os.path.isdir(dirname):
        raise ValueError("There is no directory named '%s'", dirname)

    if not os.path.exists(lookup_table_var_path):
        raise ValueError("There is no file named '%s'", lookup_table_var_path)

    if not isinstance(program, Program):
        raise ValueError("program must be an instance of fluid.Program")

    _logger.info("Start Load Sparse Program With "
                 "Distributed Lookup Table Vars from {}, time = {}".format(
                     dirname, time.ctime()))

    need_load_vars = program._parameters_on_pservers.get_distributed_vars_by_ep(
        program._ps_endpoint)
    _load_persistable_vars(executor, dirname, need_load_vars)
    __load_lookup_table_vars(executor, program, lookup_table_var,
                             lookup_table_var_path)

    _logger.info("Finish Load Sparse Program With "
                 "Distributed Lookup Table Vars from {}, time = {}".format(
                     dirname, time.ctime()))


def load_persistables_for_inference(dirname, executor, program,
                                    lookup_table_var_name):
    """
    WARNING: this function will only be used for inference with distributed lookup table.
    Inference with distributed lookup table is a little funky, this function will load distributed
    lookup table vars into sparse var, can be used in local inference mode.

    Args:
        dirname(str): The directory path
        executor(Executor): The executor to run for loading inference model.
        program(Program): The parameter server program, which will run on Pserver.
        lookup_table_var_name: the distributed lookup tables var name.
    Returns:
        None
    """

    def _load_persistable_vars(executor, dirname, program, lookup_table_vars):
        def _is_checkpoint_var(exclude_fluid_vars=None):
            """
            the checkpoint will not save or load all the variables.
            var type is FEED_MINIBATCH/FETCH_LIST/RAW or var name ends with @GRAD are discarded.

            : param var(Variable)
            """

            if exclude_fluid_vars is None:
                exclude_fluid_vars = []

            def is_valid(var):
                if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                        var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                        var.desc.type() == core.VarDesc.VarType.RAW:
                    return False
                # @GRAD are named for gradient variables, checkpoint will not save it.
                if "@GRAD" in var.name:
                    return False
                # .trainer_ are named for distribute train variables, checkpoint will not save it.
                if ".trainer_" in var.name:
                    return False

                # .block is named for distribute train variables, checkpoint will not save it.
                if ".block" in var.name:
                    return False

                if "tmp_" in var.name:
                    return False

                if var.name in exclude_fluid_vars:
                    return False

                return var.persistable

            return is_valid

        io.load_vars(
            executor,
            dirname=dirname,
            main_program=program,
            predicate=_is_checkpoint_var(lookup_table_vars),
            filename=None)

    def _load_lookup_table_vars(executor, dirname, main_program,
                                lookup_table_vars):
        if not os.path.isdir(dirname):
            raise ValueError("There is no directory named '%s'", dirname)

        lookup_table_dirname = os.path.join(dirname, lookup_table_dir)

        emb_var_name = lookup_table_vars[0]
        emb_var = main_program.global_block().var(emb_var_name)

        emb_files = []
        for emb_name in os.listdir(lookup_table_dirname):
            if emb_var_name in emb_name:
                emb_files.append(emb_name)

        convert_program = Program()
        global_block = convert_program.global_block()

        emb_var = global_block.create_var(
            name=emb_var.name,
            shape=emb_var.shape,
            dtype=emb_var.dtype,
            type=core.VarDesc.VarType.SELECTED_ROWS,
            persistable=True)
        emb_var.desc.set_type(core.VarDesc.VarType.SELECTED_ROWS)

        sums = []

        for i, emb_file in enumerate(emb_files):
            var_name = "{}_{}".format(emb_var.name, i)
            param_var = global_block.create_var(
                name=var_name,
                shape=emb_var.shape,
                dtype=emb_var.dtype,
                type=core.VarDesc.VarType.SELECTED_ROWS,
                persistable=True)
            param_var.desc.set_type(core.VarDesc.VarType.SELECTED_ROWS)
            global_block.append_op(
                type='load',
                inputs={},
                outputs={'Out': [param_var]},
                attrs={
                    'file_path': os.path.join(lookup_table_dirname, var_name)
                })
            sums.append(param_var)
        global_block.append_op(
            type='merge_sparse_lookup_table',
            inputs={"X": sums},
            outputs={'Out': emb_var},
            attrs={})
        global_block.append_op(
            type='save',
            inputs={"X": [emb_var]},
            outputs={},
            attrs={
                'file_path': os.path.join(lookup_table_dirname, emb_var.name)
            })
        global_block.append_op(type='delete_var', inputs={'X': sums})
        executor.run(convert_program)

    if not os.path.isdir(dirname):
        raise ValueError("There is no directory named '%s'", dirname)

    if program:
        if not isinstance(program, Program):
            raise ValueError("program must be an instance of fluid.Program")
    else:
        local_model = os.path.join(dirname, model_filename)

        with open(local_model, "rb") as f:
            program_desc_str = f.read()

        program = Program.parse_from_string(program_desc_str)

        if not core._is_program_version_supported(program._version()):
            raise ValueError("Unsupported program version: %d\n" %
                             program._version())

    _logger.info("Start Load Sparse Program With "
                 "Distributed Lookup Table Vars from {}, time = {}".format(
                     dirname, time.ctime()))

    _load_persistable_vars(executor, dirname, program, [lookup_table_var_name])
    _load_lookup_table_vars(executor, dirname, program, [lookup_table_var_name])

    _logger.info("Finish Load Sparse Program With "
                 "Distributed Lookup Table Vars from {}, time = {}".format(
                     dirname, time.ctime()))

    return program


def get_inference_model(main_program, feeded_var_names, target_vars):
    """
    Prune the given `main_program` to build a new program especially for inference with distributed lookup table ,
    and then add `feeded_vars` and `target_vars` in this program.

    Args:
        main_program(Program|None): The original program, which will be pruned to
                                    build the inference model. If is set None,
                                    the default main program will be used.
                                    Default: None.
        feeded_var_names(list[str]): Names of variables that need to be fed data
                                     during inference.
        target_vars(list[Variable]): Variables from which we can get inference
                                     results.
    Returns:
        program(Program)

    Raises:
        ValueError: If `feed_var_names` is not a list of basestring.
        ValueError: If `target_vars` is not a list of Variable.

    """

    def prepend_feed_ops(inference_program,
                         feed_target_names,
                         feed_holder_name='feed'):
        if len(feed_target_names) == 0:
            return

        global_block = inference_program.global_block()

        feed_var = global_block.create_var(
            name=feed_holder_name,
            type=core.VarDesc.VarType.FEED_MINIBATCH,
            persistable=True)

        for i, name in enumerate(feed_target_names):
            out = global_block.var(name)
            global_block._prepend_op(
                type='feed',
                inputs={'X': [feed_var]},
                outputs={'Out': [out]},
                attrs={'col': i})

    def append_fetch_ops(inference_program,
                         fetch_target_names,
                         fetch_holder_name='fetch'):
        global_block = inference_program.global_block()
        fetch_var = global_block.create_var(
            name=fetch_holder_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True)

        for i, name in enumerate(fetch_target_names):
            global_block.append_op(
                type='fetch',
                inputs={'X': [name]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})

    origin_program = main_program.clone()
    main_program = main_program.clone()
    global_block = main_program.global_block()

    need_to_remove_op_index = []
    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            need_to_remove_op_index.append(i)

    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)

    main_program.desc.flush()

    main_program = main_program._prune(targets=target_vars)
    main_program = main_program._inference_optimize(prune_read_op=True)

    fetch_var_names = [v.name for v in target_vars]

    prepend_feed_ops(main_program, feeded_var_names)
    append_fetch_ops(main_program, fetch_var_names)

    return main_program
