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

from __future__ import print_function

import os
import time
import logging

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import io
from paddle.fluid import Program

__all__ = [
    "load_inference_model", "load_persistable_vars",
    "convert_dist_to_sparse_program"
]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger("lookup_table_utils")
_logger.setLevel(logging.INFO)

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


def convert_dist_to_sparse_program(main_program):
    if not main_program._distributed_lookup_table:
        _logger.warn(
            "There are no distributed lookup tables need to be converted")
        return

    # create table param and grad var in pserver program
    origin_emb_var = "{}.origin".format(main_program._distributed_lookup_table)
    emb_var = main_program._distributed_lookup_table
    main_program.global_block()._rename_var(emb_var, origin_emb_var)
    origin_param_var = main_program.global_block().vars[origin_emb_var]

    param_var = main_program.global_block().create_var(
        name=emb_var,
        shape=origin_param_var.shape,
        dtype=origin_param_var.dtype,
        type=core.VarDesc.VarType.SELECTED_ROWS,
        persistable=True)
    # parameter must be selected rows
    param_var.desc.set_type(core.VarDesc.VarType.SELECTED_ROWS)
    main_program._sync_with_cpp()

    prefetch_op_tuples = __get_prefetch_op_tuples(main_program)

    split_ids_id = prefetch_op_tuples[0]

    for idx in range(split_ids_id + 2, split_ids_id - 1, -1):
        main_program.global_block()._remove_op(idx)
    main_program.desc.flush()

    in_out_pairs = zip(prefetch_op_tuples[1], prefetch_op_tuples[2])

    for in_out_pair in in_out_pairs:
        idx = split_ids_id
        ids = main_program.global_block().vars[in_out_pair[0]]
        out = main_program.global_block().vars[in_out_pair[1]]
        __insert_lookup_sparse_table_op(main_program, idx, ids, param_var, out)
        main_program.desc.flush()
    return main_program


def load_persistable_vars(executor, dirname, program, lookup_table_var):
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
            type='sum', inputs={"X": sums}, outputs={'Out': emb_var}, attrs={})
        global_block.append_op(type='delete_var', inputs={'X': sums})
        executor.run(convert_program)

    _logger.info("Start Load Sparse Program With "
                 "Distributed Lookup Table Vars from {}, time = {}".format(
                     dirname, time.ctime()))

    lookup_table_vars = [lookup_table_var]

    io.load_vars(
        executor,
        dirname=dirname,
        main_program=program,
        predicate=_is_checkpoint_var(lookup_table_vars),
        filename=None)

    _load_lookup_table_vars(executor, dirname, program, lookup_table_vars)

    _logger.info("Finish Load Sparse Program With "
                 "Distributed Lookup Table Vars from {}, time = {}".format(
                     dirname, time.ctime()))


def load_inference_model(dirname, executor, lookup_table_var_name):
    if not os.path.isdir(dirname):
        raise ValueError("There is no directory named '%s'", dirname)

    local_model = os.path.join(dirname, model_filename)

    with open(local_model, "rb") as f:
        program_desc_str = f.read()

    program = Program.parse_from_string(program_desc_str)

    if not core._is_program_version_supported(program._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program._version())

    # Binary data also need version.
    load_persistable_vars(executor, dirname, program, lookup_table_var_name)

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]
