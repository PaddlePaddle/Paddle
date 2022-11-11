import os
import errno
import warnings
import logging
import pickle
import contextlib
from functools import reduce
import sys
from io import BytesIO

import numpy as np
import math
import paddle



def _save_distributed_persistables(executor, dirname, main_program):
    """
    save_persistables for distributed training.
    the method will do things listed below:
    1.save part of persistable variables on trainer.
    2.receive "remote prefetch variables" from parameter servers and merge them.
    3.save "distributed lookup table" on parameter servers.
    4.receive "optimizer variables" from parameter servers and merge them.

    Args:
        executor(Executor): The executor to run for saving parameters.
        dirname(str): The saving directory path.
        main_program(Program): The program whose parameters will be
                            saved. the main_program must be the trainer_program
                            get after transpiler.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            paddle.enable_static()
            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(...)
            train_program = t.get_trainer_program()
            _save_distributed_persistables(executor=exe, dirname=param_path, main_program=train_program)
    """

    def __save_remote_params(executor, dirname, remote_params_map):
        """
        receive params on pserver through rpc.
        if the params are be sliced, will concat them to one, then save it.
        """
        if not remote_params_map:
            return

        prog = paddle.static.Program()
        block = prog.global_block()

        # recv optimize vars from pserver
        for name, remote_params in remote_params_map.items():
            origin = remote_params[0].origin
            is_slice = remote_params[0].is_slice

            slices = [None] * len(remote_params)
            slice_varnames = [None] * len(remote_params)
            remote_varnames = [None] * len(remote_params)
            endpoints = [None] * len(remote_params)

            for idx, optimizer in enumerate(remote_params):
                block_id = optimizer.block_id
                slice = optimizer.slice
                endpoint = optimizer.endpoint

                index = block_id if is_slice else idx
                slices[index] = slice
                slice_varnames[index] = "{}.slice.{}".format(slice.name, idx)
                remote_varnames[index] = slice.name
                endpoints[index] = endpoint

            slice_shapes = []
            for slice in slices:
                tmp = [str(dim) for dim in slice.shape]
                slice_shapes.append(",".join(tmp))

            block.append_op(
                type='recv_save',
                attrs={
                    "trainer_id": 0,
                    "shape": origin.shape,
                    "slice_shapes": slice_shapes,
                    "slice_varnames": slice_varnames,
                    "remote_varnames": remote_varnames,
                    "endpoints": endpoints,
                    "file_path": os.path.join(dirname, origin.name),
                },
            )

        executor.run(prog)

    def __save_distributed_lookup_tables(
        executor, dirname, distributed_lookup_table, endpoints
    ):
        """
        because the distributed lookup table may too huge to merge and save at one place,
        it will be saved at parameter server independent respectively.

        the save directory is dirname/"__lookup_table__".

        """
        prog = paddle.static.Program()
        block = prog.global_block()

        # if there is lookup table, the trainer 0 will notify all pserver to save.
        lookup_table_filename = os.path.join(dirname, "__lookup_table__")
        attrs = {}
        attrs['epmap'] = endpoints
        attrs['dir'] = lookup_table_filename
        attrs['lookup_table'] = distributed_lookup_table
        block.append_op(
            type='checkpoint_notify', inputs={}, outputs={}, attrs=attrs
        )
        executor.run(prog)

    def __exclude_vars(exclude_var_names=[]):
        def is_valid(var):
            if var.name in exclude_var_names:
                return False
            if (
                var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH
                or var.desc.type() == core.VarDesc.VarType.FETCH_LIST
                or var.desc.type() == core.VarDesc.VarType.READER
            ):
                return False
            return var.persistable

        return is_valid

    if not isinstance(main_program, Program):
        raise TypeError("'main_program' should be an instance of Program.")

    if not main_program._is_distributed:
        raise ValueError(
            "'_save_distributed_persistables' just be designed for distributed training."
        )

    remote_params_map = (
        main_program._parameters_on_pservers.get_distributed_vars_by_vtypes(
            ["Optimizer", "RemotePrefetch"], groupby=True
        )
    )

    exclude_var_names = []
    if remote_params_map:
        exclude_var_names.extend(remote_params_map.keys())

    if main_program._distributed_lookup_table:
        if isinstance(main_program._distributed_lookup_table, list):
            exclude_var_names.extend(main_program._distributed_lookup_table)
        else:
            exclude_var_names.append(main_program._distributed_lookup_table)

    local_vars = list(
        filter(__exclude_vars(exclude_var_names), main_program.list_vars())
    )
    paddle.static.save_vars(
        executor, main_program=main_program, dirname=dirname, vars=local_vars
    )

    if main_program._is_chief:
        if remote_params_map:
            __save_remote_params(executor, dirname, remote_params_map)
        if main_program._distributed_lookup_table:
            __save_distributed_lookup_tables(
                executor,
                dirname,
                main_program._distributed_lookup_table,
                main_program._endpoints,
            )


