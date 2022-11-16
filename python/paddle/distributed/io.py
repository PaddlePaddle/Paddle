# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
from paddle.framework import dygraph_not_support, core
from paddle.fluid.framework import Program


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
            import paddle

            paddle.enable_static()
            exe = paddle.static.Executor(paddle.CPUPlace())
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


def is_persistable(var):
    """
    Check whether the given variable is persistable.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is persistable
        False if not.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            paddle.enable_static()
            param = fluid.default_main_program().global_block().var('fc.b')
            res = fluid.io.is_persistable(param)
    """
    if (
        var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH
        or var.desc.type() == core.VarDesc.VarType.FETCH_LIST
        or var.desc.type() == core.VarDesc.VarType.READER
    ):
        return False
    return var.persistable


@dygraph_not_support
def save_persistables(executor, dirname, main_program=None, filename=None):
    """
    Save all persistable variables from :code:`main_program` to
    the folder :code:`dirname` or file :code:`filename`. You can refer to
    :ref:`api_guide_model_save_reader_en` for more details. And then
    saves these persistables variables to the folder :code:`dirname` or file
    :code:`filename`.

    The :code:`dirname` is used to specify the folder where persistable variables
    are going to be saved. If you would like to save variables in separate
    files, set :code:`filename` None; if you would like to save all variables in a
    single file, use :code:`filename` to specify the file name.

    Args:
        executor(Executor): The executor to run for saving persistable variables.
                            You can refer to :ref:`api_guide_executor_en` for
                            more details.

        dirname(str, optional): The saving directory path.
                            When you need to save the parameter to the memory, set it to None.
        main_program(Program, optional): The program whose persistbale variables will
                                         be saved. You can refer to
                                         :ref:`api_guide_Program_en` for more details.
                                         If it is None, the default main program will
                                         be used.
                                         Default: None.
        filename(str, optional): The file to save all variables. If you prefer to
                                 save variables in different files, set it to None.
                                 Default: None.

    Returns:
        str: When saving parameters to a file, returns None.
             When saving parameters to memory, returns a binary string containing parameters.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()
            dir_path = "./my_paddle_model"
            file_name = "persistables"
            image = paddle.static..data(name='img', shape=[None, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            feeder = paddle.static.DataFeeder(feed_list=[image, label], place=paddle.CPUPlace())

            predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')
            loss = paddle.nn.functional.cross_entropy(input=predict, label=label)
            avg_loss = paddle.mean(loss)
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            paddle.distributed.io.save_persistables(executor=exe, dirname=dir_path, filename=file_name)
            # The persistables variables weights and bias in the fc layer of the network
            # are going to be saved in the same file named "persistables" in the path
            # "./my_paddle_model"
    """
    if main_program and main_program._is_distributed:
        return _save_distributed_persistables(
            executor, dirname=dirname, main_program=main_program
        )
    else:
        return paddle.static.save_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            vars=None,
            predicate=is_persistable,
            filename=filename,
        )
