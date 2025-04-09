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
from paddle.base.framework import Program, static_only
from paddle.framework import core, dygraph_not_support


def _load_distributed_persistables(executor, dirname, main_program=None):
    """
    customized load_persistables for distributed training.
    it should be used on parameter server,

    Args:
        executor(Executor): The executor to run for saving parameters.
        dirname(str): The load directory path.
        main_program(Program): The program whose parameters will be
                            loaded. the main_program must be the pserver_program
                            get after transpiler.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle.base as base

            >>> paddle.enable_static()
            >>> exe = base.Executor(base.CPUPlace())
            >>> param_path = "./my_paddle_model"
            >>> t = paddle.distributed.transpiler.DistributeTranspiler()
            >>> t.transpile(...)
            >>> pserver_prog = t.get_pserver_program(...)
            >>> _load_distributed_persistables(executor=exe, dirname=param_path, main_program=pserver_prog)
    """

    def __is_distributed_part_var(varname):
        trainer_idx = varname.find(".trainer_")
        block_idx = varname.find(".block")
        return trainer_idx or block_idx

    def __load_persistable_vars(executor, dirname, need_load_vars):
        load_prog = Program()
        load_block = load_prog.global_block()
        need_delete_vars = []

        for param in need_load_vars:
            origin_var = param.origin
            slice_var = param.slice
            is_slice = param.is_slice
            offset = param.offset

            if is_slice:
                slice = load_block.create_var(
                    name=slice_var.name,
                    type=slice_var.type,
                    shape=slice_var.shape,
                    dtype=slice_var.dtype,
                    persistable=True,
                )

                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [slice]},
                    attrs={
                        'file_path': os.path.join(dirname, origin_var.name),
                        'seek': offset,
                        'shape': slice.shape,
                    },
                )
            else:
                origin = load_block.create_var(
                    name=f"{origin_var.name}",
                    type=origin_var.type,
                    shape=origin_var.shape,
                    dtype=origin_var.dtype,
                    persistable=True,
                )
                load_block.append_op(
                    type='load',
                    inputs={},
                    outputs={'Out': [origin]},
                    attrs={'file_path': os.path.join(dirname, origin_var.name)},
                )

        executor.run(load_prog)

    if not isinstance(main_program, Program):
        raise TypeError("'main_program' should be an instance of Program.")

    if not main_program._is_distributed:
        raise ValueError(
            "'_load_distributed_persistables' just be designed for distributed training."
        )

    if not main_program._ps_endpoint:
        raise ValueError(
            "'_load_distributed_persistables' need current_endpoint set in DistributeTranspiler.transpile"
        )

    need_load_vars = (
        main_program._parameters_on_pservers.get_distributed_vars_by_ep(
            main_program._ps_endpoint
        )
    )
    __load_persistable_vars(executor, dirname, need_load_vars)


@dygraph_not_support
def load_persistables(executor, dirname, main_program=None, filename=None):
    """
    :api_attr: Static Graph

    This API filters out all variables with ``persistable==True`` from the
    given ``main_program`` and then tries to load these variables from the
    directory ``dirname`` or the file ``filename``.

    Use the ``dirname`` to specify the directory where persistable variables
    (refer to :ref:`api_guide_model_save_reader_en`) were saved. If variables
    were saved in separate files, set ``filename`` as None; if all variables
    were saved in a single file, use ``filename`` to specify the file name.

    Args:
        executor(Executor): The executor used for loading persistable variables.
                            See :ref:`api_guide_executor_en` for more details about it.
        dirname(str): The directory path.
        main_program(Program, optional): The program whose persistable variables will
                                    be loaded. If it is None, the ``default_main_program``
                                    will be used automatically. See :ref:`api_guide_Program_en`
                                    for more about ``Program``.
                                    Default: None.
        filename(str, optional): The file which saved all persistable variables. If variables
                                 were saved in separated files, set it to None.
                                 Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.base as base

            >>> paddle.enable_static()
            >>> exe = base.Executor(base.CPUPlace())
            >>> param_path = "./my_paddle_model"
            >>> prog = base.default_main_program()
            >>> paddle.distributed.io.load_persistables(executor=exe, dirname=param_path,
            ...                             main_program=None)
    """

    if main_program and main_program._is_distributed:
        _load_distributed_persistables(
            executor, dirname=dirname, main_program=main_program
        )
    else:
        paddle.static.io.load_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            predicate=is_persistable,
            filename=filename,
        )


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

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle
            >>> import paddle

            >>> paddle.enable_static()
            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> param_path = "./my_paddle_model"
            >>> t = paddle.distributed.transpiler.DistributeTranspiler()
            >>> t.transpile(...)
            >>> train_program = t.get_trainer_program()
            >>> _save_distributed_persistables(executor=exe, dirname=param_path, main_program=train_program)
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
                slice_varnames[index] = f"{slice.name}.slice.{idx}"
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


            >>> import paddle
            >>> paddle.enable_static()
            >>> image = paddle.static.data(
            ...     name='image', shape=[None, 28], dtype='float32')
            >>> bias_attr = paddle.ParamAttr('fc.b')
            >>> fc = paddle.static.nn.fc(image, size=10, bias_attr=bias_attr)
            >>> param = paddle.static.default_main_program().global_block().var('fc.b')
            >>> res = paddle.distributed.io.is_persistable(param)

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
        main_program(Program, optional): The program whose persistable variables will
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

            >>> import paddle

            >>> paddle.enable_static()
            >>> dir_path = "./my_paddle_model"
            >>> file_name = "persistables"
            >>> image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            >>> label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            >>> feeder = paddle.base.DataFeeder(feed_list=[image, label], place=paddle.CPUPlace())

            >>> predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')
            >>> loss = paddle.nn.functional.cross_entropy(input=predict, label=label)
            >>> avg_loss = paddle.mean(loss)
            >>> exe = paddle.static.Executor(paddle.CPUPlace())
            >>> exe.run(paddle.static.default_startup_program())
            >>> paddle.distributed.io.save_persistables(executor=exe, dirname=dir_path, filename=file_name)
            >>> # The persistables variables weights and bias in the fc layer of the network
            >>> # are going to be saved in the same file named "persistables" in the path
            >>> # "./my_paddle_model"
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


@static_only
def load_inference_model_distributed(
    dirname,
    executor,
    model_filename=None,
    params_filename=None,
    pserver_endpoints=None,
):
    """
    Load the inference model from a given directory. By this API, you can get the model
    structure(Inference Program) and model parameters.
    You can refer to :ref:`api_guide_model_save_reader_en` for more details.

    Args:
        dirname(str): One of the following:
          - The given directory path.
          - Set to None when reading the model from memory.
        executor(Executor): The executor to run for loading inference model.
                            See :ref:`api_guide_executor_en` for more details about it.
        model_filename(str, optional): One of the following:
          - The name of file to load the inference program.
          - If it is None, the default filename ``__model__`` will be used.
          - When ``dirname`` is ``None``, it must be set to a string containing model.
          Default: ``None``.
        params_filename(str, optional): It is only used for the case that all
            parameters were saved in a single binary file. One of the following:
          - The name of file to load all parameters.
          - When ``dirname`` is ``None``, it must be set to a string containing all the parameters.
          - If parameters were saved in separate files, set it as ``None``.
            Default: ``None``.

        pserver_endpoints(list, optional): It is only needed by the distributed inference.
                                    If using a distributed look up table during the training,
                                    this table is also needed by the inference process. Its value is
                                    a list of pserver endpoints.

    Returns:
        list: The return of this API is a list with three elements:
        (program, feed_target_names, fetch_targets). The `program` is a
        ``Program`` (refer to :ref:`api_guide_Program_en`), which is used for inference.
        The `feed_target_names` is a list of ``str``, which contains names of variables
        that need to feed data in the inference program. The `fetch_targets` is a list of
        ``Variable`` (refer to :ref:`api_guide_Program_en`). It contains variables from which
        we can get inference results.


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.base as base
            >>> import numpy as np

            >>> paddle.enable_static()
            >>> # Build the model
            >>> main_prog = paddle.static.Program()
            >>> startup_prog = paddle.static.Program()
            >>> with paddle.static.program_guard(main_prog, startup_prog):
            ...     data = paddle.static.data(name="img", shape=[64, 784], append_batch_size=False)
            ...     w = paddle.create_parameter(shape=[784, 200], dtype='float32')
            ...     b = paddle.create_parameter(shape=[200], dtype='float32')
            ...     hidden_w = paddle.matmul(x=data, y=w)
            ...     hidden_b = base.layers.elementwise_add(hidden_w, b)
            >>> place = base.CPUPlace()
            >>> exe = base.Executor(place)
            >>> exe.run(startup_prog)

            >>> # Save the inference model
            >>> path = "./infer_model"
            >>> base.io.save_inference_model(dirname=path, feeded_var_names=['img'],
            ...                 target_vars=[hidden_b], executor=exe, main_program=main_prog)
            ...
            >>> # Demo one. Not need to set the distributed look up table, because the
            >>> # training doesn't use a distributed look up table.
            >>> [inference_program, feed_target_names, fetch_targets] = (
            ...     paddle.distributed.io.load_inference_model_distributed(dirname=path, executor=exe))
            >>> tensor_img = np.array(np.random.random((1, 64, 784)), dtype=np.float32)
            >>> results = exe.run(inference_program,
            ...                 feed={feed_target_names[0]: tensor_img},
            ...                 fetch_list=fetch_targets)
            ...
            >>> # Demo two. If the training uses a distributed look up table, the pserver
            >>> # endpoints list should be supported when loading the inference model.
            >>> # The below is just an example.
            >>> endpoints = ["127.0.0.1:2023","127.0.0.1:2024"]
            >>> [dist_inference_program, dist_feed_target_names, dist_fetch_targets] = (
            ...     paddle.distributed.io.load_inference_model_distributed(dirname=path,
            ...                                     executor=exe,
            ...                                     pserver_endpoints=endpoints))
            ...
            >>> # In this example, the inference program was saved in the file
            >>> # "./infer_model/__model__" and parameters were saved in
            >>> # separate files under the directory "./infer_model".
            >>> # By the inference program, feed_target_names and
            >>> # fetch_targets, we can use an executor to run the inference
            >>> # program for getting the inference result.
    """
    load_from_memory = False
    if dirname is not None:
        load_dirname = os.path.normpath(dirname)
        if not os.path.isdir(load_dirname):
            raise ValueError(f"There is no directory named '{dirname}'")

        if model_filename is None:
            model_filename = '__model__'

        model_filename = os.path.join(
            load_dirname, os.path.basename(model_filename)
        )

        if params_filename is not None:
            params_filename = os.path.basename(params_filename)

        with open(model_filename, "rb") as f:
            program_desc_str = f.read()
    else:
        load_from_memory = True
        if params_filename is None:
            raise ValueError(
                "The path of params cannot be None when the directory path is None."
            )
        load_dirname = dirname
        program_desc_str = model_filename
        params_filename = params_filename

    program = Program.parse_from_string(program_desc_str)
    if not core._is_program_version_supported(program._version()):
        raise ValueError(f"Unsupported program version: {program._version()}\n")
    # Binary data also need versioning.
    load_persistables(executor, load_dirname, program, params_filename)

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]
