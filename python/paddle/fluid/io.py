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
from paddle.fluid import layers
from paddle.fluid.executor import Executor, global_scope
from paddle.fluid.framework import (
    Program,
    Parameter,
    default_main_program,
    default_startup_program,
    Variable,
    program_guard,
    dygraph_not_support,
    static_only,
)
from paddle.reader import (
    cache,
    map_readers,
    buffered,
    compose,
    chain,
    shuffle,
    ComposeNotAligned,
    firstn,
    xmap_readers,
    multiprocess_reader,
)
from .wrapped_decorator import signature_safe_contextmanager
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.log_helper import get_logger
from . import reader
from . import unique_name
from .reader import *
from . import core
from paddle.utils import deprecated
from paddle.fluid.framework import static_only

__all__ = [
    'save_inference_model',
    'load_inference_model',
] + reader.__all__


_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


def prepend_feed_ops(
    inference_program, feed_target_names, feed_holder_name='feed'
):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True,
    )

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                "The feeded_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                "Please check whether '{name}' is a valid feed_var name, or remove it from feeded_var_names "
                "if '{name}' is not involved in the target_vars calculation.".format(
                    i=i, name=name
                )
            )
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i},
        )


def append_fetch_ops(
    inference_program, fetch_target_names, fetch_holder_name='fetch'
):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True,
    )

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i},
        )


@static_only
@deprecated(since="2.0.0", update_to="paddle.static.save_inference_model")
def save_inference_model(
    dirname,
    feeded_var_names,
    target_vars,
    executor,
    main_program=None,
    model_filename=None,
    params_filename=None,
    export_for_deployment=True,
    program_only=False,
    clip_extra=True,
    legacy_format=False,
):
    """
    Prune the given `main_program` to build a new program especially for inference,
    and then save it and all related parameters to given `dirname` .
    If you just want to save parameters of your trained model, please use the
    :ref:`api_fluid_io_save_params` . You can refer to :ref:`api_guide_model_save_reader_en`
    for more details.

    Note:
        The :code:`dirname` is used to specify the folder where inference model
        structure and parameters are going to be saved. If you would like to save params of
        Program in separate files, set `params_filename` None; if you would like to save all
        params of Program in a single file, use `params_filename` to specify the file name.

    Args:
        dirname(str): The directory path to save the inference model.
        feeded_var_names(list[str]): list of string. Names of variables that need to be fed
                                     data during inference.
        target_vars(list[Variable]): list of Variable. Variables from which we can get
                                     inference results.
        executor(Executor): The executor that saves the inference model. You can refer
                            to :ref:`api_guide_executor_en` for more details.
        main_program(Program, optional): The original program, which will be pruned to
                                         build the inference model. If is set None,
                                         the global default :code:`_main_program_` will be used.
                                         Default: None.
        model_filename(str, optional): The name of file to save the inference program
                                       itself. If is set None, a default filename
                                       :code:`__model__` will be used.
        params_filename(str, optional): The name of file to save all related parameters.
                                        If it is set None, parameters will be saved
                                        in separate files .
        export_for_deployment(bool, optional): If True, programs are modified to only support
                                     direct inference deployment. Otherwise,
                                     more information will be stored for flexible
                                     optimization and re-training. Currently, only
                                     True is supported.
                                     Default: True.
        program_only(bool, optional): If True, It will save inference program only, and do not
                                      save params of Program.
                                      Default: False.
        legacy_format(bool, optional): Whether to save program in legacy format.
                                       Default: False.

    Returns:
        list, The fetch variables' name list.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            paddle.enable_static()
            path = "./infer_model"

            # User defined network, here a softmax regession example
            image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
            feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
            predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')

            loss = paddle.nn.functional.cross_entropy(
                input=predict, label=label,
                reduction='none', use_softmax=False
            )
            avg_loss = paddle.mean(loss)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            # Feed data and train process

            # Save inference model. Note we don't save label and loss in this example
            fluid.io.save_inference_model(dirname=path,
                                          feeded_var_names=['img'],
                                          target_vars=[predict],
                                          executor=exe)

            # In this example, the save_inference_mode inference will prune the default
            # main program according to the network's input node (img) and output node(predict).
            # The pruned inference program is going to be saved in the "./infer_model/__model__"
            # and parameters are going to be saved in separate files under folder
            # "./infer_model".

    """
    if isinstance(feeded_var_names, str):
        feeded_var_names = [feeded_var_names]
    elif export_for_deployment:
        if len(feeded_var_names) > 0:
            # TODO(paddle-dev): polish these code blocks
            if not (
                bool(feeded_var_names)
                and all(isinstance(name, str) for name in feeded_var_names)
            ):
                raise ValueError("'feed_var_names' should be a list of str.")

    if isinstance(target_vars, Variable):
        target_vars = [target_vars]
    elif export_for_deployment:
        if not (
            bool(target_vars)
            and all(isinstance(var, Variable) for var in target_vars)
        ):
            raise ValueError("'target_vars' should be a list of Variable.")

    main_program = paddle.static.io._get_valid_program(main_program)

    # remind user to set auc_states to zeros if the program contains auc op
    all_ops = main_program.global_block().ops
    for op in all_ops:
        # clear device of Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        op._set_attr(device_attr_name, "")
        if op.type == 'auc':
            warnings.warn(
                "please ensure that you have set the auc states to zeros before saving inference model"
            )
            break

    with program_guard(main_program):
        uniq_target_vars = []
        for i, var in enumerate(target_vars):
            uniq_target_vars.append(var)
        target_vars = uniq_target_vars
    target_var_name_list = [var.name for var in target_vars]

    # when a pserver and a trainer running on the same machine, mkdir may conflict
    save_dirname = dirname
    try:
        save_dirname = os.path.normpath(dirname)
        os.makedirs(save_dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if model_filename is not None:
        model_basename = os.path.basename(model_filename)
    else:
        model_basename = "__model__"
    model_basename = os.path.join(save_dirname, model_basename)

    # When export_for_deployment is true, we modify the program online so that
    # it can only be loaded for inference directly. If it's false, the whole
    # original program and related meta are saved so that future usage can be
    # more flexible.

    origin_program = main_program.clone()

    if export_for_deployment:
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

        main_program = main_program._prune_with_input(
            feeded_var_names=feeded_var_names, targets=target_vars
        )
        main_program = main_program._inference_optimize(prune_read_op=True)
        fetch_var_names = [v.name for v in target_vars]

        for target_v in target_vars:
            if not main_program.global_block().has_var(target_v.name):
                main_program.global_block().create_var(
                    name=target_v.name,
                    shape=target_v.shape,
                    dtype=target_v.dtype,
                    persistable=target_v.persistable,
                )

        prepend_feed_ops(main_program, feeded_var_names)
        append_fetch_ops(main_program, fetch_var_names)

        with open(model_basename, "wb") as f:
            f.write(
                main_program._remove_training_info(
                    clip_extra=clip_extra
                ).desc.serialize_to_string()
            )
    else:
        # TODO(panyx0718): Save more information so that it can also be used
        # for training and more flexible post-processing.
        with open(model_basename + ".main_program", "wb") as f:
            f.write(
                main_program._remove_training_info(
                    clip_extra=clip_extra
                ).desc.serialize_to_string()
            )

    if program_only:
        warnings.warn(
            "save_inference_model specified the param `program_only` to True, It will not save params of Program."
        )
        return target_var_name_list

    main_program._copy_dist_param_info_from(origin_program)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    paddle.distributed.io.save_persistables(
        executor, save_dirname, main_program, params_filename
    )
    return target_var_name_list


@static_only
@deprecated(since="2.0.0", update_to="paddle.static.load_inference_model")
def load_inference_model(
    dirname,
    executor,
    model_filename=None,
    params_filename=None,
    pserver_endpoints=None,
):
    """
    Load the inference model from a given directory. By this API, you can get the model
    structure(Inference Program) and model parameters. If you just want to load
    parameters of the pre-trained model, please use the :ref:`api_fluid_io_load_params` API.
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

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            paddle.enable_static()
            # Build the model
            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                data = paddle.static.data(name="img", shape=[-1, 64, 784])
                w = paddle.create_parameter(shape=[784, 200], dtype='float32')
                b = paddle.create_parameter(shape=[200], dtype='float32')
                hidden_w = paddle.matmul(x=data, y=w)
                hidden_b = paddle.add(hidden_w, b)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_prog)

            # Save the inference model
            path = "./infer_model"
            fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
                         target_vars=[hidden_b], executor=exe, main_program=main_prog)

            # Demo one. Not need to set the distributed look up table, because the
            # training doesn't use a distributed look up table.
            [inference_program, feed_target_names, fetch_targets] = (
                fluid.io.load_inference_model(dirname=path, executor=exe))
            tensor_img = np.array(np.random.random((1, 64, 784)), dtype=np.float32)
            results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)

            # Demo two. If the training uses a distributed look up table, the pserver
            # endpoints list should be supported when loading the inference model.
            # The below is just an example.
            endpoints = ["127.0.0.1:2023","127.0.0.1:2024"]
            [dist_inference_program, dist_feed_target_names, dist_fetch_targets] = (
                fluid.io.load_inference_model(dirname=path,
                                              executor=exe,
                                              pserver_endpoints=endpoints))

            # In this example, the inference program was saved in the file
            # "./infer_model/__model__" and parameters were saved in
            # separate files under the directory "./infer_model".
            # By the inference program, feed_target_names and
            # fetch_targets, we can use an executor to run the inference
            # program for getting the inference result.
    """
    load_from_memory = False
    if dirname is not None:
        load_dirname = os.path.normpath(dirname)
        if not os.path.isdir(load_dirname):
            raise ValueError("There is no directory named '%s'" % dirname)

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
        raise ValueError(
            "Unsupported program version: %d\n" % program._version()
        )
    # Binary data also need versioning.
    paddle.distributed.io.load_persistables(
        executor, load_dirname, program, params_filename
    )

    if pserver_endpoints:
        program = _endpoints_replacement(program, pserver_endpoints)

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]


def _endpoints_replacement(program, endpoints):
    ENDPOINT_MAP = "epmap"
    for op in program.global_block().ops:
        if op.has_attr(ENDPOINT_MAP):
            op.set_attr(ENDPOINT_MAP, endpoints)
    program._sync_with_cpp()
    return program
