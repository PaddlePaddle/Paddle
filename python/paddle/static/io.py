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


import errno
import inspect
import logging
import os
import six

import paddle
from paddle.fluid import core, Variable, CompiledProgram, program_guard, default_main_program, Program
from paddle.fluid.framework import static_only
from paddle.fluid import layers

from paddle.fluid.io import _get_valid_program, save_vars, _save_distributed_persistables
from paddle.fluid.io import prepend_feed_ops, append_fetch_ops, save_persistables
from paddle.fluid.io import load_persistables, _endpoints_replacement
from paddle.fluid.log_helper import get_logger


__all__ = [
    'save_inference_model',
    'load_inference_model',
]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def _check_args(caller, args, supported_args=[], deprecated_args=[]):
    for arg in args:
        if arg in deprecated_args:
            raise ValueError("argument '{}' in function '{}' is deprecated, only {} are supported.".format(arg, caller, supported_args))
        elif arg not in supported_args:
            raise ValueError(
                "function '{}' doesn't support argument '{}',\n only {} are supported.".format(caller, arg, supported_args))


@static_only
def save_inference_model(path_prefix, feed_vars, fetch_vars, executor):
    """
    :api_attr: Static Graph

    Save current model and its parameters to given path.
    If you just want to save parameters of your trained model, please use the
    :ref:`api_fluid_io_save_params` . You can refer to :ref:`api_guide_model_save_reader_en`
    for more details.

    Note:
        The :code:`dirname` is used to specify the folder where inference model
        structure and parameters are going to be saved. If you would like to save params of
        Program in separate files, set `params_filename` None; if you would like to save all
        params of Program in a single file, use `params_filename` to specify the file name.

    Args:
        path(str): The directory path to save the inference model.
        feed_vars(Variable | list[str]): Variable or list of Variable. Variables needed by
                                     inference.
        fetch_vars(Variable | list[Variable]): Variable or list of Variable. Variables returned by
                                     inference.
        executor(Executor): The executor that saves the inference model. You can refer
                            to :ref:`api_guide_executor_en` for more details.
    Returns:
        The fetch variables' name list

     Return Type:
        None

    Raises:
        ValueError: If `feed_vars` is not a Variable or a list of Variable, an exception is thrown.
        ValueError: If `fetch_vars` is not a Variable or a list of Variable, an exception is thrown.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            path = "./infer_model"

            # User defined network, here a softmax regession example
            image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
            predict = fluid.layers.fc(input=image, size=10, act='softmax')

            loss = fluid.layers.cross_entropy(input=predict, label=label)
            avg_loss = fluid.layers.mean(loss)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            # Feed data and train process

            # Save inference model. Note we don't save label and loss in this example
            fluid.io.save_inference_model(dirname=path,
                                          feeded_var_names=['img'],
                                          fetch_vars=[predict],
                                          executor=exe)

            # In this example, the save_inference_mode inference will prune the default
            # main program according to the network's input node (img) and output node(predict).
            # The pruned inference program is going to be saved in the "./infer_model/__model__"
            # and parameters are going to be saved in separate files under folder
            # "./infer_model".

    """
    # check path_prefix, set model_path and params_path
    if not isinstance(path_prefix, six.string_types):
        raise ValueError("'path_prefix' should be a string.")
    if path_prefix.endswith("/"):
        raise ValueError("'path_prefix' should not be a directory")
    path_prefix = os.path.normpath(path_prefix)
    path_prefix = os.path.abspath(path_prefix)
    try:
        # mkdir may conflict if pserver and trainer are running on the same machine
        dirname = os.path.dirname(path_prefix)
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    model_path = path_prefix + ".pdmodel"
    params_path = path_prefix + ".pdiparams"
    if os.path.isdir(model_path):
        raise ValueError("'{}' is an existing directory.".format(model_path))
    if os.path.isdir(params_path):
        raise ValueError("'{}' is an existing directory.".format(params_path))

    # verify feed_vars
    if not isinstance(feed_vars, list):
        feed_vars = [feed_vars]
    if not feed_vars or not all([isinstance(var, Variable) for var in feed_vars]):
        raise ValueError("'feed_vars' should be a Variable or a list of Variable.")

    # verify fetch_vars
    if not isinstance(fetch_vars, list):
        fetch_vars = [fetch_vars]
    if not fetch_vars or not all([isinstance(var, Variable) for var in fetch_vars]):
        raise ValueError("'fetch_vars' should be a Variable or a list of Variable.")

    main_program = _get_valid_program()
    # remind users to set auc_states to 0 if auc op were found.
    for op in main_program.global_block().ops:
        # clear device of Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        op._set_attr(device_attr_name, "")
        if op.type == 'auc':
            warnings.warn("Be sure that you have set auc states to 0 before saving inference model.")
            break

    # fix the bug that the activation op's output as target will be pruned.
    # will affect the inference performance.
    # TODO(Superjomn) add an IR pass to remove 1-scale op.
    with program_guard(main_program):
        uniq_fetch_vars = []
        for i, var in enumerate(fetch_vars):
            var = layers.scale(var, 1., name="save_infer_model/scale_{}".format(i))
            uniq_fetch_vars.append(var)
        fetch_vars = uniq_fetch_vars
    
    # save model
    origin_program = main_program.clone()
    main_program = main_program.clone()
    global_block = main_program.global_block()
    remove_op_idx = []
    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            remove_op_idx.append(i)
    for idx in remove_op_idx[::-1]:
        global_block._remove_op(idx)
    main_program.desc.flush()

    feed_var_names = [var.name for var in feed_vars]
    main_program = main_program._prune_with_input(
        feeded_var_names=feed_var_names, targets=fetch_vars)
    main_program = main_program._inference_optimize(prune_read_op=True)
    fetch_var_names = [var.name for var in fetch_vars]
    prepend_feed_ops(main_program, feed_var_names)
    append_fetch_ops(main_program, fetch_var_names)
    main_program.desc._set_version()
    paddle.fluid.core.save_op_version_info(main_program.desc)
    with open(model_path, "wb") as f:
        f.write(main_program.desc.serialize_to_string())
    main_program._copy_dist_param_info_from(origin_program)

    # save params
    dirname = os.path.dirname(params_path)
    basename = os.path.basename(params_path)
    save_persistables(executor, dirname, main_program, basename)


@static_only
def load_inference_model(path_prefix, executor, **configs):
    """
    :api_attr: Static Graph

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

    Raises:
        ValueError: If `dirname` is not a existing directory.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Build the model
            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
                w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
                b = fluid.layers.create_parameter(shape=[200], dtype='float32')
                hidden_w = fluid.layers.matmul(x=data, y=w)
                hidden_b = fluid.layers.elementwise_add(hidden_w, b)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_prog)

            # Save the inference model
            path = "./infer_model"
            fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],
                         fetch_vars=[hidden_b], executor=exe, main_program=main_prog)

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
    # check configs
    supported_args = ('model_filename', 'params_filename')
    deprecated_args = ('pserver_endpoints',)
    caller = inspect.currentframe().f_code.co_name
    _check_args(caller, configs, supported_args, deprecated_args)

    # load from memory
    if path_prefix is None:
        _logger.warning("Load inference model from memory is deprecated.")
        model_filename = configs.get('model_filename', None)
        params_filename = configs.get('params_filename', None)
        if params_filename is None:
            raise ValueError(
                "params_filename cannot be None when path_prefix is None."
            )
        load_dirname = path_prefix
        program_desc_str = model_filename
        params_filename = params_filename
    # load from file
    else:
        # check and norm path_prefix
        if not isinstance(path_prefix, six.string_types):
            raise ValueError("'path_prefix' should be a string.")
        if path_prefix.endswith("/"):
            raise ValueError("'path_prefix' should not be a directory")
        path_prefix = os.path.normpath(path_prefix)
        path_prefix = os.path.abspath(path_prefix)

        # set model_path and params_path in new way,
        # path_prefix represents a file path without suffix in this case.
        if not configs:
            model_path = path_prefix + ".pdmodel"
            params_path = path_prefix + ".pdiparams"
        # set model_path and params_path in old way for compatible,
        # path_prefix represents a directory path.
        else:
            model_filename = configs.get('model_filename', None)
            params_filename = configs.get('params_filename', None)
            # set model_path
            if model_filename is None:
                model_path = os.path.join(path_prefix, "__model__")
            else:
                model_path = os.path.join(path_prefix, model_filename + ".pdmodel")
                if not os.path.exists(model_path):
                    model_path = os.path.join(path_prefix, model_filename)
            # set params_path
            if params_filename is None:
                params_path = os.path.join(path_prefix, "")
            else:
                params_path = os.path.join(path_prefix, params_filename + ".pdiparams")
                if not os.path.exists(params_path):
                    params_path = os.path.join(path_prefix, params_filename)
            _logger.warning("The old way to load inference model is deprecated."
                    " model path: {}, params path: {}".format(model_path, params_path))
        with open(model_path, "rb") as f:
            program_desc_str = f.read()
        load_dirname = os.path.dirname(params_path)
        params_filename = os.path.basename(params_path)


    program = Program.parse_from_string(program_desc_str)
    if not core._is_program_version_supported(program._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program._version())
    # Binary data also need versioning.
    load_persistables(executor, load_dirname, program, params_filename)

    feed_target_names = program.desc.get_feed_target_names()
    fetch_target_names = program.desc.get_fetch_target_names()
    fetch_targets = [
        program.global_block().var(name) for name in fetch_target_names
    ]

    return [program, feed_target_names, fetch_targets]

