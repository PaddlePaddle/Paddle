# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Contains common utility functions."""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import distutils.util
import os
import numpy as np
import six
import logging
import paddle.fluid as fluid
import paddle.compat as cpt
from paddle.fluid import core
from paddle.fluid.framework import Program

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def save_persistable_nodes(executor, dirname, graph):
    """
    Save persistable nodes to the given directory by the executor.

    Args:
        executor(Executor): The executor to run for saving node values.
        dirname(str): The directory path.
        graph(IrGraph): All the required persistable nodes in the graph will be saved.
    """
    persistable_node_names = set()
    persistable_nodes = []
    all_persistable_nodes = graph.all_persistable_nodes()
    for node in all_persistable_nodes:
        name = cpt.to_text(node.name())
        if name not in persistable_node_names:
            persistable_node_names.add(name)
            persistable_nodes.append(node)
    program = Program()
    var_list = []
    for node in persistable_nodes:
        var_desc = node.var()
        if var_desc.type() == core.VarDesc.VarType.RAW or \
                var_desc.type() == core.VarDesc.VarType.READER:
            continue
        var = program.global_block().create_var(
            name=var_desc.name(),
            shape=var_desc.shape(),
            dtype=var_desc.dtype(),
            type=var_desc.type(),
            lod_level=var_desc.lod_level(),
            persistable=var_desc.persistable())
        var_list.append(var)
    fluid.io.save_vars(executor=executor, dirname=dirname, vars=var_list)


def load_persistable_nodes(executor, dirname, graph):
    """
    Load persistable node values from the given directory by the executor.

    Args:
        executor(Executor): The executor to run for loading node values.
        dirname(str): The directory path.
        graph(IrGraph): All the required persistable nodes in the graph will be loaded.
    """
    persistable_node_names = set()
    persistable_nodes = []
    all_persistable_nodes = graph.all_persistable_nodes()
    for node in all_persistable_nodes:
        name = cpt.to_text(node.name())
        if name not in persistable_node_names:
            persistable_node_names.add(name)
            persistable_nodes.append(node)
    program = Program()
    var_list = []

    def _exist(var):
        return os.path.exists(os.path.join(dirname, var.name))

    def _load_var(name, scope):
        return np.array(scope.find_var(name).get_tensor())

    def _store_var(name, array, scope, place):
        tensor = scope.find_var(name).get_tensor()
        tensor.set(array, place)

    for node in persistable_nodes:
        var_desc = node.var()
        if var_desc.type() == core.VarDesc.VarType.RAW or \
                var_desc.type() == core.VarDesc.VarType.READER:
            continue
        var = program.global_block().create_var(
            name=var_desc.name(),
            shape=var_desc.shape(),
            dtype=var_desc.dtype(),
            type=var_desc.type(),
            lod_level=var_desc.lod_level(),
            persistable=var_desc.persistable())
        if _exist(var):
            var_list.append(var)
        else:
            _logger.info("Cannot find the var %s!!!" % (node.name()))
    fluid.io.load_vars(executor=executor, dirname=dirname, vars=var_list)
