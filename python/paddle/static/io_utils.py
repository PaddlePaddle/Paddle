# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import pickle
import warnings

import paddle
from paddle import pir
from paddle.base import (
    CompiledProgram,
    Variable,
)


def _check_args(caller, args, supported_args=None, deprecated_args=None):
    supported_args = [] if supported_args is None else supported_args
    deprecated_args = [] if deprecated_args is None else deprecated_args
    for arg in args:
        if arg in deprecated_args:
            raise ValueError(
                f"argument '{arg}' in function '{caller}' is deprecated, only {supported_args} are supported."
            )
        elif arg not in supported_args:
            raise ValueError(
                f"function '{caller}' doesn't support argument '{arg}',\n only {supported_args} are supported."
            )


def _check_vars(name, var_list):
    if not isinstance(var_list, list):
        var_list = [var_list]
    if not all(isinstance(var, (Variable, pir.Value)) for var in var_list):
        raise ValueError(
            f"'{name}' should be a Variable or a list of Variable."
        )


def _normalize_path_prefix(path_prefix):
    """
    convert path_prefix to absolute path.
    """
    if not isinstance(path_prefix, str):
        raise ValueError("'path_prefix' should be a string.")
    if path_prefix.endswith("/"):
        raise ValueError("'path_prefix' should not be a directory")
    path_prefix = os.path.normpath(path_prefix)
    path_prefix = os.path.abspath(path_prefix)
    return path_prefix


def _get_valid_program(program=None):
    """
    return default main program if program is None.
    """
    if program is None:
        program = paddle.static.default_main_program()
    elif isinstance(program, CompiledProgram):
        program = program._program
        if program is None:
            raise TypeError(
                "The type of input program is invalid, expected type is Program, but received None"
            )
        warnings.warn(
            "The input is a CompiledProgram, this is not recommended."
        )
    if not isinstance(program, paddle.static.Program):
        raise TypeError(
            f"The type of input program is invalid, expected type is base.Program, but received {type(program)}"
        )
    return program


def _safe_load_pickle(file, encoding="ASCII"):
    load_dict = pickle.Unpickler(file, encoding=encoding).load()
    return load_dict
