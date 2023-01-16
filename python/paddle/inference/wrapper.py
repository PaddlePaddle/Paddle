#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Set

import numpy as np

import paddle
import paddle.fluid.core as core
from paddle.fluid.core import (
    AnalysisConfig,
    PaddleDType,
    PaddleInferPredictor,
    PaddleInferTensor,
    PaddlePlace,
    convert_to_mixed_precision_bind,
)

DataType = PaddleDType
PlaceType = PaddlePlace
PrecisionType = AnalysisConfig.Precision
Config = AnalysisConfig
Tensor = PaddleInferTensor
Predictor = PaddleInferPredictor


def tensor_copy_from_cpu(self, data):
    '''
    Support input type check based on tensor.copy_from_cpu.
    '''
    if isinstance(data, np.ndarray) or (
        isinstance(data, list) and len(data) > 0 and isinstance(data[0], str)
    ):
        self.copy_from_cpu_bind(data)
    else:
        raise TypeError(
            "In copy_from_cpu, we only support numpy ndarray and list[str] data type."
        )


def tensor_share_external_data(self, data):
    '''
    Support input type check based on tensor.share_external_data.
    '''
    if isinstance(data, core.LoDTensor):
        self.share_external_data_bind(data)
    elif isinstance(data, paddle.Tensor):
        self.share_external_data_paddle_tensor_bind(data)
    else:
        raise TypeError(
            "In share_external_data, we only support LoDTensor data type."
        )


def convert_to_mixed_precision(
    model_file: str,
    params_file: str,
    mixed_model_file: str,
    mixed_params_file: str,
    mixed_precision: PrecisionType,
    backend: PlaceType,
    keep_io_types: bool = True,
    black_list: Set = set(),
):
    '''
    Convert a fp32 model to mixed precision model.

    Args:
        model_file: fp32 model file, e.g. inference.pdmodel.
        params_file: fp32 params file, e.g. inference.pdiparams.
        mixed_model_file: The storage path of the converted mixed-precision model.
        mixed_params_file: The storage path of the converted mixed-precision params.
        mixed_precision: The precision, e.g. PrecisionType.Half.
        backend: The backend, e.g. PlaceType.GPU.
        keep_io_types: Whether the model input and output dtype remains unchanged.
        black_list: Operators that do not convert precision.
    '''
    mixed_model_dirname = os.path.dirname(mixed_model_file)
    mixed_params_dirname = os.path.dirname(mixed_params_file)
    if not os.path.exists(mixed_model_dirname):
        os.makedirs(mixed_model_dirname)
    if not os.path.exists(mixed_params_dirname):
        os.makedirs(mixed_params_dirname)
    convert_to_mixed_precision_bind(
        model_file,
        params_file,
        mixed_model_file,
        mixed_params_file,
        mixed_precision,
        backend,
        keep_io_types,
        black_list,
    )


Tensor.copy_from_cpu = tensor_copy_from_cpu
Tensor.share_external_data = tensor_share_external_data
