// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/cinn/common/macros.h"

CINN_USE_REGISTER(paddle_argsort)
CINN_USE_REGISTER(paddle_fetch_feed)
CINN_USE_REGISTER(paddle_mul)
CINN_USE_REGISTER(paddle_slice)
CINN_USE_REGISTER(paddle_relu)
CINN_USE_REGISTER(paddle_softmax)
CINN_USE_REGISTER(paddle_scale)
CINN_USE_REGISTER(paddle_batchnorm)
CINN_USE_REGISTER(paddle_dropout)
CINN_USE_REGISTER(paddle_elementwise)
CINN_USE_REGISTER(paddle_pool2d)
CINN_USE_REGISTER(paddle_conv2d)
CINN_USE_REGISTER(paddle_transpose)
CINN_USE_REGISTER(paddle_reshape)
CINN_USE_REGISTER(paddle_matmul)
CINN_USE_REGISTER(paddle_compare)
CINN_USE_REGISTER(paddle_log)
CINN_USE_REGISTER(paddle_concat)
CINN_USE_REGISTER(paddle_constant)
CINN_USE_REGISTER(paddle_where)
CINN_USE_REGISTER(paddle_layer_norm)
CINN_USE_REGISTER(paddle_squeeze)
CINN_USE_REGISTER(paddle_clip)
CINN_USE_REGISTER(paddle_unsqueeze)
CINN_USE_REGISTER(paddle_expand)
CINN_USE_REGISTER(paddle_lookup_table)
CINN_USE_REGISTER(paddle_take_along_axis)
CINN_USE_REGISTER(paddle_unary)
CINN_USE_REGISTER(paddle_binary)
CINN_USE_REGISTER(paddle_gather)
CINN_USE_REGISTER(paddle_gather_nd)
CINN_USE_REGISTER(paddle_reduce)
CINN_USE_REGISTER(paddle_atan)
CINN_USE_REGISTER(paddle_gaussian_random)
CINN_USE_REGISTER(paddle_uniform_random)
CINN_USE_REGISTER(paddle_top_k)
CINN_USE_REGISTER(paddle_one_hot)
CINN_USE_REGISTER(paddle_cumsum)
CINN_USE_REGISTER(paddle_norm)
CINN_USE_REGISTER(paddle_tile)
CINN_USE_REGISTER(paddle_strided_slice)
CINN_USE_REGISTER(paddle_arg)
CINN_USE_REGISTER(paddle_triangular_solve)
CINN_USE_REGISTER(paddle_flip)
CINN_USE_REGISTER(paddle_reverse)
CINN_USE_REGISTER(paddle_randint)
CINN_USE_REGISTER(paddle_roll)
CINN_USE_REGISTER(paddle_cholesky)
CINN_USE_REGISTER(paddle_scatter)

CINN_USE_REGISTER(science_broadcast)
CINN_USE_REGISTER(science_transform)
CINN_USE_REGISTER(science_math)
CINN_USE_REGISTER(science_compare)
