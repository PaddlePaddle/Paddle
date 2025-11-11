// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#pragma once
#include "paddle/phi/kernels/legacy/gpu/ln.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <unordered_map>

#include "paddle/extension.h"

namespace layer_norm {
// Create registries and provide runtime versions of config hash functions.

FwdRegistry FWD_FUNCS;
BwdRegistry BWD_FUNCS;

uint32_t get_type_id(paddle::DataType dtype) {
  if (dtype == paddle::DataType::FLOAT16) {
    return TypeToIdTrait<fp16>::Value;
  } else if (dtype == paddle::DataType::BFLOAT16) {
    return TypeToIdTrait<bf16>::Value;
  } else if (dtype == paddle::DataType::FLOAT32) {
    return TypeToIdTrait<float>::Value;
  } else {
    PD_CHECK(false, "Type not supported: ", dtype);
  }
}

uint64_t get_key(paddle::DataType weight_type,
                 paddle::DataType input_type,
                 paddle::DataType output_type,
                 paddle::DataType compute_type,
                 uint64_t hidden_size) {
  uint64_t type_key =
      get_type_id(weight_type) | (get_type_id(input_type) << 2) |  // NOLINT
      (get_type_id(output_type) << 4) | (get_type_id(compute_type) << 6);
  uint64_t launcher_key = (type_key << 32) | hidden_size;
  return launcher_key;
}

}  // namespace layer_norm

layer_norm::FwdFunction& get_fwd_launcher(paddle::DataType weight_type,
                                          paddle::DataType input_type,
                                          paddle::DataType output_type,
                                          paddle::DataType compute_type,
                                          uint32_t hidden_size) {
  auto iter = layer_norm::FWD_FUNCS.find(layer_norm::get_key(
      weight_type, input_type, output_type, compute_type, hidden_size));
  if (iter != layer_norm::FWD_FUNCS.end()) {
    return iter->second;
  } else {
    PD_CHECK(false,
             "FWD: Unsupported hidden_size or types: ",
             hidden_size,
             weight_type,
             input_type,
             output_type,
             compute_type);
  }
}

layer_norm::BwdFunction& get_bwd_launcher(paddle::DataType weight_type,
                                          paddle::DataType input_type,
                                          paddle::DataType output_type,
                                          paddle::DataType compute_type,
                                          uint32_t hidden_size) {
  auto iter = layer_norm::BWD_FUNCS.find(layer_norm::get_key(
      weight_type, input_type, output_type, compute_type, hidden_size));
  if (iter != layer_norm::BWD_FUNCS.end()) {
    return iter->second;
  } else {
    PD_CHECK(false,
             "BWD: Unsupported hidden_size or types: ",
             hidden_size,
             weight_type,
             input_type,
             output_type,
             compute_type);
  }
}

void LaunchNormFwd(const cudaStream_t& stream,
                   const paddle::Place& place,
                   const void* x_ptr,
                   const void* scale_ptr,
                   const void* bias_ptr,
                   void* y_ptr,
                   void* mean_ptr,
                   void* invvar_ptr,
                   const paddle::DataType weight_type,
                   const paddle::DataType input_type,
                   const paddle::DataType output_type,
                   const paddle::DataType compute_type,
                   const uint32_t hidden_size,
                   const int64_t rows,
                   const int64_t cols,
                   const float epsilon) {
  layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

  launch_params.props = GetDeviceProp();
  launch_params.stream = stream;

  // Request the kernel launcher.
  auto launcher = get_fwd_launcher(
      weight_type, input_type, output_type, compute_type, hidden_size);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  // Set the kernel runtime parameters.
  layer_norm::FwdParams& params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = const_cast<void*>(x_ptr);
  params.scale = const_cast<void*>(scale_ptr);
  params.bias = const_cast<void*>(bias_ptr);
  params.y = y_ptr;
  params.mean = mean_ptr;
  params.invvar = invvar_ptr;
  params.epsilon = epsilon;

  paddle::Tensor workspace, barrier;
  if (launch_params.barrier_size > 0) {
    barrier = paddle::zeros({static_cast<int64_t>(launch_params.barrier_size)},
                            paddle::DataType::INT32,
                            place);
    workspace =
        paddle::empty({static_cast<int64_t>(launch_params.workspace_bytes)},
                      paddle::DataType::UINT8,
                      place);
    params.workspace = workspace.data();
    params.barrier = barrier.data<int>();
  }

  launcher(launch_params, false);
}

void LaunchNormBwd(const cudaStream_t& stream,
                   const paddle::Place& place,
                   const void* x_ptr,
                   const void* scale_ptr,
                   const void* mean_ptr,
                   const void* invvar_ptr,
                   const void* dy_ptr,
                   void* dx_ptr,
                   void* dscale_ptr,
                   void* dbias_ptr,
                   const paddle::DataType weight_type,
                   const paddle::DataType input_type,
                   const paddle::DataType output_type,
                   const paddle::DataType compute_type,
                   const uint32_t hidden_size,
                   const int64_t rows,
                   const int64_t cols,
                   const float epsilon) {
  layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
  launch_params.stream = stream;
  launch_params.props = GetDeviceProp();

  auto launcher = get_bwd_launcher(
      weight_type, input_type, output_type, compute_type, hidden_size);

  launcher(launch_params, true);

  paddle::Tensor dscale_part, dbias_part;

  dscale_part = paddle::empty(
      {launch_params.params.ctas_per_col, hidden_size}, compute_type, place);
  if (dbias_ptr) {
    dbias_part = paddle::empty(
        {launch_params.params.ctas_per_col, hidden_size}, compute_type, place);
  }

  layer_norm::BwdParams& params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = const_cast<void*>(x_ptr);
  params.scale = const_cast<void*>(scale_ptr);
  params.mean = const_cast<void*>(mean_ptr);
  params.invvar = const_cast<void*>(invvar_ptr);
  params.dy = const_cast<void*>(dy_ptr);
  params.dx = dx_ptr;
  params.dscale = dscale_ptr;
  params.dbias = dbias_ptr;
  params.dscale_part = dscale_part.data();
  params.dbias_part = dbias_ptr ? dbias_part.data() : nullptr;

  paddle::Tensor workspace, barrier;
  if (launch_params.barrier_size > 0) {
    barrier = paddle::zeros({static_cast<int64_t>(launch_params.barrier_size)},
                            paddle::DataType::INT32,
                            place);
    workspace =
        paddle::empty({static_cast<int64_t>(launch_params.workspace_bytes)},
                      paddle::DataType::UINT8,
                      place);
    params.workspace = workspace.data();
    params.barrier = barrier.data<int>();
  }

  launcher(launch_params, false);
}
