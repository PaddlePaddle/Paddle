// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/fast_ln_v2_common.h"

namespace phi {
namespace funcs {
namespace fast_ln_v2 {

#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)

bool has_fast_ln_v2_fwd_kernel(phi::DataType weight_type,
                               phi::DataType input_type,
                               phi::DataType output_type,
                               phi::DataType compute_type,
                               uint32_t hidden_size);

bool has_fast_ln_v2_bwd_kernel(phi::DataType weight_type,
                               phi::DataType input_type,
                               phi::DataType output_type,
                               phi::DataType compute_type,
                               uint32_t hidden_size);

FwdFunction& get_fwd_launcher(phi::DataType weight_type,
                              phi::DataType input_type,
                              phi::DataType output_type,
                              phi::DataType compute_type,
                              uint32_t hidden_size);

BwdFunction& get_bwd_launcher(phi::DataType weight_type,
                              phi::DataType input_type,
                              phi::DataType output_type,
                              phi::DataType compute_type,
                              uint32_t hidden_size);

inline static cudaDeviceProp GetDevicePropImpl() {
  int device = -1;
  PD_CHECK(cudaGetDevice(&device) == cudaSuccess);
  cudaDeviceProp prop;
  PD_CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
  return prop;
}

inline static cudaDeviceProp* GetDeviceProp() {
  static auto prop = GetDevicePropImpl();
  return &prop;
}

template <typename T, typename Context>
void LaunchNormFwd(const Context& dev_ctx,
                   const cudaStream_t& stream,
                   const phi::Place& place,
                   const void* x_ptr,
                   const void* scale_ptr,
                   const void* bias_ptr,
                   void* y_ptr,
                   void* mean_ptr,
                   void* invvar_ptr,
                   const phi::DataType weight_type,
                   const phi::DataType input_type,
                   const phi::DataType output_type,
                   const phi::DataType compute_type,
                   const uint32_t hidden_size,
                   const int64_t rows,
                   const int64_t cols,
                   const float epsilon) {
  LaunchParams<FwdParams> launch_params;

  launch_params.props = GetDeviceProp();
  launch_params.stream = stream;

  // Request the kernel launcher.
  auto launcher = get_fwd_launcher(
      weight_type, input_type, output_type, compute_type, hidden_size);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  // Set the kernel runtime parameters.
  FwdParams& params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = const_cast<void*>(x_ptr);
  params.scale = const_cast<void*>(scale_ptr);
  params.bias = const_cast<void*>(bias_ptr);
  params.y = y_ptr;
  params.mean = mean_ptr;
  params.invvar = invvar_ptr;
  params.epsilon = epsilon;

  DenseTensor workspace = phi::Empty<uint8_t, Context>(
      dev_ctx,
      phi::IntArray({static_cast<int64_t>(launch_params.workspace_bytes)}));
  DenseTensor barrier = phi::Full<int, Context>(
      dev_ctx,
      phi::IntArray({static_cast<int64_t>(launch_params.barrier_size)}),
      0);

  params.workspace = workspace.data();
  params.barrier = barrier.data<int>();

  launcher(launch_params, false);
}

template <typename T, typename Context>
void LaunchNormBwd(const Context& dev_ctx,
                   const cudaStream_t& stream,
                   const phi::Place& place,
                   const void* x_ptr,
                   const void* scale_ptr,
                   const void* mean_ptr,
                   const void* invvar_ptr,
                   const void* dy_ptr,
                   void* dx_ptr,
                   void* dscale_ptr,
                   void* dbias_ptr,
                   const phi::DataType weight_type,
                   const phi::DataType input_type,
                   const phi::DataType output_type,
                   const phi::DataType compute_type,
                   const uint32_t hidden_size,
                   const int64_t rows,
                   const int64_t cols,
                   const float epsilon) {
  LaunchParams<BwdParams> launch_params;
  launch_params.stream = stream;
  launch_params.props = GetDeviceProp();

  auto launcher = get_bwd_launcher(
      weight_type, input_type, output_type, compute_type, hidden_size);

  launcher(launch_params, true);

  DenseTensor dscale_part, dbias_part;
  dscale_part = phi::Empty<float, Context>(
      dev_ctx,
      phi::IntArray({static_cast<int64_t>(launch_params.params.ctas_per_col),
                     static_cast<int64_t>(hidden_size)}));
  if (dbias_ptr) {
    dbias_part = phi::Empty<float, Context>(
        dev_ctx,
        phi::IntArray({static_cast<int64_t>(launch_params.params.ctas_per_col),
                       static_cast<int64_t>(hidden_size)}));
  }

  BwdParams& params = launch_params.params;
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

  DenseTensor workspace = phi::Empty<uint8_t, Context>(
      dev_ctx,
      phi::IntArray({static_cast<int64_t>(launch_params.workspace_bytes)}));
  DenseTensor barrier = phi::Full<int, Context>(
      dev_ctx,
      phi::IntArray({static_cast<int64_t>(launch_params.barrier_size)}),
      0);

  params.workspace = workspace.data();
  params.barrier = barrier.data<int>();
  launcher(launch_params, false);
}

#endif

}  // namespace fast_ln_v2
}  // namespace funcs
}  // namespace phi
