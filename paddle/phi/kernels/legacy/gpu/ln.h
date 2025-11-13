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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <unordered_map>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

#define EMPTY_LIKE(dev_ctx, x)                                   \
  [&]() -> DenseTensor {                                         \
    if (x.dtype() == phi::DataType::BFLOAT16) {                  \
      return phi::EmptyLike<phi::bfloat16, Context>(dev_ctx, x); \
    } else if (x.dtype() == phi::DataType::FLOAT32) {            \
      return phi::EmptyLike<float, Context>(dev_ctx, x);         \
    } else if (x.dtype() == phi::DataType::FLOAT16) {            \
      return phi::EmptyLike<phi::float16, Context>(dev_ctx, x);  \
    } else {                                                     \
      PD_THROW("Unsupported data type for EMPTY_LIKE macro.");   \
    }                                                            \
  }()

namespace layer_norm {

template <typename Params>
struct LaunchParams {
  size_t workspace_bytes;
  size_t barrier_size;

  cudaDeviceProp* props;

  cudaStream_t stream;

  Params params;
};

struct ParamsBase {
  ParamsBase()
      : ctas_per_col(0),
        rows(0),
        cols(0),
        x(nullptr),
        mean(nullptr),
        invvar(nullptr),
        scale(nullptr),
        workspace(nullptr),
        barrier(nullptr) {}

  // For Multi-CTA, number of different CTA groups. Otherwise same as gridDim.x.
  int ctas_per_col;

  // Input is interpreted as matrix. We normalize across columns.
  int rows;
  int cols;

  // Common data pointers.
  void* x;
  void* mean;
  void* invvar;
  void* scale;

  // Multi-CTA workspace in gmem.
  void* workspace;

  // Multi-CTA sync barriers in gmem.
  int* barrier;
};

struct FwdParams : public ParamsBase {
  FwdParams() : ParamsBase(), y(nullptr), bias(nullptr), epsilon(0.f) {}

  // Output of LN FWD.
  void* y;
  void* bias;
  float epsilon;
};

struct BwdParams : public ParamsBase {
  BwdParams()
      : ParamsBase(),
        dy(nullptr),
        dbias_part(nullptr),
        dscale_part(nullptr),
        dx(nullptr),
        dbias(nullptr),
        dscale(nullptr) {}

  // Input: gradient wrt. LN FWD output.
  void* dy;

  // Workspace for Wgrad pre-reduction.
  void* dbias_part;
  void* dscale_part;

  // Output: Dgrad.
  void* dx;
  // Output: Wgrad.
  void* dbias;
  void* dscale;
};

using FwdFunction = std::function<void(LaunchParams<FwdParams>&, const bool)>;
using BwdFunction = std::function<void(LaunchParams<BwdParams>&, const bool)>;
using FunctionKey = uint64_t;
using FwdRegistry = std::unordered_map<FunctionKey, FwdFunction>;
using BwdRegistry = std::unordered_map<FunctionKey, BwdFunction>;

extern FwdRegistry FWD_FUNCS;
extern BwdRegistry BWD_FUNCS;

using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;

template <typename T>
struct TypeToIdTrait {};

template <>
struct TypeToIdTrait<fp16> {
  constexpr static uint32_t Value = 0;
};

template <>
struct TypeToIdTrait<bf16> {
  constexpr static uint32_t Value = 1;
};

template <>
struct TypeToIdTrait<fp32> {
  constexpr static uint32_t Value = 2;
};

template <typename T, int Significant>
struct Type2KeyTrait {
  constexpr static uint32_t Value = TypeToIdTrait<T>::Value << Significant;
};

template <typename T>
struct WeightType2KeyTrait : public Type2KeyTrait<T, 0> {};

template <typename T>
struct InputType2KeyTrait : public Type2KeyTrait<T, 2> {};

template <typename T>
struct OutputType2KeyTrait : public Type2KeyTrait<T, 4> {};

template <typename T>
struct ComputeType2KeyTrait : public Type2KeyTrait<T, 6> {};

template <typename WeightT,
          typename InputT,
          typename OutputT,
          typename ComputeT>
struct Types2KeyTrait {
  constexpr static uint32_t Value = WeightType2KeyTrait<WeightT>::Value |
                                    InputType2KeyTrait<InputT>::Value |
                                    OutputType2KeyTrait<OutputT>::Value |
                                    ComputeType2KeyTrait<ComputeT>::Value;
  constexpr static inline uint64_t get(const uint64_t hidden_size) {
    constexpr uint64_t type_key = Value;
    return (type_key << 32) | hidden_size;
  }
};

template <typename WeightT,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          uint64_t HIDDEN_SIZE>
struct FwdRegistrar {
  FwdRegistrar(FwdFunction f) {  // NOLINT
    uint64_t key =
        Types2KeyTrait<WeightT, InputT, OutputT, ComputeT>::get(HIDDEN_SIZE);
    FWD_FUNCS.insert({key, f});
  }
};

template <typename WeightT,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          uint64_t HIDDEN_SIZE>
struct BwdRegistrar {
  BwdRegistrar(BwdFunction f) {  // NOLINT
    uint64_t key =
        Types2KeyTrait<WeightT, InputT, OutputT, ComputeT>::get(HIDDEN_SIZE);
    BWD_FUNCS.insert({key, f});
  }
};

// Create registries and provide runtime versions of config hash functions.

uint32_t get_type_id(paddle::DataType dtype);

uint64_t get_key(paddle::DataType weight_type,
                 paddle::DataType input_type,
                 paddle::DataType output_type,
                 paddle::DataType compute_type,
                 uint64_t hidden_size);

}  // namespace layer_norm

namespace phi {
layer_norm::FwdFunction& get_fwd_launcher(paddle::DataType weight_type,
                                          paddle::DataType input_type,
                                          paddle::DataType output_type,
                                          paddle::DataType compute_type,
                                          uint32_t hidden_size);

layer_norm::BwdFunction& get_bwd_launcher(paddle::DataType weight_type,
                                          paddle::DataType input_type,
                                          paddle::DataType output_type,
                                          paddle::DataType compute_type,
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

}  // namespace phi
