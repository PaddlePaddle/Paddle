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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied from NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include <cstdio>
#include <unordered_map>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {
namespace funcs {
namespace fast_ln_v2 {

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

extern FwdRegistry FAST_LN_V2_FWD_FUNCS;
extern BwdRegistry FAST_LN_V2_BWD_FUNCS;

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
    FAST_LN_V2_FWD_FUNCS.insert({key, f});
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
    FAST_LN_V2_BWD_FUNCS.insert({key, f});
  }
};

// =========================================================================
// Helper functions from ln.cu (inline)
// =========================================================================

inline uint32_t get_type_id(phi::DataType dtype) {
  if (dtype == phi::DataType::FLOAT16) {
    return TypeToIdTrait<fp16>::Value;  // FLOAT16 <--> 0
  } else if (dtype == phi::DataType::BFLOAT16) {
    return TypeToIdTrait<bf16>::Value;  // BFLOAT16 <--> 1
  } else if (dtype == phi::DataType::FLOAT32) {
    return TypeToIdTrait<float>::Value;  // FLOAT32 <--> 2
  } else {
    return 3;  // Others <--> 3
  }
}

inline uint64_t get_key(phi::DataType weight_type,
                        phi::DataType input_type,
                        phi::DataType output_type,
                        phi::DataType compute_type,
                        uint64_t hidden_size) {
  uint64_t type_key =
      get_type_id(weight_type) | (get_type_id(input_type) << 2) |  // NOLINT
      (get_type_id(output_type) << 4) | (get_type_id(compute_type) << 6);
  uint64_t launcher_key = (type_key << 32) | hidden_size;
  return launcher_key;
}

}  // namespace fast_ln_v2
}  // namespace funcs
}  // namespace phi
