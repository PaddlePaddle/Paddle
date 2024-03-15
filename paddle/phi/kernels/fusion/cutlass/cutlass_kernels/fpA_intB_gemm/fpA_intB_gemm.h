/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/ft_gemm_configs.h"
// #include "src/fastertransformer/utils/allocator.h"
#include "cuda_runtime_api.h"  // NOLINT

namespace phi {

/*
  This runner only supports:
  T in {half, __nv_bfloat} WeightType in {uint8_t, cutlass::uint4b_t}

  Activations, biases, scales and outputs are all assumed to be row-major.

  However, it is assumed that B is in a special format governed by
  cutlass_extensions/gemm/kernel/mixed_gemm_B_layout. In this case, B must be
  preprocessed using the cutlass weight only quant preprocessors. The weight
  preprocessor will instantiate the layout and preprocess based on the
  instantiation, so layout changes should only require modifications to
  mix_gemm_B_layout.h.
*/

template <typename T, typename WeightType>
class CutlassFpAIntBGemmRunner {
 public:
  CutlassFpAIntBGemmRunner();
  ~CutlassFpAIntBGemmRunner();

  void gemm(const T* A,
            const WeightType* B,
            const T* weight_scales,
            T* C,
            int m,
            int n,
            int k,
            int group_size,
            char* workspace_ptr,
            const size_t workspace_bytes,
            cudaStream_t stream);

  void gemm_bias_act(const T* A,
                     const WeightType* B,
                     const T* weight_scales,
                     const T* biases,
                     T* C,
                     int m,
                     int n,
                     int k,
                     int group_size,
                     std::string activation_type,
                     char* workspace_ptr,
                     const size_t workspace_bytes,
                     cudaStream_t stream);

  // Returns desired workspace size in bytes.
  int getWorkspaceSize(const int m, const int n, const int k);

 private:
  template <typename EpilogueTag, bool FineGrained>
  void dispatch_to_arch(const T* A,
                        const WeightType* B,
                        const T* weight_scales,
                        const T* biases,
                        T* C,
                        int m,
                        int n,
                        int k,
                        int group_size,
                        CutlassGemmConfig gemm_config,
                        char* workspace_ptr,
                        const size_t workspace_bytes,
                        cudaStream_t stream,
                        int* occupancy = nullptr);

  template <typename EpilogueTag, bool FineGrained>
  void run_gemm(const T* A,
                const WeightType* B,
                const T* weight_scales,
                const T* biases,
                T* C,
                int m,
                int n,
                int k,
                int group_size,
                char* workspace_ptr,
                const size_t workspace_bytes,
                cudaStream_t stream);

 private:
  static constexpr int split_k_limit = 7;

  int sm_;
  int multi_processor_count_;
};

// This allocation is present to help with compiling with other structures in
// FT. It will throw an error in all functions because this runner assumes the
// weight type and the activation type are different. We allow empty classes to
// be created, but any calls to gemm or gemm_bias_act will throw an error.
template <typename WeightType>
class CutlassFpAIntBGemmRunner<float, WeightType> {
 public:
  CutlassFpAIntBGemmRunner() = default;
  ~CutlassFpAIntBGemmRunner() = default;

  void gemm(const float* A,
            const WeightType* B,
            const float* weight_scales,
            float* C,
            int m,
            int n,
            int k,
            int group_size,
            char* workspace_ptr,
            const size_t workspace_bytes,
            cudaStream_t stream);

  void gemm_bias_act(const float* A,
                     const WeightType* B,
                     const float* weight_scales,
                     const float* biases,
                     float* C,
                     int m,
                     int n,
                     int k,
                     int group_size,
                     std::string activation_type,
                     char* workspace_ptr,
                     const size_t workspace_bytes,
                     cudaStream_t stream);

  int getWorkspaceSize(const int m, const int n, const int k);
};
}  // namespace phi
