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

#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic pop

namespace phi {

template <typename T,
          typename WeightType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape>
void dispatch_gemm_config(const T* A,
                          const WeightType* B,
                          const float* weight_scales,
                          const T* biases,
                          T* C,
                          int m,
                          int n,
                          int k,
                          CutlassGemmConfig gemm_config,
                          char* workspace,
                          size_t workspace_bytes,
                          cudaStream_t stream,
                          int* occupancy) {
  switch (gemm_config.stages) {
    case 2:
      using DispatcherStages2 = dispatch_stages<T,
                                                WeightType,
                                                arch,
                                                EpilogueTag,
                                                ThreadblockShape,
                                                WarpShape,
                                                2>;
      DispatcherStages2::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  m,
                                  n,
                                  k,
                                  gemm_config,
                                  workspace,
                                  workspace_bytes,
                                  stream,
                                  occupancy);
      break;
    case 3:
      using DispatcherStages3 = dispatch_stages<T,
                                                WeightType,
                                                arch,
                                                EpilogueTag,
                                                ThreadblockShape,
                                                WarpShape,
                                                3>;
      DispatcherStages3::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  m,
                                  n,
                                  k,
                                  gemm_config,
                                  workspace,
                                  workspace_bytes,
                                  stream,
                                  occupancy);
      break;
    case 4:
      using DispatcherStages4 = dispatch_stages<T,
                                                WeightType,
                                                arch,
                                                EpilogueTag,
                                                ThreadblockShape,
                                                WarpShape,
                                                4>;
      DispatcherStages4::dispatch(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  m,
                                  n,
                                  k,
                                  gemm_config,
                                  workspace,
                                  workspace_bytes,
                                  stream,
                                  occupancy);
      break;
    default:
      std::string err_msg = "dispatch_gemm_config does not support stages " +
                            std::to_string(gemm_config.stages);
      throw std::runtime_error("[dispatch_gemm_config] " + err_msg);
      break;
  }
}

template <typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_to_cutlass(const T* A,
                              const WeightType* B,
                              const float* weight_scales,
                              const T* biases,
                              T* C,
                              int m,
                              int n,
                              int k,
                              char* workspace,
                              size_t workspace_bytes,
                              CutlassGemmConfig gemm_config,
                              cudaStream_t stream,
                              int* occupancy) {
  // VLOG(3)<<__PRETTY_FUNCTION__;
  // Note that SIMT configs are omitted here since they are not supported for
  // fpA_intB. We also only instantiate configs here where threadblockShapeM ==
  // warpShapeM since those usually perform the best for mixed type gemms.
  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<32, 128, 64>,
                           cutlass::gemm::GemmShape<32, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          m,
          n,
          k,
          gemm_config,
          workspace,
          workspace_bytes,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<64, 128, 64>,
                           cutlass::gemm::GemmShape<64, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          m,
          n,
          k,
          gemm_config,
          workspace,
          workspace_bytes,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<128, 128, 64>,
                           cutlass::gemm::GemmShape<128, 32, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          m,
          n,
          k,
          gemm_config,
          workspace,
          workspace_bytes,
          stream,
          occupancy);
      break;
    // config for M_16000_N_12288_K_6144 in encoder
    case CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<256, 128, 64>,
                           cutlass::gemm::GemmShape<64, 64, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          m,
          n,
          k,
          gemm_config,
          workspace,
          workspace_bytes,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
      dispatch_gemm_config<T,
                           WeightType,
                           arch,
                           EpilogueTag,
                           cutlass::gemm::GemmShape<128, 256, 64>,
                           cutlass::gemm::GemmShape<64, 64, 64>>(
          A,
          B,
          weight_scales,
          biases,
          C,
          m,
          n,
          k,
          gemm_config,
          workspace,
          workspace_bytes,
          stream,
          occupancy);
      break;
    case CutlassTileConfig::Undefined:
      throw std::runtime_error(
          "[fpA_intB][dispatch_gemm_to_cutlass] gemm config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      throw std::runtime_error(
          "[fpA_intB][dispatch_gemm_to_cutlass] gemm config should have "
          "already been set by heuristic.");
      break;
    default:
      throw std::runtime_error(
          "[fpA_intB][dispatch_gemm_to_cutlass] Config is invalid for mixed "
          "type GEMM.");
      break;
  }
}

template <typename T, typename WeightType>
CutlassFpAIntBGemmRunner<T, WeightType>::CutlassFpAIntBGemmRunner() {
  // VLOG(3)<<__PRETTY_FUNCTION__;
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  sm_ = getSMVersion();
  check_cuda_error(cudaDeviceGetAttribute(
      &multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, typename WeightType>
CutlassFpAIntBGemmRunner<T, WeightType>::~CutlassFpAIntBGemmRunner() {
  // VLOG(3)<<__PRETTY_FUNCTION__;
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void CutlassFpAIntBGemmRunner<T, WeightType>::dispatch_to_arch<EpilogueTag>(
    const T* A,
    const WeightType* B,
    const float* weight_scales,
    const T* biases,
    T* C,
    int m,
    int n,
    int k,
    CutlassGemmConfig gemm_config,
    char* workspace_ptr,
    const size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy) {
  // VLOG(3)<<__PRETTY_FUNCTION__;
  if (sm_ >= 70 && sm_ < 75) {
#if defined(USE_FPAINTB_GEMM_WITH_SM70)
    dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm70, EpilogueTag>(
        A,
        B,
        weight_scales,
        biases,
        C,
        m,
        n,
        k,
        workspace_ptr,
        workspace_bytes,
        gemm_config,
        stream,
        occupancy);
#else
    throw std::runtime_error(
        "[CutlassFpAIntBGemmRunner][GEMM Dispatch] Arch unsupported for "
        "CUTLASS mixed type GEMM");
#endif
  } else if (sm_ >= 75 && sm_ < 80) {
#if defined(USE_FPAINTB_GEMM_WITH_SM75)
    dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm75, EpilogueTag>(
        A,
        B,
        weight_scales,
        biases,
        C,
        m,
        n,
        k,
        workspace_ptr,
        workspace_bytes,
        gemm_config,
        stream,
        occupancy);
#else
    throw std::runtime_error(
        "[CutlassFpAIntBGemmRunner][GEMM Dispatch] Arch unsupported for "
        "CUTLASS mixed type GEMM");
#endif
  } else if (sm_ >= 80 && sm_ < 90) {
#if defined(USE_FPAINTB_GEMM_WITH_SM80)
    dispatch_gemm_to_cutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(
        A,
        B,
        weight_scales,
        biases,
        C,
        m,
        n,
        k,
        workspace_ptr,
        workspace_bytes,
        gemm_config,
        stream,
        occupancy);
#else
    throw std::runtime_error(
        "[CutlassFpAIntBGemmRunner][GEMM Dispatch] Arch unsupported for "
        "CUTLASS mixed type GEMM");
#endif
  } else {
    throw std::runtime_error(
        "[CutlassFpAIntBGemmRunner][GEMM Dispatch] Arch unsupported for "
        "CUTLASS mixed type GEMM");
  }
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void CutlassFpAIntBGemmRunner<T, WeightType>::run_gemm<EpilogueTag>(
    const T* A,
    const WeightType* B,
    const float* weight_scales,
    const T* biases,
    T* C,
    int m,
    int n,
    int k,
    char* workspace_ptr,
    const size_t workspace_bytes,
    cudaStream_t stream) {
  // VLOG(3)<<__PRETTY_FUNCTION__;
  static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
  const bool is_weight_only_encoder = m >= 512 ? true : false;
  std::vector<CutlassGemmConfig> candidate_configs =
      get_candidate_configs(sm_, is_weight_only, is_weight_only_encoder, false);
  std::vector<int> occupancies(candidate_configs.size());

  for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
    dispatch_to_arch<EpilogueTag>(A,
                                  B,
                                  weight_scales,
                                  biases,
                                  C,
                                  m,
                                  n,
                                  k,
                                  candidate_configs[ii],
                                  workspace_ptr,
                                  workspace_bytes,
                                  stream,
                                  &occupancies[ii]);
  }
  // Standard GEMM, so 1 "expert". We use the same function for MoE and regular
  // FFN.
  static constexpr int num_experts = 1;
  CutlassGemmConfig chosen_config =
      estimate_best_config_from_occupancies(candidate_configs,
                                            occupancies,
                                            m,
                                            n,
                                            k,
                                            num_experts,
                                            split_k_limit,
                                            workspace_bytes,
                                            multi_processor_count_,
                                            is_weight_only);

  dispatch_to_arch<EpilogueTag>(A,
                                B,
                                weight_scales,
                                biases,
                                C,
                                m,
                                n,
                                k,
                                chosen_config,
                                workspace_ptr,
                                workspace_bytes,
                                stream);
}

template <typename T, typename WeightType>
void CutlassFpAIntBGemmRunner<T, WeightType>::gemm_bias_act(
    const T* A,
    const WeightType* B,
    const float* weight_scales,
    const T* biases,
    T* C,
    int m,
    int n,
    int k,
    std::string activation_type,
    char* workspace_ptr,
    const size_t workspace_bytes,
    cudaStream_t stream) {
  // VLOG(3)<<__PRETTY_FUNCTION__;
  if (activation_type == "gelu") {
    run_gemm<EpilogueOpBiasFtGelu>(A,
                                   B,
                                   weight_scales,
                                   biases,
                                   C,
                                   m,
                                   n,
                                   k,
                                   workspace_ptr,
                                   workspace_bytes,
                                   stream);
  } else if (activation_type == "relu") {
    run_gemm<EpilogueOpBiasReLU>(A,
                                 B,
                                 weight_scales,
                                 biases,
                                 C,
                                 m,
                                 n,
                                 k,
                                 workspace_ptr,
                                 workspace_bytes,
                                 stream);
  } else if (activation_type == "none") {
    run_gemm<EpilogueOpBias>(A,
                             B,
                             weight_scales,
                             biases,
                             C,
                             m,
                             n,
                             k,
                             workspace_ptr,
                             workspace_bytes,
                             stream);
  } else {
    throw std::runtime_error(("Invalid activation type."));
  }
}

template <typename T, typename WeightType>
void CutlassFpAIntBGemmRunner<T, WeightType>::gemm(const T* A,
                                                   const WeightType* B,
                                                   const float* weight_scales,
                                                   T* C,
                                                   int m,
                                                   int n,
                                                   int k,
                                                   char* workspace_ptr,
                                                   const size_t workspace_bytes,
                                                   cudaStream_t stream) {
  // VLOG(3)<<__PRETTY_FUNCTION__;
  run_gemm<EpilogueOpNoBias>(A,
                             B,
                             weight_scales,
                             nullptr,
                             C,
                             m,
                             n,
                             k,
                             workspace_ptr,
                             workspace_bytes,
                             stream);
}

template <typename T, typename WeightType>
int CutlassFpAIntBGemmRunner<T, WeightType>::getWorkspaceSize(const int m,
                                                              const int n,
                                                              const int k) {
  // VLOG(3)<<__PRETTY_FUNCTION__;    // These are the min tile sizes for each
  // config, which would launch the maximum number of blocks
  const int max_grid_m = (m + 31) / 32;
  const int max_grid_n = (n + 127) / 128;
  // We need 4 bytes per block in the worst case. We launch split_k_limit in z
  // dim.
  return max_grid_m * max_grid_n * split_k_limit * 4;
}

// =============================== Specialization T == WeightType
// =======================================
template <typename WeightType>
void CutlassFpAIntBGemmRunner<float, WeightType>::gemm_bias_act(
    const float* A,
    const WeightType* B,
    const float* weight_scales,
    const float* biases,
    float* C,
    int m,
    int n,
    int k,
    std::string activation_type,
    char* workspace_ptr,
    const size_t workspace_bytes,
    cudaStream_t stream) {
  throw std::runtime_error(
      ("Attempting to run mixed gemm bias act when the types are the same is "
       "an error."));
}

template <typename WeightType>
void CutlassFpAIntBGemmRunner<float, WeightType>::gemm(
    const float* A,
    const WeightType* B,
    const float* weight_scales,
    float* C,
    int m,
    int n,
    int k,
    char* workspace_ptr,
    const size_t workspace_bytes,
    cudaStream_t stream) {
  throw std::runtime_error((
      "Attempting to run mixed gemm when the types are the same is an error."));
}

template <typename WeightType>
int CutlassFpAIntBGemmRunner<float, WeightType>::getWorkspaceSize(const int m,
                                                                  const int n,
                                                                  const int k) {
  return 0;
}

template class CutlassFpAIntBGemmRunner<float, uint8_t>;
template class CutlassFpAIntBGemmRunner<half, uint8_t>;
#ifdef PADDLE_CUDA_BF16
template class CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t>;
#endif
template class CutlassFpAIntBGemmRunner<float, cutlass::uint4b_t>;
template class CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t>;
#ifdef PADDLE_CUDA_BF16
template class CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif
}  // namespace phi
