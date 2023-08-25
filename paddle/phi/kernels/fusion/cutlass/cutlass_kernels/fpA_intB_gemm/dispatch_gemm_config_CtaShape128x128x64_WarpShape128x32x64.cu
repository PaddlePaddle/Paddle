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

template <typename T, typename WeightType, typename arch, typename EpilogueTag>
void dispatch_gemm_config_CtaShape128x128x64_WarpShape128x32x64(
    const T* A,
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
      using DispatcherStages2 =
          dispatch_stages<T,
                          WeightType,
                          arch,
                          EpilogueTag,
                          cutlass::gemm::GemmShape<128, 128, 64>,
                          cutlass::gemm::GemmShape<128, 32, 64>,
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
      using DispatcherStages3 =
          dispatch_stages<T,
                          WeightType,
                          arch,
                          EpilogueTag,
                          cutlass::gemm::GemmShape<128, 128, 64>,
                          cutlass::gemm::GemmShape<128, 32, 64>,
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
      using DispatcherStages4 =
          dispatch_stages<T,
                          WeightType,
                          arch,
                          EpilogueTag,
                          cutlass::gemm::GemmShape<128, 128, 64>,
                          cutlass::gemm::GemmShape<128, 32, 64>,
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

}  // namespace phi
