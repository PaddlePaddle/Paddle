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

#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_gemm_kernels.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_gemm_kernels_template.h"

namespace phi {

template class MoeGemmRunner<half, half>;
template class MoeGemmRunner<half, uint8_t>;
template class MoeGemmRunner<half, cutlass::uint4b_t>;

#if CUDA_VERSION >= 11000
template class MoeGemmRunner<__nv_bfloat16, __nv_bfloat16>;
template class MoeGemmRunner<__nv_bfloat16, uint8_t>;
template class MoeGemmRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class MoeGemmRunner<float, float>;

}  // namespace phi
