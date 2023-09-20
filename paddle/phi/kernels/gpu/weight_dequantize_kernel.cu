/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "paddle/phi/kernels/weight_dequantize_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/matmul_kernel.h"

#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/kernels/funcs/weight_dequant_functor.h"
#endif

namespace phi {

template <typename T, typename Context>
void WeightDequantizeKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& scale,
                            DataType out_dtype,
                            const std::string& algo,
                            const bool transpose,
                            DenseTensor* out) {
#if defined(PADDLE_WITH_CUTLASS)
  dev_ctx.template Alloc<T>(out);
  WeightDequantize<T, Context>(dev_ctx, x, scale, algo, true, out);
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_dequantize,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightDequantizeKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
