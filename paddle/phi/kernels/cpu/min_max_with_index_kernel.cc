// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/min_max_with_index_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define DEFINE_WITH_INDEX_KERNEL(OpType, name)                          \
  template <typename T, typename Context>                               \
  void OpType##WithIndexKernel(const Context& dev_ctx,                  \
                               const DenseTensor& x,                    \
                               const Scalar& dim,                       \
                               bool keepdim,                            \
                               bool flatten,                            \
                               DenseTensor* val_out,                    \
                               DenseTensor* ind_out) {                  \
    PADDLE_ENFORCE_EQ(0,                                                \
                      1,                                                \
                      phi::errors::Unimplemented(                       \
                          "In static graph mode, %s PHI kernel is not " \
                          "currently available on non-GPU devices.",    \
                          #name));                                      \
  }                                                                     \
  template <typename T, typename Context>                               \
  void OpType##WithIndexGradKernel(const Context& dev_ctx,              \
                                   const DenseTensor& x,                \
                                   const DenseTensor& values,           \
                                   const DenseTensor& indices,          \
                                   const DenseTensor& values_grad,      \
                                   const Scalar& dim,                   \
                                   bool keepdim,                        \
                                   DenseTensor* x_grad) {               \
    PADDLE_ENFORCE_EQ(0,                                                \
                      1,                                                \
                      phi::errors::Unimplemented(                       \
                          "In static graph mode, %s PHI kernel is not " \
                          "currently available on non-GPU devices.",    \
                          #name));                                      \
  }

namespace phi {

DEFINE_WITH_INDEX_KERNEL(Min, min_with_index)
DEFINE_WITH_INDEX_KERNEL(Max, max_with_index)
#undef DEFINE_WITH_INDEX_KERNEL

}  // namespace phi

#define REGISTER_CPU_KERNELS(OpType, OpName)                   \
  PD_REGISTER_KERNEL(OpName,                                   \
                     CPU,                                      \
                     ALL_LAYOUT,                               \
                     phi::OpType##WithIndexKernel,             \
                     phi::dtype::float16,                      \
                     phi::dtype::bfloat16,                     \
                     float,                                    \
                     double,                                   \
                     int32_t,                                  \
                     int64_t,                                  \
                     int16_t,                                  \
                     uint8_t) {                                \
    kernel->OutputAt(0).SetDataType(kernel->InputAt(0).dtype); \
    kernel->OutputAt(1).SetDataType(phi::DataType::INT64);     \
  }                                                            \
  PD_REGISTER_KERNEL(OpName##_grad,                            \
                     CPU,                                      \
                     ALL_LAYOUT,                               \
                     phi::OpType##WithIndexGradKernel,         \
                     float,                                    \
                     double,                                   \
                     uint8_t,                                  \
                     int,                                      \
                     int16_t,                                  \
                     int64_t,                                  \
                     phi::dtype::float16,                      \
                     phi::dtype::bfloat16) {}

REGISTER_CPU_KERNELS(Min, min_with_index)
REGISTER_CPU_KERNELS(Max, max_with_index)
#undef REGISTER_CPU_KERNELS
