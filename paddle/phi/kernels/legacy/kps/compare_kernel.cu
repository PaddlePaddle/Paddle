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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/legacy/impl/compare_kernel_impl.h"

#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/backends/xpu/xpu_context.h"
#else
#include <thrust/fill.h>

#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#endif

namespace phi {

template <typename T>
struct BitwiseAdd {
  // Bitwise add operator, returns <tt>a + b</tt>
  inline T initial() { return static_cast<T>(true); }

  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return a & b;
  }
};

template <typename T,
          typename Context,
          typename Functor,
          typename InverseFunctor>
inline void CompareRawKernelImpl(const Context& ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              int axis,
                              DenseTensor* out) {
  ctx.template Alloc<bool>(out);
  std::vector<const DenseTensor*> ins{&x, &y};
  std::vector<DenseTensor*> outs{out};
  funcs::BroadcastKernel<bool>(ctx, ins, &outs, Functor(), axis);
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(
    less_than_raw, KPS, ALL_LAYOUT, phi::LessThanRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(
    less_equal_raw, KPS, ALL_LAYOUT, phi::LessEqualRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(
    greater_than_raw, KPS, ALL_LAYOUT, phi::GreaterThanRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(
    greater_equal_raw, KPS, ALL_LAYOUT, phi::GreaterEqualRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(equal_raw, KPS, ALL_LAYOUT, phi::EqualRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(
    not_equal_raw, KPS, ALL_LAYOUT, phi::NotEqualRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#else

#define PD_REGISTER_COMPARE_RAW_KERNEL(name, func)            \                                                   \
  PD_REGISTER_KERNEL(name##_raw,                          \
                     KPS,                                 \
                     ALL_LAYOUT,                          \
                     phi::func##RawKernel,                \
                     bool,                                \
                     int16_t,                             \
                     int,                                 \
                     int64_t,                             \
                     float,                               \
                     double,                              \
                     phi::dtype::float16,                 \
                     phi::dtype::bfloat16) {              \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

PD_REGISTER_COMPARE_RAW_KERNEL(less_than, LessThan)
PD_REGISTER_COMPARE_RAW_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_RAW_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_RAW_KERNEL(greater_equal, GreaterEqual)
PD_REGISTER_COMPARE_RAW_KERNEL(equal, Equal)
PD_REGISTER_COMPARE_RAW_KERNEL(not_equal, NotEqual)

#endif
