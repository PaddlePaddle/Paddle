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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/phi/kernels/compare_kernel.h"
#include <limits>
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/indexing.h"
#include "paddle/phi/kernels/stride/elementwise_stride_base.cu.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename T, typename Context, typename Functor>
void LaunchCompareStrideKernel(const Context &dev_ctx,
                               const DenseTensor &x,
                               const DenseTensor &y,
                               Functor func,
                               int axis,
                               DenseTensor *out) {
  dev_ctx.template Alloc<bool>(out);
  out->set_type(phi::DataType::BOOL);
  if (out->numel() == 0) return;
  std::vector<const DenseTensor *> inputs = {&x, &y};
  std::vector<DenseTensor *> outputs = {out};
  BinaryStrideBroadcastKernel<bool, Context>(
      dev_ctx, inputs, &outputs, Functor(), axis);
}

#define DEFINE_CUDA_COMPARE_STRIDE_OP(name, functor_name)                     \
  template <typename T, typename Context>                                     \
  void name##StrideKernel(const Context &dev_ctx,                             \
                          const DenseTensor &x,                               \
                          const DenseTensor &y,                               \
                          DenseTensor *out) {                                 \
    if (!FLAGS_use_stride_kernel) {                                           \
      PADDLE_THROW(common::errors::Fatal(                                     \
          "FLAGS_use_stride_kernel is closed. Strided kernel "                \
          "be called, something wrong has happened!"));                       \
    }                                                                         \
    DenseTensor x_;                                                           \
    DenseTensor y_;                                                           \
    if (!FLAGS_use_stride_compute_kernel || x.offset() != 0 ||                \
        y.offset() != 0) {                                                    \
      if (!x.meta().is_contiguous() || x.offset() != 0) {                     \
        x_ = Tensor2Contiguous<Context>(dev_ctx, x);                          \
      } else {                                                                \
        x_ = x;                                                               \
      }                                                                       \
      if (!y.meta().is_contiguous() || y.offset() != 0) {                     \
        y_ = Tensor2Contiguous<Context>(dev_ctx, y);                          \
      } else {                                                                \
        y_ = y;                                                               \
      }                                                                       \
    } else {                                                                  \
      x_ = x;                                                                 \
      y_ = y;                                                                 \
    }                                                                         \
    if (x_.meta().is_contiguous() && y_.meta().is_contiguous()) {             \
      auto meta = out->meta();                                                \
      meta.strides = meta.calc_strides(out->dims());                          \
      out->set_meta(meta);                                                    \
      phi::name##Kernel<T, Context>(dev_ctx, x_, y_, out);                    \
      return;                                                                 \
    }                                                                         \
    if (!FLAGS_use_stride_compute_kernel) {                                   \
      PADDLE_THROW(                                                           \
          common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. " \
                                "Kernel using DenseTensorIterator "           \
                                "be called, something wrong has happened!")); \
    }                                                                         \
                                                                              \
    if (out->IsSharedWith(x_)) {                                              \
      auto x_origin = x_;                                                     \
      LaunchCompareStrideKernel<T, Context>(                                  \
          dev_ctx, x_origin, y_, funcs::functor_name##Functor<T>(), -1, out); \
    } else {                                                                  \
      LaunchCompareStrideKernel<T, Context>(                                  \
          dev_ctx, x_, y_, funcs::functor_name##Functor<T>(), -1, out);       \
    }                                                                         \
  }

DEFINE_CUDA_COMPARE_STRIDE_OP(LessThan, LessThan)
DEFINE_CUDA_COMPARE_STRIDE_OP(LessEqual, LessEqual)
DEFINE_CUDA_COMPARE_STRIDE_OP(GreaterThan, GreaterThan)
DEFINE_CUDA_COMPARE_STRIDE_OP(GreaterEqual, GreaterEqual)
DEFINE_CUDA_COMPARE_STRIDE_OP(Equal, Equal)
DEFINE_CUDA_COMPARE_STRIDE_OP(NotEqual, NotEqual)

#undef DEFINE_CUDA_COMPARE_STRIDE_OP

}  // namespace phi

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

#define REGISTER_STRIDE_COMPLEX_COMPARE_KERNEL(less_than, func) \
  PD_REGISTER_KERNEL(less_than,                                 \
                     GPU,                                       \
                     STRIDED,                                   \
                     phi::func##Kernel,                         \
                     bool,                                      \
                     int,                                       \
                     uint8_t,                                   \
                     int8_t,                                    \
                     int16_t,                                   \
                     int64_t,                                   \
                     phi::dtype::complex<float>,                \
                     phi::dtype::complex<double>,               \
                     float,                                     \
                     double,                                    \
                     phi::dtype::float16,                       \
                     phi::dtype::bfloat16) {                    \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);       \
  }

REGISTER_STRIDE_COMPLEX_COMPARE_KERNEL(less_than, LessThanStride)
REGISTER_STRIDE_COMPLEX_COMPARE_KERNEL(less_equal, LessEqualStride)
REGISTER_STRIDE_COMPLEX_COMPARE_KERNEL(greater_than, GreaterThanStride)
REGISTER_STRIDE_COMPLEX_COMPARE_KERNEL(greater_equal, GreaterEqualStride)
REGISTER_STRIDE_COMPLEX_COMPARE_KERNEL(equal, EqualStride)
REGISTER_STRIDE_COMPLEX_COMPARE_KERNEL(not_equal, NotEqualStride)

#undef REGISTER_STRIDE_COMPLEX_COMPARE_KERNEL

#endif
