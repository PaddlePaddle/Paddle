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
#include "paddle/phi/kernels/bitwise_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/bitwise_functors.h"
#include "paddle/phi/kernels/stride/elementwise_stride_base.cu.h"
#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"
#endif
COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);
namespace phi {
#define DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(name)                        \
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
    LaunchBinaryElementwiseStrideKernel<T, Context>(                          \
        dev_ctx, x_, y_, funcs::name##Functor<T>(), -1, out);                 \
  }
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(BitwiseAnd)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(BitwiseOr)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(BitwiseXor)
}  // namespace phi
using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;
PD_REGISTER_KERNEL(bitwise_and,
                   GPU,
                   STRIDED,
                   phi::BitwiseAndStrideKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(bitwise_or,
                   GPU,
                   STRIDED,
                   phi::BitwiseOrStrideKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(bitwise_xor,
                   GPU,
                   STRIDED,
                   phi::BitwiseXorStrideKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
#endif
