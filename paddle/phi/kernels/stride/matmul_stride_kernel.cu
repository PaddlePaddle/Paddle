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

#include <limits>
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename Context>
phi::DenseTensor Tensor2Contiguous(const Context &dev_ctx,
                                   const phi::DenseTensor &tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  PD_VISIT_ALL_TYPES(tensor.dtype(), "Tensor2Contiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

template <typename T, typename Context>
void MatmulStrideKernel(const Context &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        bool transpose_x,
                        bool transpose_y,
                        DenseTensor *out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  DenseTensor x_;
  DenseTensor y_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
    if (!y.meta().is_contiguous()) {
      y_ = Tensor2Contiguous<Context>(dev_ctx, y);
    } else {
      y_ = y;
    }
  } else {
    x_ = x;
    y_ = y;
  }
  if (x_.meta().is_contiguous() && y_.meta().is_contiguous()) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::MatmulKernel<T, Context>(
        dev_ctx, x_, y_, transpose_x, transpose_y, out);
    return;
  }
  if (!FLAGS_use_stride_compute_kernel) {
    PADDLE_THROW(
        common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
                              "Kernel using DenseTensorIterator "
                              "be called, something wrong has happened!"));
  }
}

}  // namespace phi

#if CUDA_VERSION >= 12010 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
PD_REGISTER_KERNEL(matmul,
                   GPU,
                   STRIDED,
                   phi::MatmulStrideKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::float8_e4m3fn,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128,
                   int8_t) {
#else
PD_REGISTER_KERNEL(matmul,
                   GPU,
                   STRIDED,
                   phi::MatmulStrideKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128,
                   int8_t) {
#endif
  if (kernel_key.dtype() == phi::DataType::INT8) {
    kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  }
  if (kernel_key.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT16);
  }
}

#endif
