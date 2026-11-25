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

#include "paddle/phi/kernels/concat_grad_kernel.h"
#include <algorithm>
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
#include "paddle/phi/kernels/funcs/strided_reshape_utils.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
#include "paddle/phi/kernels/slice_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {
template <typename Context>
void NarrowStrideKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const int64_t& dim,
                        const int64_t& begin,
                        const int64_t& length,
                        DenseTensor* out) {
  auto cur_size = dim;
  auto start = begin;
  if (start < 0) {
    start = start + cur_size;
  }

  SliceStridedKernel<Context>(dev_ctx,
                              x,
                              {dim},
                              IntArray({start}),
                              IntArray({start + length}),
                              {},
                              {},
                              out);
}

template <typename T, typename Context>
void ConcatGradStrideKernel(const Context& dev_ctx,
                            const std::vector<const DenseTensor*>& x,
                            const DenseTensor& out_grad,
                            const Scalar& axis_scalar,
                            std::vector<DenseTensor*> x_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  bool invalid_stride = false;

  for (int i = 0; i < x.size(); i++) {
    if (x[i]) {
      if (IsComplexType(x[i]->dtype())) {
        invalid_stride = true;
        break;
      }
      if (x[i]->numel() == 0) {
        invalid_stride = true;
        break;
      }
    }
  }

  if (!FLAGS_use_stride_compute_kernel || invalid_stride) {
    DenseTensor out_grad_;
    if (!out_grad.meta().is_contiguous()) {
      phi::MetaTensor meta_input(out_grad);
      phi::MetaTensor meta_out(&out_grad_);
      UnchangedInferMeta(meta_input, &meta_out);
      phi::ContiguousKernel<T, Context>(dev_ctx, out_grad, &out_grad_);
    } else {
      out_grad_ = out_grad;
    }

    for (int i = 0; i < x_grad.size(); i++) {
      if (x_grad[i]) {
        auto meta = x_grad[i]->meta();
        meta.strides = meta.calc_strides(x_grad[i]->dims());
        x_grad[i]->set_meta(meta);
      }
    }

    phi::ConcatGradKernel<T, Context>(
        dev_ctx, x, out_grad_, axis_scalar, x_grad);
    return;
  }

  auto outs = x_grad;
  {
    auto dx = x_grad;
    for (size_t i = 0; i < dx.size(); ++i) {
      if (dx[i] != nullptr) {
        dx[i]->set_lod(x[i]->lod());
      }
    }
  }
  PADDLE_ENFORCE_NOT_NULL(
      x[0],
      common::errors::NotFound("The first input tensor is not initialized."));

  auto axis = axis_scalar.to<int>();
  axis = funcs::ComputeAxis(static_cast<int64_t>(axis),
                            static_cast<int64_t>(x[0]->dims().size()));
  std::vector<DenseTensor*> outputs;
  if (out_grad.numel() == 0) {
    return;
  }

  int64_t accumulate = 0;
  for (int i = 0; i < x.size(); i++) {
    auto& shape = x[i]->dims();
    const auto& size = shape[axis];
    accumulate += size;
    if (outs[i]) {
      NarrowStrideKernel<Context>(
          dev_ctx, out_grad, axis, accumulate - size, size, outs[i]);
    }
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(concat_grad,
                   CPU,
                   STRIDED,
                   phi::ConcatGradStrideKernel,
                   double,
                   float,
                   bool,
                   int64_t,
                   int,
                   int8_t,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::complex64,
                   phi::complex128) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(concat_grad,
                   GPU,
                   STRIDED,
                   phi::ConcatGradStrideKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::complex64,
                   phi::complex128) {}
#endif

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_CUDA)
PD_REGISTER_KERNEL(concat_grad,
                   XPU,
                   STRIDED,
                   phi::ConcatGradStrideKernel,
                   float,
                   phi::float16,
                   phi::bfloat16) {}
#endif
