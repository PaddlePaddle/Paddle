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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/as_strided_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename Context>
phi::DenseTensor Tensor2Contiguous(const Context& dev_ctx,
                                   const phi::DenseTensor& tensor) {
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

template <typename Context>
phi::DenseTensor CheckMultipleUnsqueeze(const Context& dev_ctx,
                                        const DenseTensor& out_grad,
                                        const IntArray& dims,
                                        const int ndim,
                                        bool keep_dim) {
  phi::DenseTensor res = out_grad;
  if (dims.size() == 0 || keep_dim || ndim == 0) return res;
  std::vector<bool> axes(ndim, false);

  for (int i = 0; i < dims.size(); i++) {
    int tmp_dim = dims[i] >= 0 ? dims[i] : ndim + dims[i];
    axes[tmp_dim] = true;
  }

  for (int i = 0; i < axes.size(); i++) {
    phi::DenseTensor tmp;
    if (axes[i]) {
      UnsqueezeStridedKernel(dev_ctx, res, IntArray({i}), &tmp);
      res = tmp;
    }
  }

  return res;
}

template <typename T, typename Context>
void ReduceSumGradStrideKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& out_grad,
                               const IntArray& dims,
                               bool keep_dim,
                               bool reduce_all,
                               DenseTensor* x_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor out_grad_;

  if (FLAGS_use_stride_compute_kernel && out_grad.dims().size() > 0) {
    phi::DenseTensor out_tmp = CheckMultipleUnsqueeze<Context>(
        dev_ctx, out_grad, dims, x.dims().size(), keep_dim);

    phi::ExpandStrideKernel<T, Context>(
        dev_ctx, out_tmp, phi::IntArray(common::vectorize(x.dims())), x_grad);
    return;
  }

  // if x is contiguous is not relevant to sum_grad computation
  if (!out_grad.meta().is_contiguous()) {
    out_grad_ = Tensor2Contiguous<Context>(dev_ctx, out_grad);
  } else {
    out_grad_ = out_grad;
  }

  auto x_grad_meta = x_grad->meta();
  x_grad_meta.strides = x_grad_meta.calc_strides(x_grad->dims());
  x_grad->set_meta(x_grad_meta);
  phi::ReduceSumGradKernel<T>(
      dev_ctx, x, out_grad_, dims, keep_dim, reduce_all, x_grad);
}

}  // namespace phi

using float16 = phi::float16;
using bfloat16 = phi::bfloat16;
using complex64 = ::phi::complex64;
using complex128 = ::phi::complex128;

PD_REGISTER_KERNEL(sum_grad,
                   GPU,
                   STRIDED,
                   phi::ReduceSumGradStrideKernel,
                   bool,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
#endif
