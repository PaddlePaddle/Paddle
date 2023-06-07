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
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/flatten_kernel.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/reshape_kernel.h"

namespace phi {

template <typename Context>
void UnsqueezeInferStridedKernel(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const IntArray& axes,
                                 DenseTensor* out) {
  if (x.Holder() == out->Holder()) {
    return;
  }

  auto x_dims = x.dims();
  auto out_dims = out->dims();
  if (axes.FromTensor()) {
    out_dims = funcs::GetUnsqueezeShape(axes.GetData(), x_dims);
  }
  out->Resize(out_dims);
  ReshapeStridedKernel<Context>(
      dev_ctx, x, IntArray(phi::vectorize<int64_t>(out->dims())), out, nullptr);
}

template <typename Context>
void UnsqueezeStridedKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const IntArray& axes,
                            DenseTensor* out,
                            DenseTensor* xshape) {
  UnsqueezeInferStridedKernel<Context>(dev_ctx, x, axes, out);
}

}  // namespace phi
#ifndef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(unsqueeze_infer,
                                         STRIDED,
                                         phi::UnsqueezeInferStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(unsqueeze,
                                         STRIDED,
                                         phi::UnsqueezeStridedKernel) {}
#endif
