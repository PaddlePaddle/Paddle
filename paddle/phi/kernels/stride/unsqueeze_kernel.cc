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

#include "paddle/phi/kernels/unsqueeze_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"

namespace phi {

template <typename Context>
void UnsqueezeInferStridedKernel(const Context& dev_ctx,
                                 const DenseTensor& input,
                                 const IntArray& axes_arr,
                                 DenseTensor* out) {
  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> output_dims = phi::vectorize<int64_t>(input.dims());
  std::vector<int64_t> output_stride = phi::vectorize<int64_t>(input.stride());

  for (auto item : axes) {
    auto axis = item < 0 ? item + output_dims.size() + 1 : item;
    int64_t stride = static_cast<size_t>(axis) >= output_dims.size()
                         ? 1
                         : output_dims[axis] * output_stride[axis];
    output_dims.insert(output_dims.begin() + axis, 1);
    output_stride.insert(output_stride.begin() + axis, stride);
  }

  auto meta = out->meta();
  auto tmp_dim = DDim(output_dims.data(), output_dims.size());
  if (product(meta.dims) > 0 && meta.dims != tmp_dim) {
    PADDLE_THROW(
        phi::errors::Fatal("Unsqueeze kernel stride compute diff, infer shape "
                           "is %s, but compute is %s.",
                           meta.dims,
                           tmp_dim));
  }
  meta.stride = DDim(output_stride.data(), output_stride.size());
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
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
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    unsqueeze_infer, STRIDED, phi::UnsqueezeInferStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    unsqueeze, STRIDED, phi::UnsqueezeStridedKernel) {}
