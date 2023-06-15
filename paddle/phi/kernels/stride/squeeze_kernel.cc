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
#include "paddle/phi/kernels/squeeze_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context>
void SqueezeInferStridedKernel(const Context& dev_ctx,
                               const DenseTensor& input,
                               const IntArray& axes_arr,
                               DenseTensor* out) {
  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> output_dims = phi::vectorize<int64_t>(input.dims());
  std::vector<int64_t> output_stride = phi::vectorize<int64_t>(input.stride());

  for (auto item : axes) {
    if (output_dims.size() == 0) {
      break;
    }
    auto axis = item < 0 ? item + output_dims.size() + 1 : item;
    if (output_dims[axis] != 1) {
      continue;
    }
    output_dims.erase(output_dims.begin() + axis);
    output_stride.erase(output_stride.begin() + axis);
  }

  auto meta = out->meta();
  auto tmp_dim = DDim(output_dims.data(), output_dims.size());
  if (meta.dims != tmp_dim) {
    LOG(WARNING) << "Unsqueeze kernel stride compute diff, infer shape is "
                 << meta.dims << ", but compute is " << tmp_dim << ".";
    meta.dims = tmp_dim;
  }
  meta.stride = DDim(output_stride.data(), output_stride.size());
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
}

template <typename Context>
void SqueezeStridedKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const IntArray& axes,
                          DenseTensor* out,
                          DenseTensor* xshape UNUSED) {
  SqueezeInferStridedKernel<Context>(dev_ctx, x, axes, out);
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    squeeze_infer, STRIDED, phi::SqueezeInferStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    squeeze, STRIDED, phi::SqueezeStridedKernel) {}
