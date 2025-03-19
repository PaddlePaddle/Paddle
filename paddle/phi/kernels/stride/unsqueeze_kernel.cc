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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void UnsqueezeStridedKernel(const Context& dev_ctx,
                            const DenseTensor& input,
                            const IntArray& axes_arr,
                            DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> input_dims = common::vectorize<int64_t>(input.dims());
  std::vector<int64_t> input_stride =
      common::vectorize<int64_t>(input.strides());

  if (input.Holder() == out->Holder() && input.meta() == out->meta()) {
    input_dims = common::vectorize<int64_t>(out->dims());
    for (int64_t i = static_cast<int64_t>(axes.size() - 1); i >= 0; --i) {
      axes[i] = static_cast<int64_t>(axes[i] < 0 ? axes[i] + input_dims.size()
                                                 : axes[i]);
      axes[i] = axes[i] < 0 ? 0 : axes[i];
      input_dims.erase(input_dims.begin() + axes[i]);
    }
  }

  std::vector<int64_t> output_dims = input_dims;
  std::vector<int64_t> output_stride = input_stride;

  for (int64_t item : axes) {
    item =
        static_cast<int64_t>(item < 0 ? item + output_dims.size() + 1 : item);
    item = item < 0 ? 0 : item;
    int64_t stride = static_cast<size_t>(item) >= output_dims.size()
                         ? 1
                         : output_dims[item] * output_stride[item];
    output_dims.insert(output_dims.begin() + item, 1);
    output_stride.insert(output_stride.begin() + item, stride);
  }

  auto meta = out->meta();
  auto tmp_dim = DDim(output_dims.data(), static_cast<int>(output_dims.size()));
  // if (product(meta.dims) > 0 && meta.dims != tmp_dim) {
  //   PADDLE_THROW(
  //       common::errors::Fatal("Unsqueeze kernel stride compute diff, infer
  //       shape"
  //                          "is %s, but compute is %s.",
  //                          meta.dims,
  //                          tmp_dim));
  // }
  meta.dims = tmp_dim;
  meta.strides =
      DDim(output_stride.data(), static_cast<int>(output_stride.size()));
  meta.offset = input.offset();
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
  out->ShareInplaceVersionCounterWith(input);
}

template <typename Context>
void UnsqueezeWithXShapeStridedKernel(const Context& dev_ctx,
                                      const DenseTensor& x,
                                      const IntArray& axes,
                                      DenseTensor* out,
                                      DenseTensor* xshape) {
  UnsqueezeStridedKernel<Context>(dev_ctx, x, axes, out);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(unsqueeze,
                                         STRIDED,
                                         phi::UnsqueezeStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(
    unsqueeze_with_xshape, STRIDED, phi::UnsqueezeWithXShapeStridedKernel) {}
