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

#include "paddle/phi/kernels/slice_kernel.h"

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void SliceStridedKernel(const Context& ctx,
                        const DenseTensor& input,
                        const std::vector<int64_t>& axes,
                        const IntArray& starts_arr,
                        const IntArray& ends_arr,
                        const std::vector<int64_t>& infer_flags,
                        const std::vector<int64_t>& decrease_axis,
                        DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();
  const auto& in_dims = input.dims();

  auto new_axes = axes;
  for (auto& item : new_axes) {
    if (item < 0) {
      item = std::max(int64_t(0), item + int64_t(in_dims.size()));
    }
  }
  // axis = 0, dim_value = 3, st[0]=0, ed[0]=4
  // The step seems to be regarded as 1 here
  phi::funcs::CheckAndUpdateSliceAttrs<int64_t>(
      in_dims, new_axes, &starts, &ends, nullptr, nullptr);

  std::vector<int64_t> output_dims = common::vectorize<int64_t>(input.dims());
  std::vector<int64_t> output_stride =
      common::vectorize<int64_t>(input.strides());
  int64_t output_offset = static_cast<int64_t>(input.offset());

  for (size_t i = 0; i < new_axes.size(); ++i) {
    output_offset = static_cast<int64_t>(
        output_offset +
        starts[i] * output_stride[new_axes[i]] * SizeOf(out->dtype()));
    output_dims[new_axes[i]] = std::abs(ends[i] - starts[i]);
  }

  std::vector<uint8_t> decrease_flag(output_dims.size(), 0);
  if (!decrease_axis.empty()) {
    for (auto axis : decrease_axis) {
      decrease_flag[axis] = 1;
    }

    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_stride;
    for (size_t i = 0; i < output_dims.size(); ++i) {
      if (decrease_flag[i] == 0) {
        new_shape.push_back(output_dims[i]);
        new_stride.push_back(output_stride[i]);
      }
    }
    output_dims = new_shape;
    output_stride = new_stride;
  }

  auto meta = out->meta();
  meta.offset = output_offset;
  auto tmp_dim = DDim(output_dims.data(), static_cast<int>(output_dims.size()));
  // if (product(meta.dims) > 0 && meta.dims != tmp_dim) {
  //   PADDLE_THROW(
  //       common::errors::Fatal("Slice kernel stride compute diff, infer shape
  //       is
  //       "
  //                          "%s, but compute is %s.",
  //                          meta.dims,
  //                          tmp_dim));
  // }
  meta.dims = tmp_dim;
  meta.strides =
      DDim(output_stride.data(), static_cast<int>(output_stride.size()));
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
  out->ShareInplaceVersionCounterWith(input);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(slice,
                                         STRIDED,
                                         phi::SliceStridedKernel) {}
