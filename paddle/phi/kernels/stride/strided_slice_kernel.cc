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

#include "paddle/phi/kernels/strided_slice_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void StridedSliceRawStridedKernel(const Context& dev_ctx,
                                  const DenseTensor& input,
                                  const std::vector<int>& axes,
                                  const IntArray& starts_arr,
                                  const IntArray& ends_arr,
                                  const IntArray& strides_arr,
                                  const std::vector<int>& infer_flags,
                                  const std::vector<int>& decrease_axis,
                                  DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();
  std::vector<int64_t> strides = strides_arr.GetData();

  std::vector<int64_t> output_dims = common::vectorize<int64_t>(input.dims());
  std::vector<int64_t> output_stride =
      common::vectorize<int64_t>(input.strides());
  int64_t output_offset = static_cast<int64_t>(input.offset());
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis_size = input.dims()[axes[i]];

    if (axis_size < 0) {
      continue;
    }
    bool dummy_zero_dim_out = false;
    funcs::normalize_interval(starts[i],
                              ends[i],
                              strides[i],
                              axis_size,
                              &starts[i],
                              &ends[i],
                              &dummy_zero_dim_out);
    if (ends[i] == -axis_size - 1) {
      ends[i] = -1;
    }

    int64_t step_size = std::abs(strides[i]);

    auto out_dim = (std::abs(ends[i] - starts[i]) + step_size - 1) / step_size;

    output_offset += static_cast<int>(starts[i] * output_stride[axes[i]] *
                                      SizeOf(out->dtype()));
    output_dims[axes[i]] = out_dim;
    output_stride[axes[i]] *= strides[i];
  }

  // generate new shape
  if (!decrease_axis.empty()) {
    std::vector<int64_t> new_out_shape;
    std::vector<int64_t> new_out_stride;
    for (auto de_axis : decrease_axis) {
      output_dims[de_axis] = 0;
    }

    for (size_t i = 0; i < output_dims.size(); ++i) {
      if (output_dims[i] != 0) {
        new_out_shape.push_back(output_dims[i]);
        new_out_stride.push_back(output_stride[i]);
      }
    }
    output_dims = new_out_shape;
    output_stride = new_out_stride;
  }

  auto meta = out->meta();
  meta.offset = output_offset;
  auto tmp_dim = DDim(output_dims.data(), static_cast<int>(output_dims.size()));
  // if (product(meta.dims) > 0 && meta.dims != tmp_dim) {
  //   PADDLE_THROW(
  //       common::errors::Fatal("Striede_slice kernel stride compute diff,
  //       infer "
  //                          "shape is %s, but compute is %s.",
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

template <typename Context>
void StridedSliceStridedKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const std::vector<int>& axes,
                               const IntArray& starts,
                               const IntArray& ends,
                               const IntArray& strides,
                               DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  StridedSliceRawStridedKernel<Context>(
      dev_ctx, x, axes, starts, ends, strides, infer_flags, decrease_axis, out);
}
}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(strided_slice_raw,
                                         STRIDED,
                                         phi::StridedSliceRawStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(strided_slice,
                                         STRIDED,
                                         phi::StridedSliceStridedKernel) {}
