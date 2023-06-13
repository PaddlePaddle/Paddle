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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context>
void StridedSliceRawStridedKernel(const Context& dev_ctx,
                                  const DenseTensor& input,
                                  const std::vector<int>& axis,
                                  const IntArray& starts_arr,
                                  const IntArray& ends_arr,
                                  const IntArray& strides_arr,
                                  const std::vector<int>& infer_flags,
                                  const std::vector<int>& decrease_axis,
                                  DenseTensor* out) {
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();
  std::vector<int64_t> strides = strides_arr.GetData();

  std::vector<int64_t> output_dims = phi::vectorize<int64_t>(input.dims());
  std::vector<int64_t> output_stride = phi::vectorize<int64_t>(input.stride());
  int64_t output_offset = input.offset();
  for (size_t i = 0; i < axis.size(); ++i) {
    int64_t axis_size = input.dims()[axis[i]];

    if (axis_size < 0) {
      continue;
    }

    if (starts[i] < 0) {
      starts[i] = starts[i] + axis_size;
      starts[i] = std::max<int64_t>(starts[i], 0);
    }
    if (ends[i] < 0) {
      if (!(ends[i] == -1 && strides[i] < 0)) {  // skip None stop condition
        ends[i] = ends[i] + axis_size;
        if (ends[i] < 0) {
          ends[i] = 0;
        }
      }
    }
    if (strides[i] < 0) {
      starts[i] = starts[i] + 1;
      ends[i] = ends[i] + 1;
    }

    int64_t left =
        std::max(static_cast<int64_t>(0), std::min(starts[i], ends[i]));
    int64_t right = std::min(axis_size, std::max(starts[i], ends[i]));
    int64_t step = std::abs(strides[i]);

    auto dim = (std::abs(right - left) + step - 1) / step;

    if (dim <= 0) {
      dim = 0;
      strides[i] = 1;
      starts[i] = 0;
    }

    if (starts[i] >= axis_size) {
      starts[i] = (strides[i] < 0) ? axis_size - 1 : axis_size;
    }

    output_offset += starts[i] * output_stride[axis[i]] * SizeOf(out->dtype());
    output_dims[axis[i]] = dim;
    output_stride[axis[i]] *= strides[i];
  }

  if (*output_dims.begin() == 1) {
    output_dims.erase(output_dims.begin());
    output_stride.erase(output_stride.begin());
  }

  if (*output_dims.end() == 1) {
    output_dims.erase(output_dims.end());
    output_stride.erase(output_stride.end());
  }

  auto meta = out->meta();
  meta.offset = output_offset;
  auto tmp_dim = DDim(output_dims.data(), output_dims.size());
  if (meta.dims != tmp_dim) {
    LOG(WARNING) << "Striede_slice kernel stride compute diff, infer shape is "
                 << meta.dims << ", but compute is " << tmp_dim << ".";
    meta.dims = tmp_dim;
  }
  meta.stride = DDim(output_stride.data(), output_stride.size());
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
}

template <typename Context>
void StridedSliceStridedKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const std::vector<int>& axis,
                               const IntArray& starts,
                               const IntArray& ends,
                               const IntArray& strides,
                               DenseTensor* out) {
  std::vector<int> infer_flags(axis.size(), 1);
  std::vector<int> decrease_axis;
  StridedSliceRawStridedKernel<Context>(
      dev_ctx, x, axis, starts, ends, strides, infer_flags, decrease_axis, out);
}
}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    strided_slice_raw, STRIDED, phi::StridedSliceRawStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    strided_slice, STRIDED, phi::StridedSliceStridedKernel) {}
