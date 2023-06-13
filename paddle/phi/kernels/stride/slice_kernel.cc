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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {

template <typename Context>
void SliceStridedKernel(const Context& ctx,
                        const DenseTensor& input,
                        const std::vector<int64_t>& axis,
                        const IntArray& starts_arr,
                        const IntArray& ends_arr,
                        const std::vector<int64_t>& infer_flags_t,
                        const std::vector<int64_t>& decrease_axis,
                        DenseTensor* out) {
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();
  auto in_dims = input.dims();

  std::vector<int64_t> infer_flags = infer_flags_t;
  if (infer_flags.empty()) {
    // Initialize infer_flags with 1.
    // To be compatible with other op tests in which infer_flags is not set.
    infer_flags = std::vector<int64_t>(axis.size(), 1);
  }

  auto new_axis = axis;
  for (auto& item : new_axis) {
    if (item < 0) {
      item = std::max(int64_t(0), item + int64_t(in_dims.size()));
    }
  }

  phi::funcs::CheckAndUpdateSliceAttrs<int64_t>(
      in_dims, new_axis, &starts, &ends, nullptr, &infer_flags);

  for (size_t i = 0; i < new_axis.size(); ++i) {
    // when start == -1 && end == start+1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret =
          std::find(decrease_axis.begin(), decrease_axis.end(), new_axis[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[new_axis[i]];
      }
    }
  }

  std::vector<int64_t> output_dims = phi::vectorize<int64_t>(input.dims());
  std::vector<int64_t> output_stride = phi::vectorize<int64_t>(input.stride());
  int64_t output_offset = input.offset();

  for (size_t i = 0; i < new_axis.size(); ++i) {
    output_offset = output_offset + starts[i] * output_stride[new_axis[i]] *
                                        SizeOf(out->dtype());
    output_dims[new_axis[i]] = ends[i] - starts[i];
  }

  auto iter_dims = output_dims.begin();
  auto iter_stride = output_stride.begin();
  while (iter_dims != output_dims.end()) {
    if (*iter_dims == 1) {
      iter_dims = output_dims.erase(iter_dims);
      iter_stride = output_stride.erase(iter_stride);
    } else {
      iter_dims++;
      iter_stride++;
    }
  }

  auto meta = out->meta();
  meta.offset = output_offset;
  auto tmp_dim = DDim(output_dims.data(), output_dims.size());
  if (meta.dims != tmp_dim) {
    LOG(WARNING) << "Slice kernel stride compute diff, infer shape is "
                 << meta.dims << ", but compute is " << tmp_dim << ".";
    meta.dims = tmp_dim;
  }
  meta.stride = DDim(output_stride.data(), output_stride.size());
  out->set_meta(meta);
  out->ResetHolder(input.Holder());
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    slice, STRIDED, phi::SliceStridedKernel) {}
