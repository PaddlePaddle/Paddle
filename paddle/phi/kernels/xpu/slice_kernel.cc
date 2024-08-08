// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {

template <typename T, typename Context>
void SliceKernel(const Context& ctx,
                 const DenseTensor& input,
                 const std::vector<int64_t>& axes,
                 const IntArray& starts_t,
                 const IntArray& ends_t,
                 const std::vector<int64_t>& infer_flags,
                 const std::vector<int64_t>& decrease_axis,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  // Step 1: Get the accurate attribute value of starts and ends
  std::vector<int64_t> starts = starts_t.GetData();
  std::vector<int64_t> ends = ends_t.GetData();
  PADDLE_ENFORCE_EQ(
      starts.size(),
      axes.size(),
      common::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(),
                    axes.size(),
                    common::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  // Step 2: Compute output
  auto in_dims = input.dims();
  auto out_dims = out->dims();
  auto slice_dims = out_dims;
  bool is_same = true;
  if (in_dims.size() == out_dims.size()) {
    for (int i = 0; i < in_dims.size(); i++) {
      if (in_dims[i] != out_dims[i]) {
        is_same = false;
        break;
      } else {
        continue;
      }
    }
    if (is_same) {
      phi::Copy<Context>(ctx, input, ctx.GetPlace(), false, out);
      return;
    }
  }

  // 2.1 Infer output dims
  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start+1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }

  phi::funcs::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
  slice_dims = funcs::GetSliceDims<int64_t>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = funcs::GetDecreasedDims(slice_dims, decrease_axis);

  out->Resize(out_dims);

  // 2.2 Get output
  size_t shape_size = in_dims.size();
  // the slice XPU kernel require that the length of `start`, `end` must be
  // equal
  // to the dims size of input tensor, therefore, if shape_size >
  // axes.size(), the `starts_extension` and `ends_extension` is necessary.
  std::vector<int> starts_extension(shape_size, 0);
  std::vector<int> ends_extension(shape_size, 0);
  if (shape_size > axes.size()) {
    for (size_t i = 0; i < shape_size; ++i) {
      ends_extension[i] = in_dims[i];
    }
    for (size_t i = 0; i < axes.size(); ++i) {
      starts_extension[axes[i]] = starts[i];
      ends_extension[axes[i]] = ends[i];
    }
  } else {
    for (size_t i = 0; i < axes.size(); ++i) {
      starts_extension[i] = starts[i];
      ends_extension[i] = ends[i];
    }
  }

  // prepare shape on XPU
  std::vector<int> shape(shape_size, 0);
  for (size_t i = 0; i < shape_size; ++i) {
    shape[i] = in_dims[i];
  }

  ctx.template Alloc<T>(out);
  for (size_t i = 0; i < shape_size; ++i) {
    if (starts_extension[i] == ends_extension[i] || shape[i] == 0) {
      return;
    }
  }

  int r = xpu::slice<XPUType>(ctx.x_context(),
                              reinterpret_cast<const XPUType*>(input.data<T>()),
                              reinterpret_cast<XPUType*>(out->data<T>()),
                              shape,
                              starts_extension,
                              ends_extension);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
}
}  // namespace phi

PD_REGISTER_KERNEL(slice,
                   XPU,
                   ALL_LAYOUT,
                   phi::SliceKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t) {}
