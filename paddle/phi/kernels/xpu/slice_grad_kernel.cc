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

#include "paddle/phi/kernels/slice_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {

template <typename T, typename Context>
void SliceGradRawKernel(const Context& ctx,
                        const DenseTensor& input,
                        const DenseTensor& out_grad,
                        const std::vector<int64_t>& axes,
                        const IntArray& starts_t,
                        const IntArray& ends_t,
                        const std::vector<int64_t>& infer_flags,
                        const std::vector<int64_t>& decrease_axis,
                        DenseTensor* input_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(input_grad);

  // Get the accurate attribute value of starts and ends
  std::vector<int64_t> starts = starts_t.GetData();
  std::vector<int64_t> ends = ends_t.GetData();

  const auto& in_dims = input.dims();
  int rank = in_dims.size();

  std::vector<int> pad_left(rank);
  std::vector<int> out_dims(rank);
  std::vector<int> pad_right(rank);
  int cnt = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    int start = 0;
    int end = in_dims[i];
    int axis = cnt < static_cast<int>(axes.size()) ? axes[cnt] : -1;
    if (axis == i) {
      start = starts[cnt];
      if (start < 0) {
        start = (start + in_dims[i]);
      }
      start = std::max(start, static_cast<int>(0));
      end = ends[cnt];
      if (end < 0) {
        end = (end + in_dims[i]);
      }
      end = std::min(end, static_cast<int>(in_dims[i]));
      cnt++;
    }

    pad_left[i] = start;
    out_dims[i] = end - start;
    pad_right[i] = in_dims[i] - out_dims[i] - pad_left[i];
  }

  int r =
      xpu::pad<XPUType>(ctx.x_context(),
                        reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                        reinterpret_cast<XPUType*>(input_grad->data<T>()),
                        out_dims,
                        pad_left,
                        pad_right,
                        XPUType(0));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
}
}  // namespace phi

PD_REGISTER_KERNEL(slice_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SliceGradRawKernel,
                   float,
                   int,
                   phi::dtype::float16) {}
