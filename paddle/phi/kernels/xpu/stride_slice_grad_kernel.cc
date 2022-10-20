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

#include "paddle/phi/kernels/strided_slice_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void StridedSliceRawGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& out_grad,
                               const std::vector<int>& axes,
                               const IntArray& starts,
                               const IntArray& ends,
                               const IntArray& strides,
                               const std::vector<int>& infer_flags,
                               const std::vector<int>& decrease_axis,
                               DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  DDim in_dims = x.dims();
  dev_ctx.template Alloc<T>(x_grad);

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  std::vector<int> xshape;
  std::vector<int> starts_in(in_dims.size(), 0);
  std::vector<int> ends_in;
  std::vector<int> strides_in(in_dims.size(), 1);

  for (int i = 0; i < in_dims.size(); ++i) {
    xshape.emplace_back(in_dims[i]);
    ends_in.emplace_back(in_dims[i]);
  }

  int num = axes.size();

  for (int i = 0; i < num; ++i) {
    PADDLE_ENFORCE_EQ(
        strides_[i] > 0,
        true,
        errors::InvalidArgument("xpu does not support reverse strided slice"));
    int cur_axe = axes[i];
    int st = starts_[i];
    if (st > xshape[cur_axe]) {
      st = xshape[cur_axe];
    }
    if (st < 0) {
      st += xshape[cur_axe];
    }
    starts_in[cur_axe] = st;

    int end = ends_[i];
    if (end > xshape[cur_axe]) {
      end = xshape[cur_axe];
    }
    if (end < 0) {
      end += xshape[cur_axe];
    }

    ends_in[cur_axe] = end;
    strides_in[cur_axe] = strides_[i];
  }

  int r = xpu::strided_slice_grad(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(x_grad->data<T>()),
      xshape,
      starts_in,
      ends_in,
      strides_in);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_slice_raw_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::StridedSliceRawGradKernel,
                   int,
                   int16_t,
                   float,
                   phi::dtype::float16) {}
