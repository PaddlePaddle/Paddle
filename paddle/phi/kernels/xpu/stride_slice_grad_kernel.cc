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
#include "paddle/phi/kernels/xpu/stride_slice_util.h"

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
      if (!(end == -1 && strides_[i] < 0)) {
        end = end + xshape[cur_axe];
        if (end < 0) {
          end = 0;
        }
      }
    }

    ends_in[cur_axe] = end;
    strides_in[cur_axe] = strides_[i];
  }

  if (is_strided_slice_special_case(xshape, starts_in, ends_in, strides_in)) {
    PADDLE_ENFORCE_EQ(
        x.numel(),
        x_grad->numel(),
        errors::PreconditionNotMet(
            "x.numel() should be equal to x_grad->numel() in special case."));
    PADDLE_ENFORCE_EQ(
        x.numel(),
        out_grad.numel() * 2,
        errors::PreconditionNotMet("x.numel() should be equal to "
                                   "out_grad->numel() * 2 in special case."));

    /*
     * sample input: [1 2 3 4 5]
     * starts = [0/1]
     * strides = [2]
     * sample output: [1 0 2 0 3 0 4 0 5 0] (last value in starts is 0)
     * sample output: [0 1 0 2 0 3 0 4 0 5] (last value in starts is 1)
     */
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    XPUType* x_transpose = RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());

    // step 1: set all value to 0

    // int constant(Context* ctx, T* x, int len, T val)
    int r = xpu::constant(
        dev_ctx.x_context(), x_transpose, x.numel(), static_cast<XPUType>(0));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

    /*
     * step 2: copy dy to dx:
     * if starts from 0: [1 2 3 4 5 0 0 0 0 0]
     * if starts from 1: [0 0 0 0 0 1 2 3 4 5]
     */
    int offset = 0;
    if (starts_in.back() == 1) {
      offset = x.numel() / 2;
    }
    // int copy(Context* ctx, const T* x, T* y, int64_t len)
    r = xpu::copy<XPUType>(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                           x_transpose + offset,
                           x.numel() / 2);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    /*
     * step3: transpose, input shape is (2, x.numel/2):
     * input:
     * [1 2 3 4 5
     *  0 0 0 0 0]
     * after transpose:
     * [1 0
     *  2 0
     *  3 0
     *  4 0
     *  5 0]
     */
    r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                x_transpose,
                                reinterpret_cast<XPUType*>(x_grad->data<T>()),
                                {2, x.numel() / 2},
                                {1, 0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    return;
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
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
