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

#include "paddle/phi/kernels/pool_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace phi {
template <typename T, typename Context>
void Pool2dGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      const std::vector<int>& kernel_size,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      DenseTensor* dx) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  std::vector<int> paddings_t(paddings);
  std::vector<int> kernel_size_t(kernel_size);
  std::vector<int> strides_t(strides);

  PADDLE_ENFORCE_EQ(
      data_format,
      "NCHW",
      phi::errors::InvalidArgument("The Pool2d_grad XPU OP only support"
                                   "data_format is 'NCHW', but received %s",
                                   data_format));

  PADDLE_ENFORCE_EQ(
      kernel_size.size(),
      2,
      phi::errors::InvalidArgument("The Pool2d XPU OP only support 2 "
                                   "dimension pooling!, but received "
                                   "%d-dimension pool kernel size",
                                   kernel_size.size()));
  if (global_pooling) {
    for (size_t i = 0; i < kernel_size.size(); ++i) {
      paddings_t[i] = 0;
      kernel_size_t[i] = static_cast<int>(x.dims()[i + 2]);
    }
  }
  if (!dx) {
    return;
  }
  const int n = x.dims()[0];
  const int c = x.dims()[1];
  const int in_h = x.dims()[2];
  const int in_w = x.dims()[3];

  const int out_h = out.dims()[2];
  const int out_w = out.dims()[3];

  DDim data_dims;

  data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  funcs::UpdatePadding(&paddings_t,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides_t,
                       kernel_size_t);
  if (ceil_mode) {
    int in_h_ceil =
        (out_h - 1) * strides_t[0] + kernel_size_t[0] - 2 * paddings_t[0];
    int in_w_ceil =
        (out_w - 1) * strides_t[1] + kernel_size_t[1] - 2 * paddings_t[2];

    paddings_t[1] += (in_h_ceil - in_h);
    paddings_t[3] += (in_w_ceil - in_w);
  }

  ctx.template Alloc<T>(dx);
  const int* index_data = nullptr;
  int r = xpu::Error_t::SUCCESS;
  if (adaptive) {
    // floor for stride
    strides_t = {in_h / out_h, in_w / out_w};
    int kh = in_h - (out_h - 1) * strides_t[0];
    int kw = in_w - (out_w - 1) * strides_t[1];
    kernel_size_t = {kh, kw};
    paddings_t = {0, 0, 0, 0};
  }

  if (pooling_type == "max") {
    // TODO(zhanghuan05) to bind max_pool2d_grad_indices xpu api
    r = xpu::max_pool2d_grad<XPUType>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(out.data<T>()),
        index_data,
        reinterpret_cast<const XPUType*>(dout.data<T>()),
        reinterpret_cast<XPUType*>(dx->data<T>()),
        n,
        c,
        in_h,
        in_w,
        kernel_size_t,
        strides_t,
        paddings_t,
        true);
  } else if (pooling_type == "avg") {
    r = xpu::avg_pool2d_grad<XPUType>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(out.data<T>()),
        reinterpret_cast<const XPUType*>(dout.data<T>()),
        reinterpret_cast<XPUType*>(dx->data<T>()),
        n,
        c,
        in_h,
        in_w,
        kernel_size_t,
        strides_t,
        paddings_t,
        !exclusive,
        true);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported pooling type for kunlun ", pooling_type));
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pool2dgrad");
}
}  // namespace phi

PD_REGISTER_KERNEL(pool2d_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::Pool2dGradKernel,
                   float,
                   phi::dtype::float16) {}
