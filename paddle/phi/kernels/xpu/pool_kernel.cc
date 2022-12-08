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

#include "paddle/phi/kernels/pool_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace phi {
template <typename T, typename Context>
void Pool2dKernel(const Context& ctx,
                  const DenseTensor& x,
                  const IntArray& kernel_size_t,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings_t,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  std::vector<int> paddings(paddings_t);
  std::vector<int> kernel_size(kernel_size_t.GetData().begin(),
                               kernel_size_t.GetData().end());

  PADDLE_ENFORCE_EQ(kernel_size.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The Pool2d XPU OP only support 2 dimension pooling!"));

  PADDLE_ENFORCE_EQ(
      data_format,
      "NCHW",
      phi::errors::InvalidArgument("The Pool2d XPU OP only support "
                                   "data_format is 'NCHW', but received %s",
                                   data_format));

  if (global_pooling) {
    for (size_t i = 0; i < kernel_size.size(); ++i) {
      paddings[i] = 0;
      kernel_size[i] = static_cast<int>(x.dims()[i + 2]);
    }
  }

  const int n = x.dims()[0];
  const int c = x.dims()[1];
  const int in_h = x.dims()[2];
  const int in_w = x.dims()[3];

  const int out_h = out->dims()[2];
  const int out_w = out->dims()[3];

  DDim data_dims;

  data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  funcs::UpdatePadding(&paddings,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size);

  if (ceil_mode) {
    int in_h_ceil = (out_h - 1) * strides[0] + kernel_size[0] - 2 * paddings[0];
    int in_w_ceil = (out_w - 1) * strides[1] + kernel_size[1] - 2 * paddings[2];

    paddings[1] += (in_h_ceil - in_h);
    paddings[3] += (in_w_ceil - in_w);
  }

  ctx.template Alloc<T>(out);
  int* index_data = nullptr;
  int r = xpu::Error_t::SUCCESS;
  if (!adaptive) {
    if (pooling_type == "max") {
      r = xpu::max_pool2d<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          index_data,
          n,
          c,
          in_h,
          in_w,
          kernel_size,
          strides,
          paddings,
          true);
    } else if (pooling_type == "avg") {
      r = xpu::avg_pool2d<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          n,
          c,
          in_h,
          in_w,
          kernel_size,
          strides,
          paddings,
          !exclusive,
          true);
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
  } else {
    if (pooling_type == "max") {
      r = xpu::adaptive_max_pool2d<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          index_data,
          n,
          c,
          in_h,
          in_w,
          out_h,
          out_w,
          true);
    } else if (pooling_type == "avg") {
      r = xpu::adaptive_avg_pool2d<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          n,
          c,
          in_h,
          in_w,
          out_h,
          out_w,
          true);
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pool2d");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    pool2d, XPU, ALL_LAYOUT, phi::Pool2dKernel, float, phi::dtype::float16) {}
