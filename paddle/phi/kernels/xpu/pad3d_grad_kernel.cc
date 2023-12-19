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

#include "paddle/phi/kernels/pad3d_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void Pad3dGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     const IntArray& paddings,
                     const std::string& mode,
                     float pad_value,
                     const std::string& data_format,
                     DenseTensor* x_grad) {
  T value = static_cast<T>(pad_value);
  std::vector<int64_t> pads = paddings.GetData();

  auto* d_out = &out_grad;
  auto* d_in = x_grad;
  auto d_in_dims = common::vectorize<int>(d_in->dims());
  const T* d_out_data = d_out->data<T>();
  T* d_in_data = dev_ctx.template Alloc<T>(d_in);

  bool is_ncdhw = true;
  if (data_format == "NDHWC") {
    is_ncdhw = false;
  }

  const int num = d_in_dims[0];  // n
  int channels = d_in_dims[1];   // c
  int in_depth = d_in_dims[2];   // xd
  int in_height = d_in_dims[3];  // xh
  int in_width = d_in_dims[4];   // xw
  if (data_format == "NDHWC") {
    channels = d_in_dims[4];
    in_depth = d_in_dims[1];
    in_height = d_in_dims[2];
    in_width = d_in_dims[3];
  }

  std::vector<int> pads_xpu(6);
  pads_xpu[0] = pads[4];  // pf
  pads_xpu[1] = pads[5];  // pb
  pads_xpu[2] = pads[2];  // pt
  pads_xpu[3] = pads[3];  // pd
  pads_xpu[4] = pads[0];  // pl
  pads_xpu[5] = pads[1];  // pr

  if (mode == "reflect") {
    int r = xpu::reflection_pad3d_grad(dev_ctx.x_context(),
                                       d_out_data,
                                       d_in_data,
                                       num,
                                       channels,
                                       in_depth,
                                       in_height,
                                       in_width,
                                       pads_xpu,
                                       is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reflection_pad3d_grad");
  } else if (mode == "replicate") {
    int r = xpu::replication_pad3d_grad(dev_ctx.x_context(),
                                        d_out_data,
                                        d_in_data,
                                        num,
                                        channels,
                                        in_depth,
                                        in_height,
                                        in_width,
                                        pads_xpu,
                                        is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "replication_pad3d_grad");
  } else if (mode == "constant") {
    int r = xpu::constant_pad3d_grad(dev_ctx.x_context(),
                                     d_out_data,
                                     d_in_data,
                                     num,
                                     channels,
                                     in_depth,
                                     in_height,
                                     in_width,
                                     pads_xpu,
                                     value,
                                     is_ncdhw);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant_pad3d_grad");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(pad3d_grad, XPU, ALL_LAYOUT, phi::Pad3dGradKernel, float) {}
