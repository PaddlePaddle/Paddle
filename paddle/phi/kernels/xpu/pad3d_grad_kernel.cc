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
                     double pad_value,
                     const std::string& data_format,
                     DenseTensor* x_grad) {
  T value = static_cast<T>(pad_value);
  std::vector<int64_t> pads = paddings.GetData();

  auto* d_out = &out_grad;
  auto* d_in = x_grad;
  auto d_in_dims = vectorize<int64_t>(d_in->dims());
  const T* d_out_data = d_out->data<T>();
  T* d_in_data = dev_ctx.template Alloc<T>(d_in);
  if (x.numel() == 0) return;

  bool is_ncdhw = true;
  if (data_format == "NDHWC") {
    is_ncdhw = false;
  }

  const int64_t num = d_in_dims[0];  // n
  int64_t channels = d_in_dims[1];   // c
  int64_t in_depth = d_in_dims[2];   // xd
  int64_t in_height = d_in_dims[3];  // xh
  int64_t in_width = d_in_dims[4];   // xw
  if (data_format == "NDHWC") {
    channels = d_in_dims[4];
    in_depth = d_in_dims[1];
    in_height = d_in_dims[2];
    in_width = d_in_dims[3];
  }

  std::vector<int64_t> pads_xpu(6);
  pads_xpu[0] = pads[4];  // pf
  pads_xpu[1] = pads[5];  // pb
  pads_xpu[2] = pads[2];  // pt
  pads_xpu[3] = pads[3];  // pd
  pads_xpu[4] = pads[0];  // pl
  pads_xpu[5] = pads[1];  // pr

  if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>) {
    PADDLE_ENFORCE_EQ(mode,
                      "constant",
                      errors::Unimplemented(
                          "XPU pad3d_grad only supports int and int64 input "
                          "for constant mode."));
    std::vector<int64_t> pad_left;
    std::vector<int64_t> pad_right;
    if (is_ncdhw) {
      pad_left = {0, 0, -pads[4], -pads[2], -pads[0]};
      pad_right = {0, 0, -pads[5], -pads[3], -pads[1]};
    } else {
      pad_left = {0, -pads[4], -pads[2], -pads[0], 0};
      pad_right = {0, -pads[5], -pads[3], -pads[1], 0};
    }
    // Integer zeropad2d may request pad3d_grad during kernel probing, while
    // XDNN has no constant_pad3d_grad<int/int64_t> symbols. Constant padding's
    // backward is just slicing the output gradient, which xpu::pad supports via
    // negative paddings without depending on pad3d_grad symbols.
    using XPUType = typename XPUTypeTrait<T>::Type;
    int r = xpu::pad<XPUType>(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType*>(d_out_data),
                              reinterpret_cast<XPUType*>(d_in_data),
                              vectorize<int64_t>(d_out->dims()),
                              pad_left,
                              pad_right,
                              static_cast<XPUType>(value));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
    return;
  } else {
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
}

}  // namespace phi

PD_REGISTER_KERNEL(
    pad3d_grad, XPU, ALL_LAYOUT, phi::Pad3dGradKernel, float, int, int64_t) {}
