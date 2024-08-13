// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void Pad2dXPUKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<int>& paddings,
                    const std::string& mode,
                    float pad_value,
                    const std::string& data_format,
                    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<int> pads = paddings;

  auto in_dims = x.dims();
  const T* in_data = x.data<T>();
  XPUType value = static_cast<XPUType>(pad_value);

  if (data_format == "NCHW") {
    out->Resize({in_dims[0],
                 in_dims[1],
                 in_dims[2] + pads[2] + pads[3],    // xh
                 in_dims[3] + pads[0] + pads[1]});  // xw
  } else if (data_format == "NHWC") {
    out->Resize({in_dims[0],
                 in_dims[1] + pads[2] + pads[3],  // xh
                 in_dims[2] + pads[0] + pads[1],  // xw
                 in_dims[3]});
  } else {
    PADDLE_THROW(common::errors::External(
        "XPU is not support NCHW format in pad2d, data_format is %s",
        data_format));
  }

  T* out_data = dev_ctx.template Alloc<T>(out);
  const int num = in_dims[0];  // n
  int channels = in_dims[1];   // c
  int in_height = in_dims[2];  // xh
  int in_width = in_dims[3];   // xw
  if (data_format == "NHWC") {
    in_height = in_dims[1];  // xh
    in_width = in_dims[2];   // xw
    channels = in_dims[3];   // c
  }

  if (mode == "circular") {
    PADDLE_THROW(common::errors::External(
        "XPU is not support circular padding mode in pad2d"));
  }

  // check shape
  if (mode == "reflect") {
    PADDLE_ENFORCE_GT(
        in_height,
        pads[2],
        errors::InvalidArgument("The height of Input(X)'s dimension should be "
                                "greater than pad_top"
                                " in reflect mode"
                                ", but received depth(%d) and pad_top(%d).",
                                in_height,
                                pads[2]));
    PADDLE_ENFORCE_GT(
        in_height,
        pads[3],
        errors::InvalidArgument("The height of Input(X)'s dimension should be "
                                "greater than pad_bottom"
                                " in reflect mode"
                                ", but received depth(%d) and pad_bottom(%d).",
                                in_height,
                                pads[3]));

    PADDLE_ENFORCE_GT(
        in_width,
        pads[0],
        errors::InvalidArgument("The width of Input(X)'s dimension should be "
                                "greater than pad_left"
                                " in reflect mode"
                                ", but received depth(%d) and pad_left(%d).",
                                in_width,
                                pads[0]));
    PADDLE_ENFORCE_GT(
        in_width,
        pads[1],
        errors::InvalidArgument("The width of Input(X)'s dimension should be "
                                "greater than pad_right"
                                " in reflect mode"
                                ", but received depth(%d) and pad_right(%d).",
                                in_width,
                                pads[1]));
  } else if (mode == "replicate") {
    PADDLE_ENFORCE_NE(in_height * in_width,
                      0,
                      errors::InvalidArgument(
                          "The input tensor size can not be 0 for circular "
                          "or replicate padding mode."));
  }

  // set pad3d's pads to pad2d's pads_xpu
  std::vector<int> pads_xpu(4);
  pads_xpu[0] = pads[2];  // pt
  pads_xpu[1] = pads[3];  // pd
  pads_xpu[2] = pads[0];  // pl
  pads_xpu[3] = pads[1];  // pr

  // set pad3d's mode to pad2d's mode_xpu
  std::string mode_xpu = mode;
  if (mode == "replicate") {
    mode_xpu = "edge";
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  auto dev_version =
      phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId());
  if (dev_version == phi::backends::xpu::XPUVersion::XPU1) {
    if (mode_xpu == "constant" || mode_xpu == "edge" || mode_xpu == "reflect") {
      int r = xpu::pad2d<T>(dev_ctx.x_context(),
                            in_data,
                            out_data,
                            num,
                            channels,
                            in_height,
                            in_width,
                            pads_xpu,
                            mode_xpu.c_str(),
                            value,
                            (data_format == "NCHW"));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant || edge || reflect");
    } else {
      PADDLE_THROW(common::errors::External(
          "XPU is not support other padding mode in pad2d, mode_xpu is %s",
          mode_xpu));
    }
  } else if (dev_version == phi::backends::xpu::XPUVersion::XPU2 ||
             dev_version == phi::backends::xpu::XPUVersion::XPU3) {
    if (mode_xpu == "reflect") {
      int r = xpu::reflection_pad2d<T>(dev_ctx.x_context(),
                                       in_data,
                                       out_data,
                                       num,
                                       channels,
                                       in_height,
                                       in_width,
                                       pads_xpu,
                                       (data_format == "NCHW"));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "reflection_pad2d");
    } else if (mode_xpu == "constant" || mode_xpu == "edge") {
      int r = xpu::pad2d<T>(dev_ctx.x_context(),
                            in_data,
                            out_data,
                            num,
                            channels,
                            in_height,
                            in_width,
                            pads_xpu,
                            mode_xpu.c_str(),
                            value,
                            (data_format == "NCHW"));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    } else {
      PADDLE_THROW(common::errors::External(
          "XPU is not support other padding mode in pad2d, mode_xpu is %s",
          mode_xpu));
    }
  } else {
    PADDLE_THROW(common::errors::External(
        "not support other XPU version in pad2d is %s", dev_version));
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    pad2d_xpu, XPU, ALL_LAYOUT, phi::fusion::Pad2dXPUKernel, float) {}
