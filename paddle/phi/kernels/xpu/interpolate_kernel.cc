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

#include "paddle/phi/kernels/interpolate_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"

namespace phi {

template <typename T, typename Context>
void InterpolateKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  phi::funcs::ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_h = -1;
  float scale_w = -1;

  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::get_new_shape(size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  } else {
    if (scale_tensor) {
      auto scale_data =
          funcs::get_new_data_from_tensor<float>(scale_tensor.get_ptr());
      if (scale_data.size() > 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_h = scale_data[0];
        scale_w = scale_data[0];
      }
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    } else {
      if (scale.size() > 1) {
        scale_h = scale[0];
        scale_w = scale[1];

        PADDLE_ENFORCE_EQ(
            scale_w > 0,
            true,
            errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0,
            true,
            errors::InvalidArgument(
                "The scale_h in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      }
    }
    if (scale_h > 0. && scale_w > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }
    if (out_size) {
      auto out_size_data =
          funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }
  PADDLE_ENFORCE_GT(
      out_h,
      0,
      errors::InvalidArgument("out_h in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));
  PADDLE_ENFORCE_GT(
      out_w,
      0,
      errors::InvalidArgument("out_w in Attr(out_shape) of Op(interpolate) "
                              "should be greater than 0."));

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }
  output->Resize(dim_out);
  ctx.template Alloc<T>(output);

  if (in_h == out_h && in_w == out_w) {
    phi::Copy<Context>(ctx, x, ctx.GetPlace(), false, output);
    return;
  }
  bool nearest = "nearest" == interp_method;
  int trans_mode = (align_corners) ? (0) : ((align_mode == 0) ? (1) : (2));
  if (nearest) {
    trans_mode = (align_corners == true) ? 0 : 2;
    PADDLE_ENFORCE_EQ(
        (data_layout == DataLayout::kNCHW),
        true,
        errors::InvalidArgument("XPU nearest is only support NCHW"));
  }

  int r =
      xpu::interpolate2d<XPUType>(ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  reinterpret_cast<XPUType*>(output->data<T>()),
                                  n,
                                  c,
                                  in_h,
                                  in_w,
                                  out_h,
                                  out_w,
                                  nearest,
                                  trans_mode,
                                  (data_layout == DataLayout::kNCHW));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "interpolate2d");
}

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

}  // namespace phi

PD_REGISTER_KERNEL(bilinear_interp,
                   XPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpKernel,
                   phi::dtype::float16,
                   float) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_KERNEL(nearest_interp,
                   XPU,
                   ALL_LAYOUT,
                   phi::NearestInterpKernel,
                   phi::dtype::float16,
                   float,
                   int64_t) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
