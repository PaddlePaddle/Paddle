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
                      const IntArray& kernel_size_t,
                      const std::vector<int>& strides_t,
                      const std::vector<int>& paddings_t,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      DenseTensor* dx) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  std::vector<int> paddings(paddings_t);
  std::vector<int> kernel_size(kernel_size_t.GetData().begin(),
                               kernel_size_t.GetData().end());
  std::vector<int> strides(strides_t);

  PADDLE_ENFORCE_EQ(
      data_format,
      "NCHW",
      common::errors::InvalidArgument("The Pool2d_grad XPU OP only support"
                                      "data_format is 'NCHW', but received %s",
                                      data_format));

  PADDLE_ENFORCE_EQ(
      kernel_size.size(),
      2,
      common::errors::InvalidArgument("The Pool2d XPU OP only support 2 "
                                      "dimension pooling!, but received "
                                      "%d-dimension pool kernel size",
                                      kernel_size.size()));
  if (global_pooling) {
    for (size_t i = 0; i < kernel_size.size(); ++i) {
      paddings[i] = 0;
      kernel_size[i] = static_cast<int>(x.dims()[i + 2]);
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

  ctx.template Alloc<T>(dx);
  const int* index_data = nullptr;
  int r = xpu::Error_t::SUCCESS;
  if (adaptive) {
    if (pooling_type == "max") {
      r = xpu::adaptive_max_pool2d_grad<XPUType>(
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
          out_h,
          out_w,
          true);

    } else if (pooling_type == "avg") {
      r = xpu::adaptive_avg_pool2d_grad<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(dout.data<T>()),
          reinterpret_cast<XPUType*>(dx->data<T>()),
          n,
          c,
          in_h,
          in_w,
          out_h,
          out_w,
          true);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adaptive_pool2d_grad");
  } else {
    if (kernel_size[0] > (in_h + paddings[0] + paddings[1])) {
      kernel_size[0] = in_h + paddings[0] + paddings[1];
    }
    if (kernel_size[1] > (in_w + paddings[2] + paddings[3])) {
      kernel_size[1] = in_w + paddings[2] + paddings[3];
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
          kernel_size,
          strides,
          paddings,
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
          kernel_size,
          strides,
          paddings,
          !exclusive,
          true);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "pool2dgrad");
  }
}

template <typename T, typename Context>
void Pool3dGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      const std::vector<int>& kernel_size_t,
                      const std::vector<int>& strides_t,
                      const std::vector<int>& paddings_t,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      DenseTensor* dx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto x_dims = x.dims();
  const bool channel_last = data_format == "NDHWC";

  std::vector<int> paddings(paddings_t);
  std::vector<int> kernel_size(kernel_size_t);
  std::vector<int> strides(strides_t);

  PADDLE_ENFORCE_EQ(
      data_format,
      "NCDHW",
      common::errors::InvalidArgument("The Pool3d_grad XPU OP only support"
                                      "data_format is 'NCDHW', but received %s",
                                      data_format));
  if (!dx) {
    return;
  }
  int n = x.dims()[0];
  int c = x.dims()[1];
  int in_d = x.dims()[2];
  int in_h = x.dims()[3];
  int in_w = x.dims()[4];

  int out_d = out.dims()[2];
  int out_h = out.dims()[3];
  int out_w = out.dims()[4];

  if (channel_last) {
    c = x.dims()[4];
    in_d = x.dims()[1];
    in_h = x.dims()[2];
    in_w = x.dims()[3];

    out_d = out.dims()[1];
    out_h = out.dims()[2];
    out_w = out.dims()[3];
  }

  DDim data_dims;
  if (channel_last) {
    data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  } else {
    data_dims = slice_ddim(x_dims, 2, x_dims.size());
  }
  funcs::UpdatePadding(&paddings,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size);

  if (global_pooling) {
    funcs::UpdateKernelSize(&kernel_size, data_dims);
  }

  ctx.template Alloc<T>(dx);
  const int* index_data = nullptr;
  int r = xpu::Error_t::SUCCESS;

  if (adaptive) {
    if (pooling_type == "max") {
      r = xpu::adaptive_max_pool3d_grad<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<const XPUType*>(out.data<T>()),
          index_data,
          reinterpret_cast<const XPUType*>(dout.data<T>()),
          reinterpret_cast<XPUType*>(dx->data<T>()),
          n,
          c,
          in_d,
          in_h,
          in_w,
          out_d,
          out_h,
          out_w,
          !channel_last);

    } else if (pooling_type == "avg") {
      if (out_d == 1 && out_h == 1 && out_w == 1 &&
          std::is_same<T, float>::value) {
        xpu::ctx_guard RAII_GUARD(ctx.x_context());
        float scale = 1.0 / (in_d * in_h * in_w);
        float* scaled_dy = RAII_GUARD.alloc_l3_or_gm<float>(n * c);
        r = xpu::scale(ctx.x_context(),
                       dout.data<float>(),
                       scaled_dy,
                       n * c,
                       true,
                       scale,
                       0.0);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

        r = xpu::broadcast(ctx.x_context(),
                           scaled_dy,
                           dx->data<float>(),
                           {n, c, 1, 1, 1},
                           {n, c, in_d, in_h, in_w});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

        return;
      }

      r = xpu::adaptive_avg_pool3d_grad<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(dout.data<T>()),
          reinterpret_cast<XPUType*>(dx->data<T>()),
          n,
          c,
          in_d,
          in_h,
          in_w,
          out_d,
          out_h,
          out_w,
          !channel_last);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adaptive_pool3d_grad");
  } else {
    if (pooling_type == "max") {
      if (kernel_size[0] == 1 && kernel_size.size() == 3 &&
          strides.size() == 3 && paddings.size() == 6) {
        r = xpu::max_pool2d_grad<XPUType>(
            ctx.x_context(),
            reinterpret_cast<const XPUType*>(x.data<T>()),
            reinterpret_cast<const XPUType*>(out.data<T>()),
            index_data,
            reinterpret_cast<const XPUType*>(dout.data<T>()),
            reinterpret_cast<XPUType*>(dx->data<T>()),
            n,
            c * in_d,
            in_h,
            in_w,
            {kernel_size[1], kernel_size[2]},
            {strides[1], strides[2]},
            {paddings[2], paddings[3], paddings[4], paddings[5]},
            !channel_last);
      } else {
        r = xpu::max_pool3d_grad<XPUType>(
            ctx.x_context(),
            reinterpret_cast<const XPUType*>(x.data<T>()),
            reinterpret_cast<const XPUType*>(out.data<T>()),
            index_data,
            reinterpret_cast<const XPUType*>(dout.data<T>()),
            reinterpret_cast<XPUType*>(dx->data<T>()),
            n,
            c,
            in_d,
            in_h,
            in_w,
            kernel_size,
            strides,
            paddings,
            !channel_last);
      }
    } else if (pooling_type == "avg") {
      r = xpu::avg_pool3d_grad<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(dout.data<T>()),
          reinterpret_cast<XPUType*>(dx->data<T>()),
          n,
          c,
          in_d,
          in_h,
          in_w,
          kernel_size,
          strides,
          paddings,
          !exclusive,
          !channel_last);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "pool3dgrad");
  }
}

template <typename T, typename Context>
void MaxPool2dWithIndexGradKernel(const Context& ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& mask,
                                  const DenseTensor& dout,
                                  const std::vector<int>& kernel_size,
                                  const std::vector<int>& strides_t,
                                  const std::vector<int>& paddings_t,
                                  bool global_pooling,
                                  bool adaptive,
                                  bool ceil_mode UNUSED,
                                  DenseTensor* dx) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  ctx.template Alloc<T>(dx);
  auto input_grad = reinterpret_cast<XPUType*>(dx->data<T>());
  std::vector<int> ksize(kernel_size);
  std::vector<int> strides(strides_t);
  std::vector<int> paddings(paddings_t);
  const auto* index_data = mask.data<int>();

  PADDLE_ENFORCE_NOT_NULL(index_data,
                          errors::NotFound("index data should not be nullptr"));
  PADDLE_ENFORCE_EQ(
      ksize.size(),
      2,
      common::errors::InvalidArgument("The Pool2d XPU OP only support 2 "
                                      "dimension pooling!, but received "
                                      "%d-dimension pool kernel size",
                                      ksize.size()));
  global_pooling = global_pooling || (adaptive && (ksize[0] * ksize[1] == 1));
  if (global_pooling) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[i] = 0;
      ksize[i] = static_cast<int>(dx->dims()[i + 2]);
    }
  }
  const int n = dx->dims()[0];
  const int c = dx->dims()[1];
  const int in_h = dx->dims()[2];
  const int in_w = dx->dims()[3];
  auto output_grad = reinterpret_cast<const XPUType*>(dout.data<T>());

  int r = xpu::Error_t::SUCCESS;
  // pass a nullptr as input to XDNN is fine as long as index_data exists
  r = xpu::max_pool2d_grad<XPUType>(ctx.x_context(),
                                    /*input*/ nullptr,
                                    /*output*/ nullptr,
                                    index_data,
                                    output_grad,
                                    input_grad,
                                    n,
                                    c,
                                    in_h,
                                    in_w,
                                    ksize,
                                    strides,
                                    paddings,
                                    true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "max_pool2d_with_index_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(pool2d_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::Pool2dGradKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(pool3d_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::Pool3dGradKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(max_pool2d_with_index_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaxPool2dWithIndexGradKernel,
                   float,
                   phi::dtype::float16) {}
