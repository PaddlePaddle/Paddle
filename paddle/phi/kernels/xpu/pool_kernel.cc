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

#include "paddle/common/macros.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace phi {
template <typename T, typename Context>
void Pool2dKernel(const Context& ctx,
                  const DenseTensor& x,
                  const IntArray& kernel_size_t,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings_t,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  std::vector<int64_t> kernel_size(kernel_size_t.GetData().begin(),
                                   kernel_size_t.GetData().end());
  std::vector<int64_t> paddings(paddings_t.begin(), paddings_t.end());

  PADDLE_ENFORCE_EQ(kernel_size.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The Pool2d XPU OP only support 2 dimension pooling!"));

  // old model's data_format maybe AnyLayout
  PADDLE_ENFORCE_NE(
      data_format,
      "NHWC",
      common::errors::InvalidArgument("The Pool2d XPU OP does not support "
                                      "data_format is 'NHWC', but received %s",
                                      data_format));

  if (global_pooling) {
    for (size_t i = 0; i < kernel_size.size(); ++i) {
      paddings[i] = 0;
      kernel_size[i] = x.dims()[i + 2];
    }
  }

  const int64_t n = x.dims()[0];
  const int64_t c = x.dims()[1];
  const int64_t in_h = x.dims()[2];
  const int64_t in_w = x.dims()[3];

  const int64_t out_h = out->dims()[2];
  const int64_t out_w = out->dims()[3];

  DDim data_dims;

  data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  funcs::UpdatePadding(&paddings,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size);

  ctx.template Alloc<T>(out);
  int* index_data = nullptr;
  int r = 0;
  if (!adaptive) {
    if (kernel_size[0] > (in_h + paddings[0] + paddings[1])) {
      kernel_size[0] = in_h + paddings[0] + paddings[1];
    }
    if (kernel_size[1] > (in_w + paddings[2] + paddings[3])) {
      kernel_size[1] = in_w + paddings[2] + paddings[3];
    }
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
      PADDLE_THROW(common::errors::InvalidArgument(
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
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pool2d");
}

template <typename T, typename Context>
void Pool3dKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& kernel_size_t,
                  const std::vector<int64_t>& strides,
                  const std::vector<int64_t>& paddings_t,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const bool channel_last = data_format == "NDHWC";
  std::vector<int64_t> kernel_size(kernel_size_t.begin(), kernel_size_t.end());
  std::vector<int64_t> paddings(paddings_t.begin(), paddings_t.end());

  auto x_dims = x.dims();
  int64_t n = x.dims()[0];
  int64_t c = x.dims()[1];
  int64_t in_d = x.dims()[2];
  int64_t in_h = x.dims()[3];
  int64_t in_w = x.dims()[4];

  int64_t out_d = out->dims()[2];
  int64_t out_h = out->dims()[3];
  int64_t out_w = out->dims()[4];

  if (data_format == "NDHWC") {
    c = x.dims()[4];
    in_d = x.dims()[1];
    in_h = x.dims()[2];
    in_w = x.dims()[3];

    out_d = out->dims()[1];
    out_h = out->dims()[2];
    out_w = out->dims()[3];
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

  ctx.template Alloc<T>(out);
  int* index_data = nullptr;
  int r = 0;
  if (!adaptive) {
    if (pooling_type == "max") {
      r = xpu::max_pool3d<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          index_data,
          n,
          c,
          in_d,
          in_h,
          in_w,
          kernel_size,
          strides,
          paddings,
          data_format == "NCDHW");
    } else if (pooling_type == "avg") {
      r = xpu::avg_pool3d<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          n,
          c,
          in_d,
          in_h,
          in_w,
          kernel_size,
          strides,
          paddings,
          !exclusive,
          data_format == "NCDHW");
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
  } else {
    if (pooling_type == "max") {
      r = xpu::adaptive_max_pool3d<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          index_data,
          n,
          c,
          in_d,
          in_h,
          in_w,
          out_d,
          out_h,
          out_w,
          data_format == "NCDHW");
    } else if (pooling_type == "avg") {
      r = xpu::adaptive_avg_pool3d<XPUType>(
          ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          n,
          c,
          in_d,
          in_h,
          in_w,
          out_d,
          out_h,
          out_w,
          data_format == "NCDHW");
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pool3d");
}

template <typename T, typename Context>
void MaxPool2dWithIndexKernel(const Context& ctx,
                              const DenseTensor& x,
                              const std::vector<int>& kernel_size_t,
                              const std::vector<int>& strides_t,
                              const std::vector<int>& paddings_t,
                              bool global_pooling,
                              bool adaptive,
                              bool ceil_mode UNUSED,
                              DenseTensor* out,
                              DenseTensor* mask) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  ctx.template Alloc<int>(mask);
  auto* index_data = mask->data<int>();

  std::vector<int64_t> kernel_size(kernel_size_t.begin(), kernel_size_t.end());
  std::vector<int64_t> strides(strides_t.begin(), strides_t.end());
  std::vector<int64_t> paddings(paddings_t.begin(), paddings_t.end());

  PADDLE_ENFORCE_EQ(kernel_size.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The Pool2d XPU OP only support 2 dimension pooling!"));
  PADDLE_ENFORCE_EQ(!adaptive || (kernel_size[0] * kernel_size[1] == 1),
                    true,
                    common::errors::InvalidArgument(
                        "The Pool2d XPU OP does not support (adaptive == "
                        "true && output_size != 1)"));
  global_pooling =
      global_pooling || (adaptive && (kernel_size[0] * kernel_size[1] == 1));
  if (global_pooling) {
    for (size_t i = 0; i < kernel_size.size(); ++i) {
      paddings[i] = 0;
      kernel_size[i] = x.dims()[i + 2];
    }
  }
  const int64_t n = x.dims()[0];
  const int64_t c = x.dims()[1];
  const int64_t in_h = x.dims()[2];
  const int64_t in_w = x.dims()[3];
  auto input = reinterpret_cast<const XPUType*>(x.data<T>());
  ctx.template Alloc<T>(out);
  auto output = reinterpret_cast<XPUType*>(out->data<T>());
  int r = 0;
  r = xpu::max_pool2d<XPUType>(ctx.x_context(),
                               input,
                               output,
                               index_data,
                               n,
                               c,
                               in_h,
                               in_w,
                               kernel_size,
                               strides,
                               paddings,
                               true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "max_pool2d_with_index");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    pool2d, XPU, ALL_LAYOUT, phi::Pool2dKernel, float, phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    pool3d, XPU, ALL_LAYOUT, phi::Pool3dKernel, float, phi::dtype::float16) {}

PD_REGISTER_KERNEL(max_pool2d_with_index,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaxPool2dWithIndexKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
}
