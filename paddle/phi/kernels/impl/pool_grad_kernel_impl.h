/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/pool_grad_kernel.h"
#include "paddle/phi/kernels/pool_kernel.h"

namespace phi {

template <typename T, typename Context>
void PoolGradRawKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& out,
                       const DenseTensor& dout,
                       const std::vector<int>& kernel_size,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       bool exclusive,
                       const std::string& data_format,
                       const std::string& pooling_type,
                       bool global_pooling,
                       bool adaptive,
                       const std::string& padding_algorithm,
                       DenseTensor* dx) {
  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  // update paddings
  auto x_dims = x.dims();
  DDim data_dims;
  if (channel_last) {
    data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  } else {
    data_dims = slice_ddim(x_dims, 2, x_dims.size());
  }
  funcs::UpdatePadding(&paddings_,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size_);
  if (data_dims.size() * 2 == static_cast<int>(paddings_.size())) {
    for (int i = 0; i < data_dims.size(); ++i) {
      paddings_.erase(paddings_.begin() + i + 1);
    }
  }

  if (global_pooling) {
    funcs::UpdateKernelSize(&kernel_size_, data_dims);
  }

  if (dx) {
    ctx.template Alloc<T>(dx);
    funcs::SetConstant<Context, T> set_constant;
    set_constant(ctx, dx, static_cast<T>(0.0));

    switch (kernel_size_.size()) {
      case 2: {
        if (pooling_type == "max") {
          funcs::MaxPool2dGradFunctor<Context, T> pool2d_backward;
          pool2d_backward(ctx,
                          x,
                          out,
                          dout,
                          kernel_size_,
                          strides,
                          paddings_,
                          data_format,
                          dx);
        } else if (pooling_type == "avg") {
          funcs::Pool2dGradFunctor<Context, funcs::AvgPoolGrad<T>, T>
              pool2d_backward;
          funcs::AvgPoolGrad<T> pool_process;
          pool2d_backward(ctx,
                          x,
                          out,
                          dout,
                          kernel_size_,
                          strides,
                          paddings_,
                          data_format,
                          exclusive,
                          adaptive,
                          dx,
                          pool_process);
        }
      } break;
      case 3: {
        if (pooling_type == "max") {
          funcs::MaxPool3dGradFunctor<Context, T> pool3d_backward;
          pool3d_backward(ctx,
                          x,
                          out,
                          dout,
                          kernel_size_,
                          strides,
                          paddings_,
                          data_format,
                          dx);
        } else if (pooling_type == "avg") {
          funcs::Pool3dGradFunctor<Context, funcs::AvgPoolGrad<T>, T>
              pool3d_backward;
          funcs::AvgPoolGrad<T> pool_process;
          pool3d_backward(ctx,
                          x,
                          out,
                          dout,
                          kernel_size_,
                          strides,
                          paddings_,
                          data_format,
                          exclusive,
                          adaptive,
                          dx,
                          pool_process);
        }
      } break;
      default: {
        PADDLE_THROW(
            errors::InvalidArgument("Pool op only supports 2D and 3D input."));
      }
    }
  }
}

template <typename Context, typename T1, typename T2 = int>
void MaxPoolWithIndexGradRawKernel(const Context& ctx,
                                   const DenseTensor& x,
                                   const DenseTensor& mask,
                                   const DenseTensor& dout,
                                   const std::vector<int>& kernel_size,
                                   const std::vector<int>& strides,
                                   const std::vector<int>& paddings,
                                   bool global_pooling,
                                   bool adaptive,
                                   DenseTensor* dx) {
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  if (global_pooling) {
    for (size_t i = 0; i < kernel_size_.size(); ++i) {
      paddings_[i] = 0;
      kernel_size_[i] = static_cast<int>(dx->dims()[i + 2]);
    }
  }

  if (dx) {
    ctx.template Alloc<T1>(dx);
    funcs::set_constant(ctx, dx, 0);

    switch (kernel_size_.size()) {
      case 2: {
        funcs::MaxPool2dWithIndexGradFunctor<Context, T1, T2> pool2d_backward;
        pool2d_backward(
            ctx, dout, mask, kernel_size_, strides, paddings_, adaptive, dx);
      } break;
      case 3: {
        funcs::MaxPool3dWithIndexGradFunctor<Context, T1, T2> pool3d_backward;
        pool3d_backward(
            ctx, dout, mask, kernel_size_, strides, paddings_, adaptive, dx);
      } break;
      default: {
        PADDLE_THROW(
            errors::InvalidArgument("Pool op only supports 2D and 3D input."));
      }
    }
  }
}

template <typename T, typename Context>
void Pool2dGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      const IntArray& kernel_size,
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
  std::vector<int> kernel_size_val(kernel_size.GetData().begin(),
                                   kernel_size.GetData().end());
  PoolGradRawKernel<T, Context>(ctx,
                                x,
                                out,
                                dout,
                                kernel_size_val,
                                strides,
                                paddings,
                                exclusive,
                                data_format,
                                pooling_type,
                                global_pooling,
                                adaptive,
                                padding_algorithm,
                                dx);
}

template <typename T, typename Context>
void Pool2dDoubleGradKernel(const Context& ctx,
                            const DenseTensor& x,
                            const IntArray& kernel_size,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            bool ceil_mode,
                            bool exclusive,
                            const std::string& data_format,
                            const std::string& pooling_type,
                            bool global_pooling,
                            bool adaptive,
                            const std::string& padding_algorithm,
                            DenseTensor* out) {
  if (pooling_type == "max") {
    PADDLE_THROW(
        errors::InvalidArgument("Pool op grad grad only supports avgpool."));
  } else {
    Pool2dKernel<T, Context>(ctx,
                             x,
                             kernel_size,
                             strides,
                             paddings,
                             ceil_mode,
                             exclusive,
                             data_format,
                             pooling_type,
                             global_pooling,
                             adaptive,
                             padding_algorithm,
                             out);
  }
}

template <typename T, typename Context>
void MaxPool2dWithIndexGradKernel(const Context& ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& mask,
                                  const DenseTensor& dout,
                                  const std::vector<int>& kernel_size,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  bool global_pooling,
                                  bool adaptive,
                                  DenseTensor* dx) {
  MaxPoolWithIndexGradRawKernel<Context, T>(ctx,
                                            x,
                                            mask,
                                            dout,
                                            kernel_size,
                                            strides,
                                            paddings,
                                            global_pooling,
                                            adaptive,
                                            dx);
}

template <typename T, typename Context>
void Pool3dGradKernel(const Context& ctx,
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
  PoolGradRawKernel<T, Context>(ctx,
                                x,
                                out,
                                dout,
                                kernel_size,
                                strides,
                                paddings,
                                exclusive,
                                data_format,
                                pooling_type,
                                global_pooling,
                                adaptive,
                                padding_algorithm,
                                dx);
}

template <typename T, typename Context>
void MaxPool3dWithIndexGradKernel(const Context& ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& mask,
                                  const DenseTensor& dout,
                                  const std::vector<int>& kernel_size,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  bool global_pooling,
                                  bool adaptive,
                                  DenseTensor* dx) {
  MaxPoolWithIndexGradRawKernel<Context, T>(ctx,
                                            x,
                                            mask,
                                            dout,
                                            kernel_size,
                                            strides,
                                            paddings,
                                            global_pooling,
                                            adaptive,
                                            dx);
}

}  // namespace phi
