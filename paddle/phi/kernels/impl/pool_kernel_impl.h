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

#include <algorithm>

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/pool_kernel.h"

#if defined(__HIPCC__) || defined(__NVCC__)
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#endif

namespace phi {

inline int GetReduceNum(const DenseTensor& input,
                        const DenseTensor* output,
                        const bool channel_last,
                        std::vector<int>* reduce_dim) {
  int reduce_num = 0;
  const int output_height =
      channel_last ? output->dims()[1] : output->dims()[2];
  const int output_width = channel_last ? output->dims()[2] : output->dims()[3];
  if ((output_height == 1) && (output_width == 1)) {
    if (channel_last) {
      reduce_dim->push_back(1);
      reduce_dim->push_back(2);
      reduce_num = input.dims()[1] * input.dims()[2];
    } else {
      reduce_dim->push_back(2);
      reduce_dim->push_back(3);
      reduce_num = input.dims()[2] * input.dims()[3];
    }
  }
  return reduce_num;
}

template <typename T, typename Context>
void PoolRawKernel(const Context& ctx,
                   const DenseTensor& x,
                   const std::vector<int>& kernel_size,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool exclusive,
                   const std::string& data_format,
                   const std::string& pooling_type,
                   bool global_pooling,
                   bool adaptive,
                   const std::string& padding_algorithm,
                   DenseTensor* out) {
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

  switch (kernel_size_.size()) {
    case 2: {
      if (pooling_type == "max") {
        funcs::Pool2dFunctor<Context, funcs::MaxPool<T>, T> pool2d_forward;
        funcs::MaxPool<T> pool_process;
        pool2d_forward(ctx,
                       x,
                       kernel_size_,
                       strides,
                       paddings_,
                       data_format,
                       true,
                       false,
                       out,
                       pool_process);

      } else if (pooling_type == "avg") {
        std::vector<int> reduce_dim;
        int reduce_num = GetReduceNum(x, out, channel_last, &reduce_dim);
        if (reduce_num > 0 &&
            adaptive) {  // for adaptive_avg_pool2d && output_size == 1
#if defined(__HIPCC__) || defined(__NVCC__)
          auto stream = ctx.stream();
          funcs::ReduceKernel<T, T, kps::AddFunctor, kps::DivideFunctor<T>>(
              ctx, x, out, kps::DivideFunctor<T>(reduce_num), reduce_dim);
#else  // for cpu
          funcs::Pool2dFunctor<Context, funcs::AvgPool<T>, T> pool2d_forward;
          funcs::AvgPool<T> pool_process;
          pool2d_forward(ctx,
                         x,
                         kernel_size_,
                         strides,
                         paddings_,
                         data_format,
                         exclusive,
                         adaptive,
                         out,
                         pool_process);
#endif
        } else {  // avgpool_2d or  adaptive_avg_pool2d && output_size != 1
          funcs::Pool2dFunctor<Context, funcs::AvgPool<T>, T> pool2d_forward;
          funcs::AvgPool<T> pool_process;
          pool2d_forward(ctx,
                         x,
                         kernel_size_,
                         strides,
                         paddings_,
                         data_format,
                         exclusive,
                         adaptive,
                         out,
                         pool_process);
        }
      }
    } break;
    case 3: {
      if (pooling_type == "max") {
        funcs::Pool3dFunctor<Context, funcs::MaxPool<T>, T> pool3d_forward;
        funcs::MaxPool<T> pool_process;
        pool3d_forward(ctx,
                       x,
                       kernel_size_,
                       strides,
                       paddings_,
                       data_format,
                       true,
                       false,
                       out,
                       pool_process);
      } else if (pooling_type == "avg") {
        funcs::Pool3dFunctor<Context, funcs::AvgPool<T>, T> pool3d_forward;
        funcs::AvgPool<T> pool_process;
        pool3d_forward(ctx,
                       x,
                       kernel_size_,
                       strides,
                       paddings_,
                       data_format,
                       exclusive,
                       adaptive,
                       out,
                       pool_process);
      }
    } break;
    default: {
      PADDLE_THROW(
          errors::InvalidArgument("Pool op only supports 2D and 3D input."));
    }
  }
}

template <typename Context, typename T1, typename T2 = int>
void MaxPoolWithIndexRawKernel(const Context& ctx,
                               const DenseTensor& x,
                               const std::vector<int>& kernel_size,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               bool global_pooling,
                               bool adaptive,
                               DenseTensor* out,
                               DenseTensor* mask) {
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  if (global_pooling) {
    for (size_t i = 0; i < kernel_size_.size(); ++i) {
      paddings_[i] = 0;
      kernel_size_[i] = static_cast<int>(x.dims()[i + 2]);
    }
  }

  switch (kernel_size_.size()) {
    case 2: {
      funcs::MaxPool2dWithIndexFunctor<Context, T1, T2> pool2d_forward;
      pool2d_forward(
          ctx, x, kernel_size_, strides, paddings_, adaptive, out, mask);
    } break;
    case 3: {
      funcs::MaxPool3dWithIndexFunctor<Context, T1, T2> pool3d_forward;
      pool3d_forward(
          ctx, x, kernel_size_, strides, paddings_, adaptive, out, mask);
    } break;
    default: {
      PADDLE_THROW(
          errors::InvalidArgument("Pool op only supports 2D and 3D input."));
    }
  }
}

template <typename T, typename Context>
void Pool2dKernel(const Context& ctx,
                  const DenseTensor& x,
                  const IntArray& kernel_size,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool ceil_mode UNUSED,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out) {
  std::vector<int> kernel_size_val(kernel_size.GetData().begin(),
                                   kernel_size.GetData().end());
  PoolRawKernel<T, Context>(ctx,
                            x,
                            kernel_size_val,
                            strides,
                            paddings,
                            exclusive,
                            data_format,
                            pooling_type,
                            global_pooling,
                            adaptive,
                            padding_algorithm,
                            out);
}

template <typename T, typename Context>
void MaxPool2dWithIndexKernel(const Context& ctx,
                              const DenseTensor& x,
                              const std::vector<int>& kernel_size,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              bool global_pooling,
                              bool adaptive,
                              DenseTensor* out,
                              DenseTensor* mask) {
  MaxPoolWithIndexRawKernel<Context, T>(ctx,
                                        x,
                                        kernel_size,
                                        strides,
                                        paddings,
                                        global_pooling,
                                        adaptive,
                                        out,
                                        mask);
}

template <typename T, typename Context>
void Pool3dKernel(const Context& ctx,
                  const DenseTensor& x,
                  const std::vector<int>& kernel_size,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool ceil_mode UNUSED,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out) {
  PoolRawKernel<T, Context>(ctx,
                            x,
                            kernel_size,
                            strides,
                            paddings,
                            exclusive,
                            data_format,
                            pooling_type,
                            global_pooling,
                            adaptive,
                            padding_algorithm,
                            out);
}

template <typename T, typename Context>
void MaxPool3dWithIndexKernel(const Context& ctx,
                              const DenseTensor& x,
                              const std::vector<int>& kernel_size,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              bool global_pooling,
                              bool adaptive,
                              DenseTensor* out,
                              DenseTensor* mask) {
  MaxPoolWithIndexRawKernel<Context, T>(ctx,
                                        x,
                                        kernel_size,
                                        strides,
                                        paddings,
                                        global_pooling,
                                        adaptive,
                                        out,
                                        mask);
}

}  // namespace phi
