/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/pooling.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class PoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class PoolOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class Pool2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class Pool3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};
inline void UpdatePadding(std::vector<int>* paddings, const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const framework::DDim data_dims,
                          const std::vector<int>& strides,
                          const std::vector<int>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = framework::vectorize<int>(data_dims);
  if (paddings->size() == data_dims.size()) {
    for (size_t i = 0; i < data_dims.size(); ++i) {
      int copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        data_dims.size() * 2, paddings->size(),
        "Paddings size should be the same or twice as the pooling size.");
  }

  // when padding_desc is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (size_t i = 0; i < data_dims.size(); ++i) {
      int out_size = (data_dims[i] + strides[i] - 1) / strides[0];
      int pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i], 0);
      int pad_0 = pad_sum / 2;
      int pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

inline void UpdateKsize(std::vector<int>* ksize,
                        const framework::DDim data_dims) {
  ksize->resize(static_cast<size_t>(data_dims.size()));
  for (size_t i = 0; i < ksize->size(); ++i) {
    *(ksize->begin() + i) = static_cast<int>(data_dims[i]);
  }
}

template <typename DeviceContext, typename T>
class PoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::string data_format = context.Attr<std::string>("data_format");
    bool exclusive = context.Attr<bool>("exclusive");
    bool adaptive = context.Attr<bool>("adaptive");
    bool global_pooling = context.Attr<bool>("global_pooling");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // update paddings
    auto in_x_dims = in_x->dims();
    framework::DDim data_dims;
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
    }

    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    if (data_dims.size() * 2 == paddings.size()) {
      for (size_t i = 0; i < data_dims.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }

    if (global_pooling) {
      UpdateKsize(&ksize, data_dims);
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    switch (ksize.size()) {
      case 2: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool2dFunctor<
              DeviceContext, paddle::operators::math::MaxPool<T>, T>
              pool2d_forward;
          paddle::operators::math::MaxPool<T> pool_process;
          pool2d_forward(dev_ctx, *in_x, ksize, strides, paddings, data_format,
                         pool_process, true, false, out);

        } else if (pooling_type == "avg") {
          paddle::operators::math::Pool2dFunctor<
              DeviceContext, paddle::operators::math::AvgPool<T>, T>
              pool2d_forward;
          paddle::operators::math::AvgPool<T> pool_process;
          pool2d_forward(dev_ctx, *in_x, ksize, strides, paddings, data_format,
                         pool_process, exclusive, adaptive, out);
        }
      } break;
      case 3: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool3dFunctor<
              DeviceContext, paddle::operators::math::MaxPool<T>, T>
              pool3d_forward;
          paddle::operators::math::MaxPool<T> pool_process;
          pool3d_forward(dev_ctx, *in_x, ksize, strides, paddings, data_format,
                         pool_process, true, false, out);

        } else if (pooling_type == "avg") {
          paddle::operators::math::Pool3dFunctor<
              DeviceContext, paddle::operators::math::AvgPool<T>, T>
              pool3d_forward;
          paddle::operators::math::AvgPool<T> pool_process;
          pool3d_forward(dev_ctx, *in_x, ksize, strides, paddings, data_format,
                         pool_process, exclusive, adaptive, out);
        }
      } break;
      default: { PADDLE_THROW("Pool op only supports 2D and 3D input."); }
    }
  }
};

template <typename DeviceContext, typename T>
class PoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    const Tensor* out = context.Input<Tensor>("Out");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));

    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    bool exclusive = context.Attr<bool>("exclusive");
    bool adaptive = context.Attr<bool>("adaptive");
    std::string data_format = context.Attr<std::string>("data_format");
    bool global_pooling = context.Attr<bool>("global_pooling");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // update paddings
    auto in_x_dims = in_x->dims();
    framework::DDim data_dims;
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
    }
    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    if (data_dims.size() * 2 == paddings.size()) {
      for (size_t i = 0; i < data_dims.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }

    if (global_pooling) {
      UpdateKsize(&ksize, data_dims);
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (in_x_grad) {
      in_x_grad->mutable_data<T>(context.GetPlace());
      paddle::operators::math::SetConstant<DeviceContext, T> set_constant;
      set_constant(dev_ctx, in_x_grad, 0.0);

      switch (ksize.size()) {
        case 2: {
          if (pooling_type == "max") {
            paddle::operators::math::MaxPool2dGradFunctor<DeviceContext, T>
                pool2d_backward;
            pool2d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, data_format, in_x_grad);
          } else if (pooling_type == "avg") {
            paddle::operators::math::Pool2dGradFunctor<
                DeviceContext, paddle::operators::math::AvgPoolGrad<T>, T>
                pool2d_backward;
            paddle::operators::math::AvgPoolGrad<T> pool_process;
            pool2d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, data_format, pool_process, exclusive,
                            adaptive, in_x_grad);
          }
        } break;
        case 3: {
          if (pooling_type == "max") {
            paddle::operators::math::MaxPool3dGradFunctor<DeviceContext, T>
                pool3d_backward;
            pool3d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, data_format, in_x_grad);
          } else if (pooling_type == "avg") {
            paddle::operators::math::Pool3dGradFunctor<
                DeviceContext, paddle::operators::math::AvgPoolGrad<T>, T>
                pool3d_backward;
            paddle::operators::math::AvgPoolGrad<T> pool_process;
            pool3d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, data_format, pool_process, exclusive,
                            adaptive, in_x_grad);
          }
        } break;
        default: { PADDLE_THROW("Pool op only supports 2D and 3D input."); }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
