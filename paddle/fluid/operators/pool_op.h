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
    bool exclusive = context.Attr<bool>("exclusive");
    bool adaptive = context.Attr<bool>("adaptive");
    if (context.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }
    auto& dev_ctx = context.template device_context<DeviceContext>();
    switch (ksize.size()) {
      case 2: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool2dFunctor<
              DeviceContext, paddle::operators::math::MaxPool<T>, T>
              pool2d_forward;
          paddle::operators::math::MaxPool<T> pool_process;
          pool2d_forward(dev_ctx, *in_x, ksize, strides, paddings, pool_process,
                         true, false, out);

        } else if (pooling_type == "avg") {
          paddle::operators::math::Pool2dFunctor<
              DeviceContext, paddle::operators::math::AvgPool<T>, T>
              pool2d_forward;
          paddle::operators::math::AvgPool<T> pool_process;
          pool2d_forward(dev_ctx, *in_x, ksize, strides, paddings, pool_process,
                         exclusive, adaptive, out);
        }
      } break;
      case 3: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool3dFunctor<
              DeviceContext, paddle::operators::math::MaxPool<T>, T>
              pool3d_forward;
          paddle::operators::math::MaxPool<T> pool_process;
          pool3d_forward(dev_ctx, *in_x, ksize, strides, paddings, pool_process,
                         true, false, out);
        } else if (pooling_type == "avg") {
          paddle::operators::math::Pool3dFunctor<
              DeviceContext, paddle::operators::math::AvgPool<T>, T>
              pool3d_forward;
          paddle::operators::math::AvgPool<T> pool_process;
          pool3d_forward(dev_ctx, *in_x, ksize, strides, paddings, pool_process,
                         exclusive, adaptive, out);
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

    if (context.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
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
                            paddings, in_x_grad);
          } else if (pooling_type == "avg") {
            paddle::operators::math::Pool2dGradFunctor<
                DeviceContext, paddle::operators::math::AvgPoolGrad<T>, T>
                pool2d_backward;
            paddle::operators::math::AvgPoolGrad<T> pool_process;
            pool2d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, pool_process, exclusive, adaptive,
                            in_x_grad);
          }
        } break;
        case 3: {
          if (pooling_type == "max") {
            paddle::operators::math::MaxPool3dGradFunctor<DeviceContext, T>
                pool3d_backward;
            pool3d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, in_x_grad);
          } else if (pooling_type == "avg") {
            paddle::operators::math::Pool3dGradFunctor<
                DeviceContext, paddle::operators::math::AvgPoolGrad<T>, T>
                pool3d_backward;
            paddle::operators::math::AvgPoolGrad<T> pool_process;
            pool3d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, pool_process, exclusive, adaptive,
                            in_x_grad);
          }
        } break;
        default: { PADDLE_THROW("Pool op only supports 2D and 3D input."); }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
