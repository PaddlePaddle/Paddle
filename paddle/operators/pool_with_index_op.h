/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/pooling.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T1, typename T2>
class MaxPoolWithIndexKernel : public framework::OpKernel<T1> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    Tensor* mask = context.Output<Tensor>("Mask");

    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    if (context.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }

    switch (ksize.size()) {
      case 2: {
        paddle::operators::math::MaxPool2dWithIndexFunctor<Place, T1, T2>
            pool2d_forward;
        pool2d_forward(context.device_context(), *in_x, ksize, strides,
                       paddings, out, mask);
      } break;
      case 3: {
        paddle::operators::math::MaxPool3dWithIndexFunctor<Place, T1, T2>
            pool3d_forward;
        pool3d_forward(context.device_context(), *in_x, ksize, strides,
                       paddings, out, mask);
      } break;
      default: { PADDLE_THROW("Pool op only supports 2D and 3D input."); }
    }
  }
};

template <typename Place, typename T1, typename T2>
class MaxPoolWithIndexGradKernel : public framework::OpKernel<T1> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* mask = context.Input<Tensor>("Mask");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));

    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    if (context.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x_grad->dims()[i + 2]);
      }
    }

    if (in_x_grad) {
      in_x_grad->mutable_data<T1>(context.GetPlace());
      auto& device_ctx = context.device_context();
      math::set_constant(device_ctx, in_x_grad, 0);

      switch (ksize.size()) {
        case 2: {
          paddle::operators::math::MaxPool2dWithIndexGradFunctor<Place, T1, T2>
              pool2d_backward;
          pool2d_backward(device_ctx, *out_grad, *mask, ksize, strides,
                          paddings, in_x_grad);
        } break;
        case 3: {
          paddle::operators::math::MaxPool3dWithIndexGradFunctor<Place, T1, T2>
              pool3d_backward;
          pool3d_backward(device_ctx, *out_grad, *mask, ksize, strides,
                          paddings, in_x_grad);
        } break;
        default: { PADDLE_THROW("Pool op only supports 2D and 3D input."); }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
