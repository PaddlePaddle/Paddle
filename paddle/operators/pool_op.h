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

template <typename Place, typename T>
class PoolKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_X = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    int global_pooling = context.Attr<int>("globalPooling");
    std::string pooling_type = context.Attr<std::string>("poolingType");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    if (global_pooling == 1) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        ksize[i] = in_X->dims()[i + 2];
      }
    }

    switch (ksize.size()) {
      case 2: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool2dForwardFunctor<
              Place, paddle::operators::math::pool::maxPool<T>, T>
              pool2d_forward;
          paddle::operators::math::pool::maxPool<T> pool_process;
          pool2d_forward(context.device_context(), *in_X, *out, ksize, strides,
                         paddings, pool_process);

        } else if (pooling_type == "ave") {
          paddle::operators::math::Pool2dForwardFunctor<
              Place, paddle::operators::math::pool::avePool<T>, T>
              pool2d_forward;
          paddle::operators::math::pool::avePool<T> pool_process;
          pool2d_forward(context.device_context(), *in_X, *out, ksize, strides,
                         paddings, pool_process);
        }
      } break;
      case 3: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool3dForwardFunctor<
              Place, paddle::operators::math::pool::maxPool<T>, T>
              pool3d_forward;
          paddle::operators::math::pool::maxPool<T> pool_process;
          pool3d_forward(context.device_context(), *in_X, *out, ksize, strides,
                         paddings, pool_process);
        } else if (pooling_type == "ave") {
          paddle::operators::math::Pool3dForwardFunctor<
              Place, paddle::operators::math::pool::avePool<T>, T>
              pool3d_forward;
          paddle::operators::math::pool::avePool<T> pool_process;
          pool3d_forward(context.device_context(), *in_X, *out, ksize, strides,
                         paddings, pool_process);
        }
      } break;
    }
  }
};

template <typename Place, typename T>
class PoolGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_X = context.Input<Tensor>("X");
    const Tensor* out = context.Input<Tensor>("Out");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_X_grad = context.Output<Tensor>(framework::GradVarName("X"));

    int global_pooling = context.Attr<int>("globalPooling");
    std::string pooling_type = context.Attr<std::string>("poolingType");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    if (global_pooling == 1) {
      for (size_t i = 0; i < ksize.size(); ++i) ksize[i] = in_X->dims()[i + 2];
    }

    if (in_X_grad) {
      in_X_grad->mutable_data<T>(context.GetPlace());
      auto temp = framework::EigenVector<T>::Flatten(*in_X_grad);
      temp.device(context.GetEigenDevice<Place>()) =
          temp.constant(static_cast<T>(0));

      switch (ksize.size()) {
        case 2: {
          if (pooling_type == "max") {
            paddle::operators::math::Pool2dBackwardFunctor<
                Place, paddle::operators::math::pool::maxPool<T>, T>
                pool2d_backward;
            paddle::operators::math::pool::maxPool<T> pool_process;
            pool2d_backward(context.device_context(), *in_X, *in_X_grad, *out,
                            *out_grad, ksize, strides, paddings, pool_process);
          } else if (pooling_type == "ave") {
            paddle::operators::math::Pool2dBackwardFunctor<
                Place, paddle::operators::math::pool::avePool<T>, T>
                pool2d_backward;
            paddle::operators::math::pool::avePool<T> pool_process;
            pool2d_backward(context.device_context(), *in_X, *in_X_grad, *out,
                            *out_grad, ksize, strides, paddings, pool_process);
          }
        } break;
        case 3: {
          if (pooling_type == "max") {
            paddle::operators::math::Pool3dBackwardFunctor<
                Place, paddle::operators::math::pool::maxPool<T>, T>
                pool3d_backward;
            paddle::operators::math::pool::maxPool<T> pool_process;
            pool3d_backward(context.device_context(), *in_X, *in_X_grad, *out,
                            *out_grad, ksize, strides, paddings, pool_process);
          } else if (pooling_type == "ave") {
            paddle::operators::math::Pool3dBackwardFunctor<
                Place, paddle::operators::math::pool::avePool<T>, T>
                pool3d_backward;
            paddle::operators::math::pool::avePool<T> pool_process;
            pool3d_backward(context.device_context(), *in_X, *in_X_grad, *out,
                            *out_grad, ksize, strides, paddings, pool_process);
          }
        } break;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
