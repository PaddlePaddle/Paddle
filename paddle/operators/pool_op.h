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
    const Tensor* input = context.Input<Tensor>("Input");
    Tensor* output = context.Output<Tensor>("Output");

    int global_pooling = context.Attr<int>("global_pooling");
    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    if (global_pooling == 1) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        ksize[i] = input->dims()[i + 2];
      }
    }
    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);

    switch (ksize.size()) {
      case 2: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool2dForwardFunctor<
              Place, paddle::operators::math::pool::maxPool<T>, T>
              pool2d_forward;
          paddle::operators::math::pool::maxPool<T> pool_process;
          pool2d_forward(*input, *output, ksize, strides, paddings,
                         pool_process, device_context);

        } else if (pooling_type == "ave") {
          paddle::operators::math::Pool2dForwardFunctor<
              Place, paddle::operators::math::pool::avePool<T>, T>
              pool2d_forward;
          paddle::operators::math::pool::avePool<T> pool_process;
          pool2d_forward(*input, *output, ksize, strides, paddings,
                         pool_process, device_context);
        }
      } break;
      case 3: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool3dForwardFunctor<
              Place, paddle::operators::math::pool::maxPool<T>, T>
              pool3d_forward;
          paddle::operators::math::pool::maxPool<T> pool_process;
          pool3d_forward(*input, *output, ksize, strides, paddings,
                         pool_process, device_context);
        } else if (pooling_type == "ave") {
          paddle::operators::math::Pool3dForwardFunctor<
              Place, paddle::operators::math::pool::avePool<T>, T>
              pool3d_forward;
          paddle::operators::math::pool::avePool<T> pool_process;
          pool3d_forward(*input, *output, ksize, strides, paddings,
                         pool_process, device_context);
        }
      } break;
    }
  }
};

template <typename Place, typename T>
class PoolGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output = context.Input<Tensor>("Output");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<framework::LoDTensor>(framework::GradVarName("Input"));

    int global_pooling = context.Attr<int>("global_pooling");
    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    if (global_pooling == 1) {
      for (size_t i = 0; i < ksize.size(); ++i) ksize[i] = input->dims()[i + 2];
    }
    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      auto temp = framework::EigenVector<T>::Flatten(*input_grad);
      temp.device(context.GetEigenDevice<Place>()) =
          temp.constant(static_cast<T>(0));

      switch (ksize.size()) {
        case 2: {
          if (pooling_type == "max") {
            paddle::operators::math::Pool2dBackwardFunctor<
                Place, paddle::operators::math::pool::maxPool<T>, T>
                pool2d_backward;
            paddle::operators::math::pool::maxPool<T> pool_process;
            pool2d_backward(*input, *input_grad, *output, *output_grad, ksize,
                            strides, paddings, pool_process, device_context);
          } else if (pooling_type == "ave") {
            paddle::operators::math::Pool2dBackwardFunctor<
                Place, paddle::operators::math::pool::avePool<T>, T>
                pool2d_backward;
            paddle::operators::math::pool::avePool<T> pool_process;
            pool2d_backward(*input, *input_grad, *output, *output_grad, ksize,
                            strides, paddings, pool_process, device_context);
          }
        } break;
        case 3: {
          if (pooling_type == "max") {
            paddle::operators::math::Pool3dBackwardFunctor<
                Place, paddle::operators::math::pool::maxPool<T>, T>
                pool3d_backward;
            paddle::operators::math::pool::maxPool<T> pool_process;
            pool3d_backward(*input, *input_grad, *output, *output_grad, ksize,
                            strides, paddings, pool_process, device_context);
          } else if (pooling_type == "ave") {
            paddle::operators::math::Pool3dBackwardFunctor<
                Place, paddle::operators::math::pool::avePool<T>, T>
                pool3d_backward;
            paddle::operators::math::pool::avePool<T> pool_process;
            pool3d_backward(*input, *input_grad, *output, *output_grad, ksize,
                            strides, paddings, pool_process, device_context);
          }
        } break;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
