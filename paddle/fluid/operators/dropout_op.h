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

#include <random>
#include <string>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class CPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    auto dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<T>(context.GetPlace());

      // NOTE: fixed seed should only be used in unittest or for debug.
      // Guarantee to use random seed in training.
      std::random_device rnd;
      std::minstd_rand engine;
      int seed =
          context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
      engine.seed(seed);

      std::uniform_real_distribution<float> dist(0, 1);

      size_t size = framework::product(mask->dims());
      for (size_t i = 0; i < size; ++i) {
        if (dist(engine) < dropout_prob) {
          mask_data[i] = 0;
          y_data[i] = 0;
        } else {
          if (dropout_implementation == "upscale_in_train") {
            mask_data[i] = 1.0f / static_cast<T>(1.0f - dropout_prob);
            y_data[i] = x_data[i] / static_cast<T>(1.0f - dropout_prob);
          } else {
            mask_data[i] = 1;
            y_data[i] = x_data[i];
          }
        }
      }
    } else {
      auto X = EigenMatrix<T>::Reshape(*x, 1);
      auto Y = EigenMatrix<T>::Reshape(*y, 1);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      if (dropout_implementation == "upscale_in_train") {
        Y.device(place) = X;
      } else {
        Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class DropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(!context.Attr<bool>("is_test"),
                   "GradOp is only callable when is_test is false");

    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());

    auto M = EigenMatrix<T>::Reshape(*mask, 1);
    auto dX = EigenMatrix<T>::Reshape(*grad_x, 1);
    auto dY = EigenMatrix<T>::Reshape(*grad_y, 1);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    dX.device(place) = dY * M;
  }
};

}  // namespace operators
}  // namespace paddle
