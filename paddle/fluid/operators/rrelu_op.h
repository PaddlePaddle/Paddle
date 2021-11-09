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

#include <cstring>
#include <random>
#include <string>

#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class CPURReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* seed =
        context.HasInput("Seed") ? context.Input<Tensor>("Seed") : nullptr;
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    auto* mask = context.Output<Tensor>("Mask");
    auto* mask_data = mask->mutable_data<T>(context.GetPlace());
    size_t size = framework::product(mask->dims());
    float lower_bound = context.Attr<float>("lower_bound");
    float upper_bound = context.Attr<float>("upper_bound");

    if (!context.Attr<bool>("is_test")) {
      // std::minstd_rand engine;
      // NOTE: fixed seed should only be used in unittest or for debug.
      // Guarantee to use random seed in training.
      int seed_data = 0;
      if (seed) {
        seed_data = *(seed->data<int>());
      } else {
        seed_data =
            context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : 0;
      }
      auto engine = framework::GetCPURandomEngine(seed_data);

      std::uniform_real_distribution<float> dist(lower_bound, upper_bound);

      for (size_t i = 0; i < size; ++i) {
        if (x_data[i] >= static_cast<T>(0.0f)) {
          mask_data[i] = static_cast<T>(1.0f);
          y_data[i] = x_data[i];
        } else {
          mask_data[i] = static_cast<T>(dist(*engine));
          y_data[i] = x_data[i] * mask_data[i];
        }
      }
    } else {
      const auto* X_data = x->data<T>();
      auto* Y_data = y->mutable_data<T>(context.GetPlace());
      auto middle_value = static_cast<T>((lower_bound + upper_bound) / 2.0);
      for (int i = 0; i < x->numel(); i++) {
        if (X_data[i] >= static_cast<T>(0.0f)) {
          mask_data[i] = static_cast<T>(1.0f);
          Y_data[i] = X_data[i];
        } else {
          mask_data[i] = middle_value;
          Y_data[i] = X_data[i] * middle_value;
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class RReluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());

    auto dX = EigenVector<T>::Flatten(*grad_x);
    auto dY = EigenVector<T>::Flatten(*grad_y);
    auto M = EigenVector<T>::Flatten(*mask);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    dX.device(place) = dY * M;
  }
};

}  // namespace operators
}  // namespace paddle
