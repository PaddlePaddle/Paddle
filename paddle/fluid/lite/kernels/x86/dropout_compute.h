// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <random>
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
class DropoutCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::DropoutParam;
  void Run() override {
    auto& param = *param_.get_mutable<operators::DropoutParam>();
    const auto* x_data = param.x->data<T>();
    auto* out_data = param.output->template mutable_data<T>();
    if (!param.is_test) {
      auto* mask_data = param.mask->template mutable_data<T>();
      std::random_device rnd;
      std::minstd_rand engine;
      int seed = param.fix_seed ? param.seed : rnd();
      engine.seed(seed);
      std::uniform_real_distribution<float> dist(0, 1);

      size_t size = framework::product(param.mask->dims().data());
      for (size_t i = 0; i < size; ++i) {
        if (dist(engine) < param.dropout_prob) {
          mask_data[i] = 0;
          out_data[i] = 0;
        } else {
          if (param.dropout_implementation == "upscale_in_train") {
            mask_data[i] = 1.0f / static_cast<T>(1.0f - param.dropout_prob);
            out_data[i] = x_data[i] / static_cast<T>(1.0f - param.dropout_prob);
          } else {
            mask_data[i] = 1;
            out_data[i] = x_data[i];
          }
        }
      }
    } else {
      auto X = EigenMatrix<T>::Reshape(param.x->raw_tensor(), 1);
      auto Y = EigenMatrix<T>::Reshape(param.output->raw_tensor(), 1);
      auto& place = *platform::CPUDeviceContext().eigen_device();
      if (param.dropout_implementation == "upscale_in_train") {
        Y.device(place) = X;
      } else {
        Y.device(place) = X * static_cast<T>(1.0f - param.dropout_prob);
      }
    }
  }

  virtual ~DropoutCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
