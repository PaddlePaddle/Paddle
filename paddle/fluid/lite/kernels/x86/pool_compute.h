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

#include <Eigen/Core>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/pooling.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class PoolCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::PoolParam;
  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    if (param.global_pooling) {
      for (size_t i = 0; i < param.ksize.size(); ++i) {
        param.paddings[i] = 0;
        param.ksize[i] = static_cast<int>(param.x->dims()[i + 2]);
      }
    }
    switch (param.ksize.size()) {
      case 2: {
        if (param.pooling_type == "max") {
          paddle::operators::math::Pool2dFunctor<
              platform::CPUDeviceContext, paddle::operators::math::MaxPool<T>,
              T>
              pool2d_forward;
          paddle::operators::math::MaxPool<T> pool_process;
          pool2d_forward(platform::CPUDeviceContext(), param.x->raw_tensor(),
                         param.ksize, param.strides, param.paddings,
                         pool_process, true, false,
                         &(param.output->raw_tensor()));
        } else if (param.pooling_type == "avg") {
          paddle::operators::math::Pool2dFunctor<
              platform::CPUDeviceContext, paddle::operators::math::AvgPool<T>,
              T>
              pool2d_forward;
          paddle::operators::math::AvgPool<T> pool_process;
          pool2d_forward(platform::CPUDeviceContext(), param.x->raw_tensor(),
                         param.ksize, param.strides, param.paddings,
                         pool_process, param.exclusive, param.adaptive,
                         &(param.output->raw_tensor()));
        }
      } break;
      case 3: {
      } break;
    }
  }
  virtual ~PoolCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
