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
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"
#include "paddle/fluid/lite/operators/relu_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
void scale_compute(const T* x, T* out, int size, float scale, float bias,
                   bool bias_before) {
  if (bias_before) bias *= scale;
  for (int i = 0; i < size; i++) {
    out[i] = x[i] * scale + bias;
  }
}

template <typename T>
class ScaleCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ScaleParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    scale_compute(param.x->data<T>(), param.output->mutable_data<T>(),
                  param.x->dims().production(), param.scale, param.bias,
                  param.bias_after_scale);
  }

  virtual ~ScaleCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
