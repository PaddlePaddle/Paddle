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
#include <algorithm>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class ReluCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& param = Param<operators::ReluParam>();
    auto n = param.input->dims().production();
    const float* input = param.input->data<float>();
    float* output = param.output->mutable_data<float>();
    for (int i = 0; i < n; i++) {
      output[i] = std::max(0.f, input[i]);
    }
  }

  TargetType target() const override { return TARGET(kARM); }
  PrecisionType precision() const override { return PRECISION(kFloat); }
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(relu, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::ReluCompute, def)
    .Finalize();
