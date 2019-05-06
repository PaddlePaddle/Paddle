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
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

class ReluCompute : public KernelLite<TARGET(kHost), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& param = Param<operators::ReluParam>();
    auto n = product(param.input->dims());
    const float* input = param.input->data<float>();
    float* output = TensorMutableData<float>(param.output, TARGET(kHost),
                                             product(param.output->dims()));
    for (int i = 0; i < n; i++) {
      output[i] = std::max(0.f, input[i]);
    }
  }

  TargetType target() const override { return TARGET(kHost); }
  PrecisionType precision() const override { return PRECISION(kFloat); }
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(relu, kHost, kFloat, kNCHW,
                     paddle::lite::kernels::host::ReluCompute, def)
    .Finalize();
