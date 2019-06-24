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

#include "paddle/fluid/lite/kernels/arm/dropout_compute.h"
#include <string>
#include "paddle/fluid/lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void DropoutCompute::Run() {
  auto& param = Param<operators::DropoutParam>();
  const float* x_data = param.x->data<float>();
  float* out_data = param.output->mutable_data<float>();
  int num = param.x->dims().production();
  const float prob_data = param.dropout_prob;
  if (param.dropout_implementation == "upscale_in_train") {
    lite::arm::math::dropout_up(x_data, out_data, num);
  } else {
    lite::arm::math::dropout_down(x_data, out_data, num, prob_data);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(dropout, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::DropoutCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
