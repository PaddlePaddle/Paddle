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

#include "paddle/fluid/lite/kernels/arm/calib_compute.h"
#include <vector>
#include "paddle/fluid/lite/arm/math/type_trans.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void CalibCompute::Run() {
  auto& param = this->Param<operators::CalibParam>();
  std::vector<float> scale = {param.in_scale};
  if (param.in_dtype == PRECISION(kFloat) &&
      param.out_dtype == PRECISION(kInt8)) {
    const auto* din = param.input->data<float>();
    auto* dout = param.output->mutable_data<signed char>();
    lite::arm::math::fp32_to_int8(din, dout, scale.data(), 1, 1,
                                  param.input->numel());
    return;
  }
  if (param.in_dtype == PRECISION(kInt8) &&
      param.out_dtype == PRECISION(kFloat)) {
    const auto* din = param.input->data<signed char>();
    auto* dout = param.output->mutable_data<float>();
    lite::arm::math::int8_to_fp32(din, dout, scale.data(), 1, 1,
                                  param.input->numel());
    return;
  }
  LOG(FATAL) << "Unsupport Dtype.";
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(calib, kARM, kInt8, kNCHW,
                     paddle::lite::kernels::arm::CalibCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
