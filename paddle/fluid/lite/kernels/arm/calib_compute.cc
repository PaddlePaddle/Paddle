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

void CalibComputeFp32ToInt8::Run() {
  auto& param = this->Param<operators::CalibParam>();
  std::vector<float> scale = {param.scale};
  const auto* din = param.input->data<float>();
  auto* dout = param.output->mutable_data<signed char>();
  lite::arm::math::fp32_to_int8(din, dout, scale.data(), 1, 1,
                                param.input->numel());
  return;
}

void CalibComputeInt8ToFp32::Run() {
  auto& param = this->Param<operators::CalibParam>();
  const auto* din = param.input->data<signed char>();
  std::vector<float> scale = {param.scale};
  auto* dout = param.output->mutable_data<float>();
  lite::arm::math::int8_to_fp32(din, dout, scale.data(), 1, 1,
                                param.input->numel());
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(calib, kARM, kInt8, kNCHW,
                     paddle::lite::kernels::arm::CalibComputeFp32ToInt8,
                     fp32_to_int8)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(calib, kARM, kInt8, kNCHW,
                     paddle::lite::kernels::arm::CalibComputeInt8ToFp32,
                     int8_to_fp32)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
