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

#include "paddle/fluid/lite/kernels/x86/gru_compute.h"

REGISTER_LITE_KERNEL(gru, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::GruCompute<float>, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
