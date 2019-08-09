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

#include "paddle/fluid/lite/kernels/x86/elementwise_compute.h"

REGISTER_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::ElementwiseSubCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW,
                     paddle::lite::kernels::x86::ElementwiseAddCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

#ifdef LITE_WITH_X86
REGISTER_LITE_KERNEL(
    elementwise_sub_grad, kX86, kFloat, kNCHW,
    paddle::lite::kernels::x86::ElementwiseSubGradCompute<float>, def)
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput(paddle::framework::GradVarName("Out"),
               {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput(paddle::framework::GradVarName("X"),
                {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput(paddle::framework::GradVarName("Y"),
                {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
#endif
