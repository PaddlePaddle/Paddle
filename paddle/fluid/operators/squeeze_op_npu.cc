/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/squeeze_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    squeeze,
    ops::SqueezeKernel<plat::NPUDeviceContext, float>,
    ops::SqueezeKernel<plat::NPUDeviceContext, double>,
    ops::SqueezeKernel<plat::NPUDeviceContext, plat::float16>,
    ops::SqueezeKernel<plat::NPUDeviceContext, bool>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int>,
    ops::SqueezeKernel<plat::NPUDeviceContext, uint8_t>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int8_t>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    squeeze2,
    ops::SqueezeKernel<plat::NPUDeviceContext, float>,
    ops::SqueezeKernel<plat::NPUDeviceContext, double>,
    ops::SqueezeKernel<plat::NPUDeviceContext, plat::float16>,
    ops::SqueezeKernel<plat::NPUDeviceContext, bool>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int>,
    ops::SqueezeKernel<plat::NPUDeviceContext, uint8_t>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int8_t>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<plat::NPUDeviceContext, float>,
    ops::SqueezeGradKernel<plat::NPUDeviceContext, double>,
    ops::SqueezeGradKernel<plat::NPUDeviceContext, plat::float16>,
    ops::SqueezeGradKernel<plat::NPUDeviceContext, bool>,
    ops::SqueezeGradKernel<plat::NPUDeviceContext, int>,
    ops::SqueezeGradKernel<plat::NPUDeviceContext, uint8_t>,
    ops::SqueezeGradKernel<plat::NPUDeviceContext, int8_t>,
    ops::SqueezeGradKernel<plat::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    squeeze2_grad,
    ops::Squeeze2GradKernel<plat::NPUDeviceContext, float>,
    ops::Squeeze2GradKernel<plat::NPUDeviceContext, double>,
    ops::Squeeze2GradKernel<plat::NPUDeviceContext, plat::float16>,
    ops::Squeeze2GradKernel<plat::NPUDeviceContext, bool>,
    ops::Squeeze2GradKernel<plat::NPUDeviceContext, int>,
    ops::Squeeze2GradKernel<plat::NPUDeviceContext, uint8_t>,
    ops::Squeeze2GradKernel<plat::NPUDeviceContext, int8_t>,
    ops::Squeeze2GradKernel<plat::NPUDeviceContext, int64_t>);
