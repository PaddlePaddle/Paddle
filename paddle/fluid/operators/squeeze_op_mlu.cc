/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_MLU
#include <memory>
#include <string>

#include "paddle/fluid/operators/squeeze_op.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(
    squeeze,
    ops::SqueezeKernel<plat::MLUDeviceContext, float>,
    ops::SqueezeKernel<plat::MLUDeviceContext, double>,
    ops::SqueezeKernel<plat::MLUDeviceContext, plat::float16>,
    ops::SqueezeKernel<plat::MLUDeviceContext, bool>,
    ops::SqueezeKernel<plat::MLUDeviceContext, int>,
    ops::SqueezeKernel<plat::MLUDeviceContext, uint8_t>,
    ops::SqueezeKernel<plat::MLUDeviceContext, int8_t>,
    ops::SqueezeKernel<plat::MLUDeviceContext, int64_t>);

REGISTER_OP_MLU_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<plat::MLUDeviceContext, float>,
    ops::SqueezeGradKernel<plat::MLUDeviceContext, double>,
    ops::SqueezeGradKernel<plat::MLUDeviceContext, plat::float16>,
    ops::SqueezeGradKernel<plat::MLUDeviceContext, bool>,
    ops::SqueezeGradKernel<plat::MLUDeviceContext, int>,
    ops::SqueezeGradKernel<plat::MLUDeviceContext, uint8_t>,
    ops::SqueezeGradKernel<plat::MLUDeviceContext, int8_t>,
    ops::SqueezeGradKernel<plat::MLUDeviceContext, int64_t>);

REGISTER_OP_MLU_KERNEL(
    squeeze2,
    ops::SqueezeKernel<plat::MLUDeviceContext, float>,
    ops::SqueezeKernel<plat::MLUDeviceContext, double>,
    ops::SqueezeKernel<plat::MLUDeviceContext, plat::float16>,
    ops::SqueezeKernel<plat::MLUDeviceContext, bool>,
    ops::SqueezeKernel<plat::MLUDeviceContext, int>,
    ops::SqueezeKernel<plat::MLUDeviceContext, uint8_t>,
    ops::SqueezeKernel<plat::MLUDeviceContext, int8_t>,
    ops::SqueezeKernel<plat::MLUDeviceContext, int64_t>);

REGISTER_OP_MLU_KERNEL(
    squeeze2_grad,
    ops::Squeeze2GradKernel<plat::MLUDeviceContext, float>,
    ops::Squeeze2GradKernel<plat::MLUDeviceContext, double>,
    ops::Squeeze2GradKernel<plat::MLUDeviceContext, plat::float16>,
    ops::Squeeze2GradKernel<plat::MLUDeviceContext, bool>,
    ops::Squeeze2GradKernel<plat::MLUDeviceContext, int>,
    ops::Squeeze2GradKernel<plat::MLUDeviceContext, uint8_t>,
    ops::Squeeze2GradKernel<plat::MLUDeviceContext, int8_t>,
    ops::Squeeze2GradKernel<plat::MLUDeviceContext, int64_t>);
#endif
