/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#ifdef PADDLE_WITH_XPU

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    squeeze,
    ops::SqueezeKernel<paddle::platform::XPUDeviceContext, float>,
    ops::SqueezeKernel<paddle::platform::XPUDeviceContext, double>,
    ops::SqueezeKernel<paddle::platform::XPUDeviceContext, plat::float16>,
    ops::SqueezeKernel<paddle::platform::XPUDeviceContext, bool>,
    ops::SqueezeKernel<paddle::platform::XPUDeviceContext, int>,
    ops::SqueezeKernel<paddle::platform::XPUDeviceContext, uint8_t>,
    ops::SqueezeKernel<paddle::platform::XPUDeviceContext, int8_t>,
    ops::SqueezeKernel<paddle::platform::XPUDeviceContext, int64_t>);
REGISTER_OP_XPU_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<paddle::platform::XPUDeviceContext, float>,
    ops::SqueezeGradKernel<paddle::platform::XPUDeviceContext, double>,
    ops::SqueezeGradKernel<paddle::platform::XPUDeviceContext, plat::float16>,
    ops::SqueezeGradKernel<paddle::platform::XPUDeviceContext, bool>,
    ops::SqueezeGradKernel<paddle::platform::XPUDeviceContext, int>,
    ops::SqueezeGradKernel<paddle::platform::XPUDeviceContext, uint8_t>,
    ops::SqueezeGradKernel<paddle::platform::XPUDeviceContext, int8_t>,
    ops::SqueezeGradKernel<paddle::platform::XPUDeviceContext, int64_t>);
REGISTER_OP_XPU_KERNEL(
    squeeze2,
    ops::Squeeze2Kernel<paddle::platform::XPUDeviceContext, float>,
    ops::Squeeze2Kernel<paddle::platform::XPUDeviceContext, double>,
    ops::Squeeze2Kernel<paddle::platform::XPUDeviceContext, plat::float16>,
    ops::Squeeze2Kernel<paddle::platform::XPUDeviceContext, bool>,
    ops::Squeeze2Kernel<paddle::platform::XPUDeviceContext, int>,
    ops::Squeeze2Kernel<paddle::platform::XPUDeviceContext, int8_t>,
    ops::Squeeze2Kernel<paddle::platform::XPUDeviceContext, uint8_t>,
    ops::Squeeze2Kernel<paddle::platform::XPUDeviceContext, int64_t>);
REGISTER_OP_XPU_KERNEL(
    squeeze2_grad,
    ops::Squeeze2GradKernel<paddle::platform::XPUDeviceContext, float>,
    ops::Squeeze2GradKernel<paddle::platform::XPUDeviceContext, double>,
    ops::Squeeze2GradKernel<paddle::platform::XPUDeviceContext, plat::float16>,
    ops::Squeeze2GradKernel<paddle::platform::XPUDeviceContext, bool>,
    ops::Squeeze2GradKernel<paddle::platform::XPUDeviceContext, int>,
    ops::Squeeze2GradKernel<paddle::platform::XPUDeviceContext, int8_t>,
    ops::Squeeze2GradKernel<paddle::platform::XPUDeviceContext, uint8_t>,
    ops::Squeeze2GradKernel<paddle::platform::XPUDeviceContext, int64_t>);

#endif
