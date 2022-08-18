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

#include "paddle/fluid/operators/unsqueeze_op.h"
#ifdef PADDLE_WITH_XPU
namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    unsqueeze,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, float>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, double>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, plat::float16>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, bool>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, int>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, uint8_t>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, int8_t>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, int64_t>);
REGISTER_OP_XPU_KERNEL(
    unsqueeze_grad,
    ops::UnsqueezeGradKernel<paddle::platform::XPUDeviceContext, float>,
    ops::UnsqueezeGradKernel<paddle::platform::XPUDeviceContext, double>,
    ops::UnsqueezeGradKernel<paddle::platform::XPUDeviceContext, plat::float16>,
    ops::UnsqueezeGradKernel<paddle::platform::XPUDeviceContext, bool>,
    ops::UnsqueezeGradKernel<paddle::platform::XPUDeviceContext, int>,
    ops::UnsqueezeGradKernel<paddle::platform::XPUDeviceContext, int8_t>,
    ops::UnsqueezeGradKernel<paddle::platform::XPUDeviceContext, uint8_t>,
    ops::UnsqueezeGradKernel<paddle::platform::XPUDeviceContext, int64_t>);
REGISTER_OP_XPU_KERNEL(
    unsqueeze2,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, float>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, double>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, plat::float16>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, bool>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, int>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, uint8_t>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, int8_t>,
    ops::UnsqueezeKernel<paddle::platform::XPUDeviceContext, int64_t>);
REGISTER_OP_XPU_KERNEL(
    unsqueeze2_grad,
    ops::Unsqueeze2GradKernel<paddle::platform::XPUDeviceContext, float>,
    ops::Unsqueeze2GradKernel<paddle::platform::XPUDeviceContext, double>,
    ops::Unsqueeze2GradKernel<paddle::platform::XPUDeviceContext,
                              plat::float16>,
    ops::Unsqueeze2GradKernel<paddle::platform::XPUDeviceContext, bool>,
    ops::Unsqueeze2GradKernel<paddle::platform::XPUDeviceContext, int>,
    ops::Unsqueeze2GradKernel<paddle::platform::XPUDeviceContext, uint8_t>,
    ops::Unsqueeze2GradKernel<paddle::platform::XPUDeviceContext, int8_t>,
    ops::Unsqueeze2GradKernel<paddle::platform::XPUDeviceContext, int64_t>);

#endif
