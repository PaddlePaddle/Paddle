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

#include "paddle/fluid/operators/unsqueeze_op.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(
    unsqueeze, ops::UnsqueezeKernel<plat::MLUDeviceContext, float>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, double>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, plat::float16>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, bool>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, int>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, int8_t>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, int64_t>);
REGISTER_OP_MLU_KERNEL(
    unsqueeze2, ops::UnsqueezeKernel<plat::MLUDeviceContext, float>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, double>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, plat::float16>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, bool>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, int>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, int8_t>,
    ops::UnsqueezeKernel<plat::MLUDeviceContext, int64_t>);
REGISTER_OP_MLU_KERNEL(
    unsqueeze_grad, ops::UnsqueezeGradKernel<plat::MLUDeviceContext, float>,
    ops::UnsqueezeGradKernel<plat::MLUDeviceContext, double>,
    ops::UnsqueezeGradKernel<plat::MLUDeviceContext, plat::float16>,
    ops::UnsqueezeGradKernel<plat::MLUDeviceContext, bool>,
    ops::UnsqueezeGradKernel<plat::MLUDeviceContext, int>,
    ops::UnsqueezeGradKernel<plat::MLUDeviceContext, int8_t>,
    ops::UnsqueezeGradKernel<plat::MLUDeviceContext, int64_t>);
REGISTER_OP_MLU_KERNEL(
    unsqueeze2_grad, ops::Unsqueeze2GradKernel<plat::MLUDeviceContext, float>,
    ops::Unsqueeze2GradKernel<plat::MLUDeviceContext, double>,
    ops::Unsqueeze2GradKernel<plat::MLUDeviceContext, plat::float16>,
    ops::Unsqueeze2GradKernel<plat::MLUDeviceContext, bool>,
    ops::Unsqueeze2GradKernel<plat::MLUDeviceContext, int>,
    ops::Unsqueeze2GradKernel<plat::MLUDeviceContext, int8_t>,
    ops::Unsqueeze2GradKernel<plat::MLUDeviceContext, int64_t>);
#endif
