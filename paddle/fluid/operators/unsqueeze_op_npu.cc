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

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/unsqueeze_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    unsqueeze, ops::UnsqueezeKernel<plat::NPUDeviceContext, float>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, double>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, plat::float16>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, bool>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int8_t>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    unsqueeze2, ops::UnsqueezeKernel<plat::NPUDeviceContext, float>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, double>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, plat::float16>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, bool>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int8_t>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    unsqueeze_grad, ops::UnsqueezeGradKernel<plat::NPUDeviceContext, float>,
    ops::UnsqueezeGradKernel<plat::NPUDeviceContext, double>,
    ops::UnsqueezeGradKernel<plat::NPUDeviceContext, plat::float16>,
    ops::UnsqueezeGradKernel<plat::NPUDeviceContext, bool>,
    ops::UnsqueezeGradKernel<plat::NPUDeviceContext, int>,
    ops::UnsqueezeGradKernel<plat::NPUDeviceContext, int8_t>,
    ops::UnsqueezeGradKernel<plat::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    unsqueeze2_grad, ops::Unsqueeze2GradKernel<plat::NPUDeviceContext, float>,
    ops::Unsqueeze2GradKernel<plat::NPUDeviceContext, double>,
    ops::Unsqueeze2GradKernel<plat::NPUDeviceContext, plat::float16>,
    ops::Unsqueeze2GradKernel<plat::NPUDeviceContext, bool>,
    ops::Unsqueeze2GradKernel<plat::NPUDeviceContext, int>,
    ops::Unsqueeze2GradKernel<plat::NPUDeviceContext, int8_t>,
    ops::Unsqueeze2GradKernel<plat::NPUDeviceContext, int64_t>);
#endif
