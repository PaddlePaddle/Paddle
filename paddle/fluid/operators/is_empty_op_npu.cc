/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/is_empty_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    is_empty, ops::IsEmptyOpKernel<plat::NPUDeviceContext, float>,
    ops::IsEmptyOpKernel<plat::NPUDeviceContext, plat::float16>,
    ops::IsEmptyOpKernel<plat::NPUDeviceContext, int>,
    ops::IsEmptyOpKernel<plat::NPUDeviceContext, int64_t>);
