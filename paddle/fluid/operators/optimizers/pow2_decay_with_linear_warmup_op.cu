// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/optimizers/pow2_decay_with_linear_warmup_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    pow2_decay_with_linear_warmup,
    ops::Pow2DecayWithLinearWarmupOpKernel<phi::GPUContext, double>,
    ops::Pow2DecayWithLinearWarmupOpKernel<phi::GPUContext, float>);
