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

#include "paddle/fluid/operators/collective/c_allreduce_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(c_allreduce_prod,
                       ops::CAllReduceOpMLUKernel<ops::kRedProd, float>,
                       ops::CAllReduceOpMLUKernel<ops::kRedProd, plat::float16>,
                       ops::CAllReduceOpMLUKernel<ops::kRedProd, int>,
                       ops::CAllReduceOpMLUKernel<ops::kRedProd, int16_t>,
                       ops::CAllReduceOpMLUKernel<ops::kRedProd, int8_t>,
                       ops::CAllReduceOpMLUKernel<ops::kRedProd, uint8_t>)
