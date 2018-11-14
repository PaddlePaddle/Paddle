// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/unstack_op.h"

namespace plat = paddle::platform;
namespace ops = paddle::operators;

USE_OP(stack);

REGISTER_OPERATOR(unstack, ops::UnStackOp, ops::UnStackOpMaker,
                  ops::UnStackOpInferShape, ops::UnStackGradOpDescMaker);

REGISTER_OPERATOR(unstack_grad, ops::UnStackGradOp,
                  ops::UnStackOpGradInferShape);
