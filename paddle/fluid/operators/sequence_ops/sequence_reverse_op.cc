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

#include "paddle/fluid/operators/sequence_ops/sequence_reverse_op.h"

namespace ops = paddle::operators;

REGISTER_OPERATOR(sequence_reverse, ops::SequenceReverseOp,
                  ops::SequenceReverseOpMaker,
                  ops::SequenceReverseGradOpDescMaker);

REGISTER_OP_CPU_KERNEL(
    sequence_reverse,
    ops::SequenceReverseOpKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::SequenceReverseOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequenceReverseOpKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::SequenceReverseOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequenceReverseOpKernel<paddle::platform::CPUDeviceContext, double>);
