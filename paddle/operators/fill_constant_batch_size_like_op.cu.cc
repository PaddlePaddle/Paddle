/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/fill_constant_batch_size_like_op.h"
#include "paddle/framework/op_registry.h"

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(
    fill_constant_batch_size_like,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::GPUPlace, float>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::GPUPlace, double>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::GPUPlace, int>,
    ops::FillConstantBatchSizeLikeOpKernel<paddle::platform::GPUPlace,
                                           int64_t>);
