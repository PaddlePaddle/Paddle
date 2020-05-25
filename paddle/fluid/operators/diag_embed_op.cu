// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/diag_embed_op.h"

namespace ops = paddle::operators;
namespace platform = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    diag_embed, ops::DiagEmbedKernel<paddle::platform::CUDADeviceContext, int>,
    ops::DiagEmbedKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::DiagEmbedKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DiagEmbedKernel<paddle::platform::CUDADeviceContext,
                         platform::float16>,
    ops::DiagEmbedKernel<paddle::platform::CUDADeviceContext, double>);
