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

#include "paddle/fluid/operators/sequence_ops/sequence_concat_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace platform {
class CUDADeviceContext;
}  // namespace platform
}  // namespace paddle

template <typename T>
using SeqConcatKernel =
    paddle::operators::SeqConcatKernel<paddle::platform::CUDADeviceContext, T>;
REGISTER_OP_CUDA_KERNEL(sequence_concat, SeqConcatKernel<float>, SeqConcatKernel<double>,
                        SeqConcatKernel<int>, SeqConcatKernel<int64_t>);
template <typename T>
using SeqConcatGradKernel =
    paddle::operators::SeqConcatGradKernel<paddle::platform::CUDADeviceContext,
                                           T>;
REGISTER_OP_CUDA_KERNEL(sequence_concat_grad, SeqConcatGradKernel<float>,
                        SeqConcatGradKernel<double>, SeqConcatGradKernel<int>,
                        SeqConcatGradKernel<int64_t>);
