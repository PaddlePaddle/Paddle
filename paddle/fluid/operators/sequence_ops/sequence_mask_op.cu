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

#include "paddle/fluid/operators/sequence_ops/sequence_mask_op.h"

REGISTER_OP_CUDA_KERNEL(
    sequence_mask,
    paddle::operators::SequenceMaskKernel<paddle::platform::CUDADeviceContext,
                                          int>,
    paddle::operators::SequenceMaskKernel<paddle::platform::CUDADeviceContext,
                                          int64_t>,
    paddle::operators::SequenceMaskKernel<paddle::platform::CUDADeviceContext,
                                          float>,
    paddle::operators::SequenceMaskKernel<paddle::platform::CUDADeviceContext,
                                          double>);
