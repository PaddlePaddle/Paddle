/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include <thrust/device_vector.h>

#include <algorithm>
#include <vector>

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/svd_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    svd_grad,
    ops::SvdGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SvdGradKernel<paddle::platform::CUDADeviceContext, double>);
#endif  // not PADDLE_WITH_HIP
