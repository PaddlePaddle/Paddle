// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <thrust/execution_policy.h>
#include <thrust/version.h>

// thrust::cuda::par synchronizes the stream after every algorithm call.
// thrust::cuda::par_nosync skips that sync; algorithms that return a value to
// the host (reduce, equal, unique, ...) still synchronize. par_nosync is only
// available since Thrust 1.16, so fall back to thrust::cuda::par for older
// Thrust, e.g. the one shipped with CUDA 11.x.
#if defined(PADDLE_WITH_CUDA) && THRUST_VERSION >= 101600
#define PADDLE_THRUST_CUDA_PAR thrust::cuda::par_nosync
#else
#define PADDLE_THRUST_CUDA_PAR thrust::cuda::par
#endif
