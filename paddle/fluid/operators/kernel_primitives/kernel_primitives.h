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

#pragma once
#include "paddle/fluid/operators/kernel_primitives/helper_primitives.h"
#ifdef PADDLE_WITH_XPU2
#include "paddle/fluid/operators/kernel_primitives/compute_primitives_xpu2.h"
#include "paddle/fluid/operators/kernel_primitives/datamover_primitives_xpu2.h"
#include "paddle/fluid/operators/kernel_primitives/functor_primitives_xpu2.h"

#define KPStream XPUStream
#define KPDevice paddle::platform::XPUDeviceContext
#define _ptr_ _global_ptr_
#define __forceinline__ __inline__
#define __restrict__

#define THREAD_ID_X core_id()
#define THREAD_ID_Y 0
#define THREAD_ID_Z 0

#define BLOCK_NUM_X core_num()
#define BLOCK_NUM_Y 0
#define BLOCK_NUM_Z 0

#define BLOCK_ID_X cluster_id()
#define BLOCK_ID_Y 0
#define BLOCK_ID_Z 0

#define GRID_NUM_X cluster_num()
#define GRID_NUM_Y 0
#define GRID_NUM_Z 0
#else
#include "paddle/fluid/operators/kernel_primitives/compute_primitives.h"
#include "paddle/fluid/operators/kernel_primitives/datamover_primitives.h"
#include "paddle/fluid/operators/kernel_primitives/functor_primitives.h"

#define KPStream gpuStream_t
#define KPDevice paddle::platform::CUDADeviceContext
#define _ptr_

#define THREAD_ID_X threadIdx.x
#define THREAD_ID_Y threadIdx.y
#define THREAD_ID_Z threadIdx.z

#define BLOCK_NUM_X blockDim.x
#define BLOCK_NUM_Y blockDim.y
#define BLOCK_NUM_Z blockDim.z

#define BLOCK_ID_X blockIdx.x
#define BLOCK_ID_Y blockIdx.y
#define BLOCK_ID_Z blockIdx.z

#define GRID_NUM_X gridDim.x
#define GRID_NUM_Y gridDim.y
#define GRID_NUM_Z gridDim.z
#endif

namespace paddle {
namespace operators {
namespace kernel_primitives {}
}
}
