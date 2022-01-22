// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#ifdef __xpu_kp__
#include <xpu/runtime.h>
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/debug.h"
#include "xpu/kernel/math.h"
#endif

#if (defined(__CUDACC__) || defined(__HIPCC__) || defined(__xpu_kp__))
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif
