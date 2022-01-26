/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/**************************** Enforce Wrapper **************************/

#pragma once

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#endif

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/enforce_npu.h"
#include "paddle/fluid/platform/device/npu/npu_info.h"
#endif

#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/enforce.h"
#include "paddle/fluid/platform/device/mlu/mlu_info.h"
#endif

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif
