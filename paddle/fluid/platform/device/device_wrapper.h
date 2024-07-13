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
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/callback_manager.h"
#include "paddle/phi/backends/custom/enforce_custom.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/event.h"
#include "paddle/phi/backends/stream.h"
#endif
