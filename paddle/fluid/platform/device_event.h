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
#include "paddle/fluid/platform/device_event_base.h"

/*
 * NOTE: Now we generate this file manually and will consider
 *  automatically generate it later. Just as 'paddle/fluid/pybind/pybind.h'
 *  for USE_OP from op_library macros, and
 * `paddle/fluid/inference/paddle_inference_pass.h`
 *  for USE_PASS from pass_library.
 */

using ::paddle::platform::kCPU;
using ::paddle::platform::kCUDA;
<<<<<<< HEAD
using ::paddle::platform::kCUSTOM_DEVICE;
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
using ::paddle::platform::kNPU;
using ::paddle::platform::kXPU;

USE_EVENT(kCPU)
USE_EVENT_WAIT(kCPU, kCPU)

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
USE_EVENT(kCUDA);
USE_EVENT_WAIT(kCUDA, kCUDA)
USE_EVENT_WAIT(kCPU, kCUDA)
#endif

#ifdef PADDLE_WITH_ASCEND_CL
USE_EVENT(kNPU);
USE_EVENT_WAIT(kNPU, kNPU)
USE_EVENT_WAIT(kCPU, kNPU)
#endif
<<<<<<< HEAD

#ifdef PADDLE_WITH_CUSTOM_DEVICE
USE_EVENT(kCUSTOM_DEVICE);
USE_EVENT_WAIT(kCUSTOM_DEVICE, kCUSTOM_DEVICE)
USE_EVENT_WAIT(kCPU, kCUSTOM_DEVICE)
#endif
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
