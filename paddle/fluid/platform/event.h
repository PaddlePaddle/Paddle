/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <functional>
#include <map>
#include <string>
#include <utility>
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif
#include "paddle/phi/api/profiler/event.h"

namespace paddle {
namespace platform {

using EventType = phi::EventType;
using EventRole = phi::EventRole;
using Event = phi::Event;

using EventWithStartNs = std::pair<Event *, uint64_t>;
using ThreadEvents = std::map<uint64_t, EventWithStartNs>;

using MemEvent = phi::MemEvent;
using CudaEvent = phi::CudaEvent;

}  // namespace platform
}  // namespace paddle
