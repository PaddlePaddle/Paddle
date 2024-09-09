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

#include <string>
#include <type_traits>
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/phi/api/profiler/host_event_recorder.h"
#include "paddle/phi/core/os_info.h"

namespace paddle {
namespace platform {

template <typename EventType>
using EventContainer = phi::EventContainer<EventType>;

template <typename EventType>
using ThreadEventSection = phi::ThreadEventSection<EventType>;

template <typename EventType>
using ThreadEventRecorder = phi::ThreadEventRecorder<EventType>;

template <typename EventType>
using HostEventSection = phi::HostEventSection<EventType>;

template <typename EventType>
using HostEventRecorder = phi::HostEventRecorder<EventType>;

}  // namespace platform
}  // namespace paddle
