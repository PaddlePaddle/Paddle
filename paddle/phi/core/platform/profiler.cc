// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/platform/profiler.h"

#include <mutex>  // NOLINT
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/common_event.h"
#include "paddle/fluid/platform/profiler/host_tracer.h"
#include "paddle/fluid/platform/profiler/profiler.h"
#include "paddle/phi/api/profiler/device_tracer.h"
#include "paddle/phi/core/platform/profiler/host_event_recorder.h"
#include "paddle/phi/core/platform/profiler_helper.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/nvtx.h"
#endif
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/os_info.h"

COMMON_DECLARE_bool(enable_record_memory);

#if defined(_WIN32) && defined(PHI_SHARED)
phi::ProfilerState phi::ProfilerHelper::g_state = phi::ProfilerState::kDisabled;
bool phi::ProfilerHelper::g_enable_nvprof_hook = false;
thread_local uint64_t phi::ProfilerHelper::g_thread_id;
uint32_t phi::ProfilerHelper::g_next_thread_id = 0;
std::mutex phi::ProfilerHelper::g_all_event_lists_mutex;
std::list<std::shared_ptr<phi::EventList<phi::Event>>>
    phi::ProfilerHelper::g_all_event_lists;
thread_local std::shared_ptr<phi::EventList<phi::Event>>
    phi::ProfilerHelper::g_event_list;
std::list<std::shared_ptr<phi::EventList<phi::MemEvent>>>
    phi::ProfilerHelper::g_all_mem_event_lists;
thread_local std::shared_ptr<phi::EventList<phi::MemEvent>>
    phi::ProfilerHelper::g_mem_event_list;
std::mutex phi::ProfilerHelper::g_all_mem_event_lists_mutex;
#endif
namespace paddle::platform {

MemEventRecorder MemEventRecorder::recorder;

RecordInstantEvent::RecordInstantEvent(const char *name,
                                       phi::TracerEventType type,
                                       uint32_t level) {
  if (UNLIKELY(HostTraceLevel::GetInstance().NeedTrace(level) == false)) {
    return;
  }
  auto start_end_ns = phi::PosixInNsec();
  HostEventRecorder<CommonEvent>::GetInstance().RecordEvent(
      name, start_end_ns, start_end_ns, EventRole::kOrdinary, type);
}

RecordOpInfoSupplement::RecordOpInfoSupplement(
    const std::string &type,
    const framework::AttributeMap &attrs,
    const framework::InferShapeContext &shape_ctx,
    const framework::RuntimeContext &ctx,
    uint64_t op_id) {
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return;
  }
  if (IsEnabled() == false) {
    return;
  }
  std::map<std::string, std::vector<phi::DDim>> input_shapes;
  std::map<std::string, std::vector<framework::proto::VarType::Type>> dtypes;
  for (const auto &input : ctx.inputs) {
    input_shapes[input.first] = shape_ctx.GetInputsDim(input.first);
    dtypes[input.first] = shape_ctx.GetInputsVarType(input.first);
  }

  HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance().RecordEvent(
      phi::PosixInNsec(), type, input_shapes, dtypes, attrs, op_id);
}

RecordOpInfoSupplement::RecordOpInfoSupplement(
    const std::string &type,
    const framework::AttributeMap &attrs,
    const framework::InferShapeContext &shape_ctx,
    const phi::KernelSignature &kernel_signature) {
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return;
  }
  if (IsEnabled() == false) {
    return;
  }
  std::map<std::string, std::vector<phi::DDim>> input_shapes;
  std::map<std::string, std::vector<framework::proto::VarType::Type>> dtypes;
  for (auto input_name_char : kernel_signature.input_names) {
    std::string input_name(input_name_char);
    if (shape_ctx.HasInputs(input_name)) {
      input_shapes[input_name] = shape_ctx.GetInputsDim(input_name);
      dtypes[input_name] = shape_ctx.GetInputsVarType(input_name);
    }
  }
  uint64_t op_id = 0;
  HostEventRecorder<OperatorSupplementOriginEvent>::GetInstance().RecordEvent(
      phi::PosixInNsec(), type, input_shapes, dtypes, attrs, op_id);
}

bool RecordMemEvent::IsEnabled() { return FLAGS_enable_record_memory; }

std::map<const char *, std::map<uint64_t, std::vector<uint64_t>>>
    RecordMemEvent::size_cache;

std::map<const char *, std::map<uint64_t, bool>>
    RecordMemEvent::has_initialized;

RecordMemEvent::RecordMemEvent(const void *ptr,
                               const phi::Place &place,
                               size_t size,
                               const phi::TracerMemEventType type) {
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled &&
      FLAGS_enable_host_event_recorder_hook == false) {
    return;
  }

  if (IsEnabled() == false) {
    return;
  }

  if (type == phi::TracerMemEventType::Allocate) {
    uint64_t current_allocated = 0;
    uint64_t peak_allocated = 0;
    uint64_t current_reserved = 0;  // 0 means keep the same as before
    uint64_t peak_reserved = 0;     // 0 means keep the same as before
    if (phi::is_cpu_place(place) || phi::is_cuda_pinned_place(place)) {
      if (RecordMemEvent::has_initialized["cpu"][place.GetDeviceId()] ==
          false) {
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId()));
        current_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][0];
        current_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][1];
        peak_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][2];
        peak_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][3];
        RecordMemEvent::has_initialized["cpu"][place.GetDeviceId()] = true;
      } else {
        current_allocated =
            HOST_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId());
        peak_allocated =
            HOST_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId());
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][0] =
            current_allocated;
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][2] =
            peak_allocated;
        current_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][1];
        peak_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][3];
      }

    } else {
      if (RecordMemEvent::has_initialized["gpu"][place.GetDeviceId()] ==
          false) {
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId()));
        current_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][0];
        current_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][1];
        peak_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][2];
        peak_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][3];
        RecordMemEvent::has_initialized["gpu"][place.GetDeviceId()] = true;
      } else {
        current_allocated = DEVICE_MEMORY_STAT_CURRENT_VALUE(
            Allocated, place.GetDeviceId());  // NOLINT
        peak_allocated =
            DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId());
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][0] =
            current_allocated;
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][2] =
            peak_allocated;
        current_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][1];
        peak_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][3];
      }
    }
    platform::MemEventRecorder::Instance().PushMemRecord(ptr,
                                                         place,
                                                         size,
                                                         type,
                                                         current_allocated,
                                                         current_reserved,
                                                         peak_allocated,
                                                         peak_reserved);
  } else if (type == phi::TracerMemEventType::ReservedAllocate) {
    uint64_t current_reserved = 0;
    uint64_t peak_reserved = 0;
    uint64_t current_allocated = 0;  // 0 means keep the same as before
    uint64_t peak_allocated = 0;     // 0 means keep the same as before
    if (phi::is_cpu_place(place) || phi::is_cuda_pinned_place(place)) {
      if (RecordMemEvent::has_initialized["cpu"][place.GetDeviceId()] ==
          false) {
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId()));
        current_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][0];
        current_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][1];
        peak_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][2];
        peak_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][3];
        RecordMemEvent::has_initialized["cpu"][place.GetDeviceId()] = true;
      } else {
        current_reserved =
            HOST_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId());
        peak_reserved =
            HOST_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId());
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][1] =
            current_reserved;
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][3] =
            peak_reserved;
        current_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][0];
        peak_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][2];
      }
    } else {
      if (RecordMemEvent::has_initialized["gpu"][place.GetDeviceId()] ==
          false) {
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId()));
        current_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][0];
        current_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][1];
        peak_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][2];
        peak_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][3];
        RecordMemEvent::has_initialized["gpu"][place.GetDeviceId()] = true;
      } else {
        current_reserved = DEVICE_MEMORY_STAT_CURRENT_VALUE(
            Reserved, place.GetDeviceId());  // NOLINT
        peak_reserved = DEVICE_MEMORY_STAT_PEAK_VALUE(
            Reserved, place.GetDeviceId());  // NOLINT
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][1] =
            current_reserved;
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][3] =
            peak_reserved;
        current_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][0];
        peak_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][2];
      }
    }
    platform::MemEventRecorder::Instance().PushMemRecord(ptr,
                                                         place,
                                                         size,
                                                         type,
                                                         current_allocated,
                                                         current_reserved,
                                                         peak_allocated,
                                                         peak_reserved);
  } else if (type == phi::TracerMemEventType::Free) {
    uint64_t current_allocated = 0;
    uint64_t peak_allocated = 0;
    uint64_t current_reserved = 0;  // 0 means keep the same as before
    uint64_t peak_reserved = 0;     // 0 means keep the same as before
    if (phi::is_cpu_place(place) || phi::is_cuda_pinned_place(place)) {
      if (RecordMemEvent::has_initialized["cpu"][place.GetDeviceId()] ==
          false) {
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId()));
        current_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][0];
        current_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][1];
        peak_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][2];
        peak_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][3];
        RecordMemEvent::has_initialized["cpu"][place.GetDeviceId()] = true;
      } else {
        current_allocated =
            HOST_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId());
        peak_allocated =
            HOST_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId());
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][0] =
            current_allocated;
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][2] =
            peak_allocated;
        current_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][1];
        peak_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][3];
      }
    } else {
      if (RecordMemEvent::has_initialized["gpu"][place.GetDeviceId()] ==
          false) {
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId()));
        current_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][0];
        current_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][1];
        peak_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][2];
        peak_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][3];
        RecordMemEvent::has_initialized["gpu"][place.GetDeviceId()] = true;
      } else {
        current_allocated = DEVICE_MEMORY_STAT_CURRENT_VALUE(
            Allocated, place.GetDeviceId());  // NOLINT
        peak_allocated = DEVICE_MEMORY_STAT_PEAK_VALUE(
            Allocated, place.GetDeviceId());  // NOLINT
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][0] =
            current_allocated;
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][2] =
            peak_allocated;
        current_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][1];
        peak_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][3];
      }
    }
    platform::MemEventRecorder::Instance().PopMemRecord(ptr,
                                                        place,
                                                        size,
                                                        type,
                                                        current_allocated,
                                                        current_reserved,
                                                        peak_allocated,
                                                        peak_reserved);
  } else if (type == phi::TracerMemEventType::ReservedFree) {
    uint64_t current_reserved = 0;
    uint64_t peak_reserved = 0;
    uint64_t current_allocated = 0;  // 0 means keep the same as before
    uint64_t peak_allocated = 0;     // 0 means keep the same as before
    if (phi::is_cpu_place(place) || phi::is_cuda_pinned_place(place)) {
      if (RecordMemEvent::has_initialized["cpu"][place.GetDeviceId()] ==
          false) {
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()].push_back(
            HOST_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId()));
        current_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][0];
        current_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][1];
        peak_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][2];
        peak_reserved =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][3];
        RecordMemEvent::has_initialized["cpu"][place.GetDeviceId()] = true;
      } else {
        current_reserved =
            HOST_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId());
        peak_reserved =
            HOST_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId());
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][1] =
            current_reserved;
        RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][3] =
            peak_reserved;
        current_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][0];
        peak_allocated =
            RecordMemEvent::size_cache["cpu"][place.GetDeviceId()][2];
      }
    } else {
      if (RecordMemEvent::has_initialized["gpu"][place.GetDeviceId()] ==
          false) {
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_CURRENT_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_CURRENT_VALUE(Reserved, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, place.GetDeviceId()));
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()].push_back(
            DEVICE_MEMORY_STAT_PEAK_VALUE(Reserved, place.GetDeviceId()));
        current_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][0];
        current_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][1];
        peak_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][2];
        peak_reserved =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][3];
        RecordMemEvent::has_initialized["gpu"][place.GetDeviceId()] = true;
      } else {
        current_reserved = DEVICE_MEMORY_STAT_CURRENT_VALUE(
            Reserved, place.GetDeviceId());  // NOLINT
        peak_reserved = DEVICE_MEMORY_STAT_PEAK_VALUE(
            Reserved, place.GetDeviceId());  // NOLINT
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][1] =
            current_reserved;
        RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][3] =
            peak_reserved;
        current_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][0];
        peak_allocated =
            RecordMemEvent::size_cache["gpu"][place.GetDeviceId()][2];
      }
    }
    platform::MemEventRecorder::Instance().PopMemRecord(ptr,
                                                        place,
                                                        size,
                                                        type,
                                                        current_allocated,
                                                        current_reserved,
                                                        peak_allocated,
                                                        peak_reserved);
  }
}

void MemEventRecorder::PushMemRecord(const void *ptr,
                                     const Place &place,
                                     size_t size) {
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled) {
    return;
  }
  std::lock_guard<std::mutex> guard(mtx_);
  auto &events = address_memevent_[place];
  PADDLE_ENFORCE_EQ(events.count(ptr),
                    0,
                    common::errors::InvalidArgument(
                        "The Place can't exist in the stage of PushMemRecord"));
  events.emplace(
      ptr, std::make_unique<MemEventRecorder::RecordMemEvent>(place, size));
}

void MemEventRecorder::PushMemRecord(const void *ptr,
                                     const Place &place,
                                     size_t size,
                                     phi::TracerMemEventType type,
                                     uint64_t current_allocated,
                                     uint64_t current_reserved,
                                     uint64_t peak_allocated,
                                     uint64_t peak_reserved) {
  std::lock_guard<std::mutex> guard(mtx_);
  if (FLAGS_enable_host_event_recorder_hook) {  // new MemRecord
    HostEventRecorder<CommonMemEvent>::GetInstance().RecordEvent(
        phi::PosixInNsec(),
        reinterpret_cast<uint64_t>(ptr),
        type,
        size,
        place,
        current_allocated,
        current_reserved,
        peak_allocated,
        peak_reserved);
    return;
  }
  if (type == phi::TracerMemEventType::ReservedAllocate) {
    // old profiler only analyse memory managed by paddle.
    return;
  }
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled) return;
  auto &events = address_memevent_[place];
  PADDLE_ENFORCE_EQ(events.count(ptr),
                    0,
                    common::errors::InvalidArgument(
                        "The Place can't exist in the stage of PushMemRecord"));
  events.emplace(
      ptr, std::make_unique<MemEventRecorder::RecordMemEvent>(place, size));
}

void MemEventRecorder::PopMemRecord(const void *ptr, const Place &place) {
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled) {
    return;
  }
  std::lock_guard<std::mutex> guard(mtx_);
  auto &events = address_memevent_[place];
  auto iter = events.find(ptr);
  // The ptr maybe not in address_memevent
  if (iter != events.end()) {
    events.erase(iter);
  }
}

void MemEventRecorder::PopMemRecord(const void *ptr,
                                    const Place &place,
                                    size_t size,
                                    phi::TracerMemEventType type,
                                    uint64_t current_allocated,
                                    uint64_t current_reserved,
                                    uint64_t peak_allocated,
                                    uint64_t peak_reserved) {
  std::lock_guard<std::mutex> guard(mtx_);
  if (FLAGS_enable_host_event_recorder_hook) {  // new MemRecord
    HostEventRecorder<CommonMemEvent>::GetInstance().RecordEvent(
        phi::PosixInNsec(),
        reinterpret_cast<uint64_t>(ptr),
        type,
        -size,
        place,
        current_allocated,
        current_reserved,
        peak_allocated,
        peak_reserved);
    return;
  }
  if (type == phi::TracerMemEventType::ReservedFree) {
    // old profiler only analyse memory managed by paddle.
    return;
  }
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled) return;
  auto &events = address_memevent_[place];
  auto iter = events.find(ptr);
  // The ptr maybe not in address_memevent
  if (iter != events.end()) {
    events.erase(iter);
  }
}

void MemEventRecorder::Flush() {
  std::lock_guard<std::mutex> guard(mtx_);
  address_memevent_.clear();
}

MemEventRecorder::RecordMemEvent::RecordMemEvent(const Place &place,
                                                 size_t bytes)
    : place_(place),
      bytes_(bytes),
      start_ns_(phi::PosixInNsec()),
      end_ns_(0),
      alloc_in_(phi::CurAnnotationName()) {
  PushMemEvent(start_ns_, end_ns_, bytes_, place_, alloc_in_);
}

MemEventRecorder::RecordMemEvent::~RecordMemEvent() {  // NOLINT
  phi::DeviceTracer *tracer = phi::GetDeviceTracer();
  end_ns_ = phi::PosixInNsec();

  auto annotation_free = phi::CurAnnotationName();
  if (tracer) {
    tracer->AddMemInfoRecord(start_ns_,
                             end_ns_,
                             bytes_,
                             place_,
                             alloc_in_,
                             annotation_free,
                             g_mem_thread_id);
  }
  PopMemEvent(start_ns_, end_ns_, bytes_, place_, annotation_free);
}

RecordBlock::RecordBlock(int block_id)
    : is_enabled_(false), start_ns_(phi::PosixInNsec()) {
  // lock is not needed, the code below is thread-safe
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled) return;
  is_enabled_ = true;
  phi::SetCurBlock(block_id);
  name_ = string::Sprintf("block_%d", block_id);
}

RecordBlock::~RecordBlock() {
  // lock is not needed, the code below is thread-safe
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled || !is_enabled_)
    return;
  phi::DeviceTracer *tracer = phi::GetDeviceTracer();
  if (tracer) {
    // We try to put all blocks at the same nested depth in the
    // same timeline lane. and distinguish the using thread_id.
    tracer->AddCPURecords(name_,
                          start_ns_,
                          phi::PosixInNsec(),
                          phi::BlockDepth(),
                          phi::ProfilerHelper::g_thread_id);
  }
  phi::ClearCurBlock();
}

void PushMemEvent(uint64_t start_ns,
                  uint64_t end_ns,
                  size_t bytes,
                  const Place &place,
                  const std::string &annotation) {
  GetMemEventList().Record(EventType::kPushRange,
                           start_ns,
                           end_ns,
                           bytes,
                           place,
                           g_mem_thread_id,
                           annotation);
}

void PopMemEvent(uint64_t start_ns,
                 uint64_t end_ns,
                 size_t bytes,
                 const Place &place,
                 const std::string &annotation) {
  GetMemEventList().Record(EventType::kPopRange,
                           start_ns,
                           end_ns,
                           bytes,
                           place,
                           g_mem_thread_id,
                           annotation);
}

void Mark(const std::string &name) {
  if (FLAGS_enable_host_event_recorder_hook) {
    HostEventRecorder<CommonEvent>::GetInstance().RecordEvent(
        name, 0, 0, EventRole::kOrdinary, phi::TracerEventType::UserDefined);
    return;
  }
  GetEventList().Record(
      EventType::kMark, name, phi::ProfilerHelper::g_thread_id);
}

void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE_NE(state,
                    ProfilerState::kDisabled,
                    common::errors::InvalidArgument(
                        "Can't enable profiling, since the input state is"
                        "ProfilerState::kDisabled"));
  SynchronizeAllDevice();
  std::lock_guard<std::mutex> l(profiler_mu);
  if (state == phi::ProfilerHelper::g_state) {
    return;
  }
  phi::ProfilerHelper::g_state = state;
  ProfilerOptions option;
  HostTraceLevel::GetInstance().SetLevel(option.trace_level);
  should_send_profile_state = true;
  phi::GetDeviceTracer()->Enable();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::ProfilerHelper::g_state == ProfilerState::kCUDA ||
      phi::ProfilerHelper::g_state == ProfilerState::kAll ||
      phi::ProfilerHelper::g_state == ProfilerState::kCPU) {
    // Generate some dummy events first to reduce the startup overhead.
    DummyKernelAndEvent();
    phi::GetDeviceTracer()->Reset();
  }
#endif
  // Mark the profiling start.
  Mark("_start_profiler_");
}

void ResetProfiler() {
  SynchronizeAllDevice();
  phi::GetDeviceTracer()->Reset();
  MemEventRecorder::Instance().Flush();
  std::lock_guard<std::mutex> guard(
      phi::ProfilerHelper::g_all_event_lists_mutex);
  for (auto &all_event_list : phi::ProfilerHelper::g_all_event_lists) {
    all_event_list->Clear();
  }
  for (auto &all_mem_event_list : phi::ProfilerHelper::g_all_mem_event_lists) {
    all_mem_event_list->Clear();
  }
}

static std::map<uint64_t, phi::ThreadEvents> DockHostEventRecorderHostPart();
static void DockHostEventRecorderDevicePart(
    const std::map<uint64_t, phi::ThreadEvents> &thr_events);

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string &profile_path) {
  SynchronizeAllDevice();
  auto thr_events = DockHostEventRecorderHostPart();
  MemEventRecorder::Instance().Flush();

  std::lock_guard<std::mutex> l(profiler_mu);
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled) return;
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  DealWithShowName();

  phi::DeviceTracer *tracer = phi::GetDeviceTracer();
  if (tracer->IsEnabled()) {
    tracer->Disable();
    DockHostEventRecorderDevicePart(thr_events);
    tracer->GenEventKernelCudaElapsedTime();
    tracer->GenProfile(profile_path);
  }

  std::vector<std::vector<Event>> all_events = GetAllEvents();

  ParseEvents(all_events, true, sorted_key);
  ParseEvents(all_events, false, sorted_key);

  std::vector<std::vector<MemEvent>> all_mem_events = GetMemEvents();
  ParseMemEvents(all_mem_events);

  ResetProfiler();
  phi::ProfilerHelper::g_state = ProfilerState::kDisabled;
  g_tracer_option = TracerOption::kDefault;
  should_send_profile_state = true;
}

void CompleteProfilerEvents(phi::proto::Profile *tracer_profile,
                            std::vector<std::vector<Event>> *time_events,
                            std::vector<std::vector<MemEvent>> *mem_events) {
  SynchronizeAllDevice();
  auto thr_events = DockHostEventRecorderHostPart();
  MemEventRecorder::Instance().Flush();
  std::lock_guard<std::mutex> l(profiler_mu);
  if (phi::ProfilerHelper::g_state == ProfilerState::kDisabled) return;
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  phi::DeviceTracer *tracer = phi::GetDeviceTracer();
  if (tracer->IsEnabled() && tracer_profile != nullptr) {
    tracer->Disable();
    DockHostEventRecorderDevicePart(thr_events);
    tracer->GenEventKernelCudaElapsedTime();
    *tracer_profile = tracer->GetProfile();
  }
  if (time_events != nullptr) {
    *time_events = GetAllEvents();
  }

  if (mem_events != nullptr) {
    *mem_events = GetMemEvents();
  }
  ResetProfiler();
  phi::ProfilerHelper::g_state = ProfilerState::kDisabled;
  g_tracer_option = TracerOption::kDefault;
  should_send_profile_state = true;
}

std::vector<std::vector<Event>> GetAllEvents() {
  std::lock_guard<std::mutex> guard(
      phi::ProfilerHelper::g_all_event_lists_mutex);
  std::vector<std::vector<Event>> result;
  for (auto &all_event_list : phi::ProfilerHelper::g_all_event_lists) {
    result.emplace_back(all_event_list->Reduce());
  }
  return result;
}

bool IsProfileEnabled() {
  return phi::ProfilerHelper::g_state != ProfilerState::kDisabled;
}

bool ShouldSendProfileState() { return should_send_profile_state; }

std::string OpName(const framework::VariableNameMap &name_map,
                   const std::string &type_name) {
  if (platform::GetTracerOption() != platform::TracerOption::kAllOpDetail ||
      !IsProfileEnabled())
    return "";

  std::string ret = type_name + "%";
  for (const auto &map_item : name_map) {
    auto name_outputs = map_item.second;
    if (!name_outputs.empty()) {
      ret.append(name_outputs[0]);
      break;
    }
  }
  ret = ret + "%";

  return ret;
}

void SetTracerOption(TracerOption option) {
  std::lock_guard<std::mutex> l(profiler_mu);
  g_tracer_option = option;
}

platform::TracerOption GetTracerOption() { return g_tracer_option; }

void SetProfileListener() {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(
      1, std::numeric_limits<int>::max());
  profiler_lister_id = dist6(rng);
}

int64_t ListenerId() { return profiler_lister_id; }

void NvprofEnableRecordEvent() {
  SynchronizeAllDevice();
  phi::ProfilerHelper::g_enable_nvprof_hook = true;
}

void NvprofDisableRecordEvent() {
  phi::ProfilerHelper::g_enable_nvprof_hook = false;
}

void EnableHostEventRecorder() { FLAGS_enable_host_event_recorder_hook = true; }

void DisableHostEventRecorder() {
  FLAGS_enable_host_event_recorder_hook = false;
}

void EnableMemoryRecorder() { FLAGS_enable_record_memory = true; }

void DisableMemoryRecorder() { FLAGS_enable_record_memory = false; }

std::string PrintHostEvents() {
  std::ostringstream oss;
  auto host_evt_sec =
      HostEventRecorder<CommonEvent>::GetInstance().GatherEvents();
  for (const auto &thr_evt_sec : host_evt_sec.thr_sections) {
    oss << thr_evt_sec.thread_id << std::endl;
    for (const auto &evt : thr_evt_sec.events) {
      oss << "{ " << evt.name << " | " << evt.start_ns << "ns | " << evt.end_ns
          << "ns | " << (evt.end_ns - evt.start_ns) / 1000.000  // NOLINT
          << "us }" << std::endl;
    }
  }
  return oss.str();
}

static void EmulateEventPushAndPop(
    const HostEventSection<CommonEvent> &host_sec,
    std::map<uint64_t, phi::ThreadEvents> *out) {
  for (const auto &thr_sec : host_sec.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    auto cur_thr_list = std::make_shared<EventList<Event>>();
    phi::ProfilerHelper::g_all_event_lists.emplace_front(cur_thr_list);
    // for nesting events
    std::stack<size_t> evt_stk;
    std::stack<std::string> prefix_stk;
    std::map<uint64_t, size_t> start2evt;
    for (size_t i = 0; i < thr_sec.events.size(); ++i) {
      const auto &evt = thr_sec.events[i];
      start2evt[evt.start_ns] = i;
    }
    auto iter = start2evt.begin();
    // loop events
    for (size_t i = 0; i < thr_sec.events.size(); ++i) {
      const auto &thr_evts = thr_sec.events;
      const auto &evt = thr_evts[i];
      // For nesting events
      while (!evt_stk.empty() && thr_evts[evt_stk.top()].end_ns <= evt.end_ns) {
        evt_stk.pop();
        prefix_stk.pop();
      }
      while (iter != start2evt.end() &&
             thr_evts[iter->second].start_ns < evt.start_ns) {
        if (thr_evts[iter->second].end_ns > evt.start_ns) {
          evt_stk.push(iter->second);
          std::string prefix = thr_evts[iter->second].name;
          if (!prefix_stk.empty()) {
            // prefix = prefix_stk.top() + "/" + prefix;
            prefix.insert(0, "/").insert(0, prefix_stk.top());
          }
          prefix_stk.push(prefix);
        }
        ++iter;
      }
      // Record orig event pair
      std::string name =
          prefix_stk.empty() ? evt.name : prefix_stk.top() + "/" + evt.name;
      const char *attr = (evt.attr == nullptr ? "none" : evt.attr);
      Event *orig_evt = cur_thr_list->Record(
          EventType::kPushRange, name, tid, evt.role, attr);
      (*out)[tid][evt.end_ns] = std::make_pair(orig_evt, evt.start_ns);
      cur_thr_list->Record(EventType::kPopRange, name, tid, evt.role, attr);
    }
  }
}

static void EmulateCPURecordsAdd(
    const HostEventSection<CommonEvent> &host_sec) {
  phi::DeviceTracer *tracer = phi::GetDeviceTracer();
  if (tracer == nullptr) {
    return;
  }
  for (const auto &thr_sec : host_sec.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    for (const auto &evt : thr_sec.events) {
      tracer->AddCPURecords(
          evt.name, evt.start_ns, evt.end_ns, phi::BlockDepth(), tid);
    }
  }
}

static void EmulateCorrelation(
    const std::map<uint64_t, phi::ThreadEvents> &thr_events) {
  phi::DeviceTracer *tracer = phi::GetDeviceTracer();
  if (tracer == nullptr) {
    return;
  }
  tracer->AddAnnotations(thr_events);
}

static std::map<uint64_t, phi::ThreadEvents> DockHostEventRecorderHostPart() {
  std::map<uint64_t, phi::ThreadEvents> thr_events;
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return thr_events;
  }
  auto host_evt_sec =
      HostEventRecorder<CommonEvent>::GetInstance().GatherEvents();
  EmulateEventPushAndPop(host_evt_sec, &thr_events);
  EmulateCPURecordsAdd(host_evt_sec);
  return thr_events;
}

static void DockHostEventRecorderDevicePart(
    const std::map<uint64_t, phi::ThreadEvents> &thr_events) {
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return;
  }
  EmulateCorrelation(thr_events);
}

}  // namespace paddle::platform
