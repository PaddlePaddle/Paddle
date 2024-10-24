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

#pragma once
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/supplement_tracing.h"
#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/phi/core/platform/profiler/mem_tracing.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#include "paddle/phi/api/profiler/profiler.h"
#include "paddle/utils/test_macros.h"

namespace phi {

namespace proto {
class Profile;
}
}  // namespace phi

namespace paddle {
namespace platform {

using phi::Event;
using phi::EventRole;
using phi::EventType;
using phi::MemEvent;

namespace proto {
class Profile;
}

const int kEnableProfiler = 1;
const int kDisableProfiler = 2;

using ProfilerState = phi::ProfilerState;
using TracerOption = phi::TracerOption;

// Candidate keys to sort the profiling report
enum class EventSortingKey {
  kDefault,
  kCalls,
  kTotal,
  kMin,
  kMax,
  kAve,
  kCPUTime,
  kGPUTime
};

struct MemoryProfilerReport {
  size_t alloc_times{0};
  size_t alloc_size{0};
  size_t free_times{0};
  size_t free_size{0};
};

// The information of each event given in the profiling report
struct EventItem {
  std::string name;
  int calls;
  double total_time;
  double max_time;
  double ave_time;
  double min_time;
  double cpu_time;
  double gpu_time;
  float ratio;
  EventRole role;
};

struct OverHead {
  bool print_overhead = false;
  bool print_explanation = false;
  double elapsed_time = 0.;      // the elapsed time of all events
  double accumulated_time = 0.;  // the accumulated time of all events
  double compute_time = 0.0;
  double framework_time = 0.0;
  EventItem memcpy_item;
  std::vector<EventItem> sub_memcpy_items;
};

struct MemEventRecorder {
 public:
  void PushMemRecord(const void* ptr, const Place& place, size_t size);
  void PopMemRecord(const void* ptr, const Place& place);
  void PushMemRecord(const void* ptr,
                     const Place& place,
                     size_t size,
                     phi::TracerMemEventType type,
                     uint64_t current_allocated,
                     uint64_t current_reserved,
                     uint64_t peak_allocated,
                     uint64_t peak_reserved);
  void PopMemRecord(const void* ptr,
                    const Place& place,
                    size_t size,
                    phi::TracerMemEventType type,
                    uint64_t current_allocated,
                    uint64_t current_reserved,
                    uint64_t peak_allocated,
                    uint64_t peak_reserved);
  void Flush();
  static MemEventRecorder& Instance() { return recorder; }

 private:
  struct RecordMemEvent {
    RecordMemEvent(const Place& place, size_t bytes);
    ~RecordMemEvent();

    Place place_;
    size_t bytes_;
    uint64_t start_ns_;
    uint64_t end_ns_;
    std::string alloc_in_;
    std::string free_in_;
  };

  static MemEventRecorder recorder;
  std::map<Place,
           std::unordered_map<const void*, std::unique_ptr<RecordMemEvent>>>
      address_memevent_;
  std::mutex mtx_;
  MemEventRecorder() {}
  DISABLE_COPY_AND_ASSIGN(MemEventRecorder);
};

struct RecordBlock {
  explicit RecordBlock(int block_id);
  ~RecordBlock();

 private:
  bool is_enabled_;
  std::string name_;
  uint64_t start_ns_;
};

template <typename T>
using EventList = phi::EventList<T>;

void Mark(const std::string& name);
void PushMemEvent(uint64_t start_ns,
                  uint64_t end_ns,
                  size_t bytes,
                  const Place& place,
                  const std::string& annotation);
void PopMemEvent(uint64_t start_ns,
                 uint64_t end_ns,
                 size_t bytes,
                 const Place& place,
                 const std::string& annotation);

using phi::PopEvent;
using phi::PushEvent;

// Return the event list of all threads. Assumed the returned value calls
// event_lists, event_lists[i][j] represents the j-th Event of i-th thread.
std::vector<std::vector<Event>> GetAllEvents();

// Enable the profiling function.
TEST_API void EnableProfiler(ProfilerState state);
// Clear the phi::ProfilerHelper::g_all_event_lists, which is total event lists
// of all threads.
TEST_API void ResetProfiler();
TEST_API void DisableProfiler(EventSortingKey sorted_key,
                              const std::string& profile_path);
// Disable profiler but return events instead of print it.
void CompleteProfilerEvents(phi::proto::Profile* tracer_profile,
                            std::vector<std::vector<Event>>* time_events,
                            std::vector<std::vector<MemEvent>>* mem_events);

// Test if the profiler is currently enabled.
bool IsProfileEnabled();
// Whether the trainer should send profiling state to PS.
bool ShouldSendProfileState();
std::string OpName(const framework::VariableNameMap& name_map,
                   const std::string& type_name);
void SetTracerOption(TracerOption option);
platform::TracerOption GetTracerOption();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void DummyKernelAndEvent();
#endif

// Mark current process as PS by assigning a lister id.
void SetProfileListener();
int64_t ListenerId();

void NvprofEnableRecordEvent();
void NvprofDisableRecordEvent();

void EnableHostEventRecorder();
void DisableHostEventRecorder();

void EnableMemoryRecorder();
void DisableMemoryRecorder();

// Defined for UT
std::string PrintHostEvents();

}  // namespace platform
}  // namespace paddle
