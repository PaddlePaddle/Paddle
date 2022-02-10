/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <mutex>  // NOLINT
#include <random>
#include <sstream>
#include <string>
#include <type_traits>

#include "paddle/fluid/platform/device_tracer.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/common_event.h"
#include "paddle/fluid/platform/profiler/host_event_recorder.h"
#include "paddle/fluid/platform/profiler/host_tracer.h"
#include "paddle/fluid/platform/profiler_helper.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/nvtx.h"
#endif
#include "paddle/fluid/platform/os_info.h"

PADDLE_DEFINE_EXPORTED_bool(enable_rpc_profiler, false,
                            "Enable rpc profiler or not.");

DEFINE_bool(enable_host_event_recorder_hook, false,
            "enable HostEventRecorder, hook Profiler");

namespace paddle {
namespace platform {

MemEvenRecorder MemEvenRecorder::recorder;

Event::Event(EventType type, std::string name, uint32_t thread_id,
             EventRole role, std::string attr)
    : type_(type),
      name_(name),
      thread_id_(thread_id),
      role_(role),
      attr_(attr) {
  cpu_ns_ = GetTimeInNsec();
}

const EventType &Event::type() const { return type_; }

double Event::CpuElapsedMs(const Event &e) const {
  return (e.cpu_ns_ - cpu_ns_) / (1000000.0);
}

double Event::CudaElapsedMs(const Event &e) const {
#ifdef PADDLE_WITH_CUPTI
  return gpu_ns_ / 1000000.0;
#else
  LOG_FIRST_N(WARNING, 1) << "CUDA CUPTI is not enabled";
  return 0;
#endif
}

RecordEvent::RecordEvent(const char *name, const EventRole role,
                         uint32_t level) {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name);
    is_pushed_ = true;
  }
#endif
#endif
  if (FLAGS_enable_host_event_recorder_hook == false) {
    OriginalConstruct(name, role, "none");
    return;
  }
  if (UNLIKELY(HostTraceLevel::GetInstance().NeedTrace(level) == false)) {
    return;
  }
  is_enabled_ = true;
  shallow_copy_name_ = name;
  role_ = role;
  start_ns_ = PosixInNsec();
}

RecordEvent::RecordEvent(const std::string &name, const EventRole role,
                         uint32_t level) {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name.c_str());
    is_pushed_ = true;
  }
#endif
#endif
  if (FLAGS_enable_host_event_recorder_hook == false) {
    OriginalConstruct(name, role, "none");
    return;
  }
  if (UNLIKELY(HostTraceLevel::GetInstance().NeedTrace(level) == false)) {
    return;
  }
  is_enabled_ = true;
  name_ = new std::string(name);
  role_ = role;
  start_ns_ = PosixInNsec();
}

RecordEvent::RecordEvent(const std::string &name, const EventRole role,
                         const std::string &attr, uint32_t level) {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name.c_str());
    is_pushed_ = true;
  }
#endif
#endif
  if (FLAGS_enable_host_event_recorder_hook == false) {
    OriginalConstruct(name, role, attr);
    return;
  }
  if (UNLIKELY(HostTraceLevel::GetInstance().NeedTrace(level) == false)) {
    return;
  }
  is_enabled_ = true;
  name_ = new std::string(name);
  start_ns_ = PosixInNsec();
  attr_ = new std::string(attr);
}

void RecordEvent::OriginalConstruct(const std::string &name,
                                    const EventRole role,
                                    const std::string &attr) {
  if (g_state == ProfilerState::kDisabled || name.empty()) return;

  // do some initialization
  name_ = new std::string(name);
  start_ns_ = PosixInNsec();
  role_ = role;
  attr_ = new std::string(attr);
  is_enabled_ = true;
  // lock is not needed, the code below is thread-safe
  // Maybe need the same push/pop behavior.
  Event *e = PushEvent(name, role, attr);
  SetCurAnnotation(e);
  *name_ = e->name();
}

void RecordEvent::End() {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook && is_pushed_) {
    dynload::nvtxRangePop();
  }
#endif
#endif
  uint64_t end_ns = PosixInNsec();
  if (LIKELY(FLAGS_enable_host_event_recorder_hook && is_enabled_)) {
    if (LIKELY(shallow_copy_name_ != nullptr)) {
      HostEventRecorder::GetInstance().RecordEvent(shallow_copy_name_,
                                                   start_ns_, end_ns, role_,
                                                   TracerEventType::NumTypes);
    } else if (name_ != nullptr) {
      if (attr_ == nullptr) {
        HostEventRecorder::GetInstance().RecordEvent(
            *name_, start_ns_, end_ns, role_, TracerEventType::NumTypes);
      } else {
        HostEventRecorder::GetInstance().RecordEvent(
            *name_, start_ns_, end_ns, role_, TracerEventType::NumTypes,
            *attr_);
        delete attr_;
      }
      delete name_;
    }
    // use this flag to avoid double End();
    is_enabled_ = false;
    return;
  }

  if (g_state == ProfilerState::kDisabled || !is_enabled_) return;
  // lock is not needed, the code below is thread-safe
  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer) {
    tracer->AddCPURecords(CurAnnotationName(), start_ns_, end_ns, BlockDepth(),
                          g_thread_id);
  }
  ClearCurAnnotation();
  PopEvent(*name_, role_);
  delete name_;
  delete attr_;
  // use this flag to avoid double End();
  is_enabled_ = false;
}

RecordInstantEvent::RecordInstantEvent(const char *name, TracerEventType type,
                                       uint32_t level) {
  if (UNLIKELY(HostTraceLevel::GetInstance().NeedTrace(level) == false)) {
    return;
  }
  auto start_end_ns = PosixInNsec();
  HostEventRecorder::GetInstance().RecordEvent(name, start_end_ns, start_end_ns,
                                               EventRole::kOrdinary, type);
}

void MemEvenRecorder::PushMemRecord(const void *ptr, const Place &place,
                                    size_t size) {
  if (g_state == ProfilerState::kDisabled) return;
  std::lock_guard<std::mutex> guard(mtx_);
  auto &events = address_memevent_[place];
  PADDLE_ENFORCE_EQ(events.count(ptr), 0,
                    platform::errors::InvalidArgument(
                        "The Place can't exist in the stage of PushMemRecord"));
  events.emplace(ptr, std::unique_ptr<RecordMemEvent>(
                          new MemEvenRecorder::RecordMemEvent(place, size)));
}

void MemEvenRecorder::PopMemRecord(const void *ptr, const Place &place) {
  if (g_state == ProfilerState::kDisabled) return;
  std::lock_guard<std::mutex> guard(mtx_);
  auto &events = address_memevent_[place];
  auto iter = events.find(ptr);
  // The ptr maybe not in address_memevent
  if (iter != events.end()) {
    events.erase(iter);
  }
}

void MemEvenRecorder::Flush() {
  std::lock_guard<std::mutex> guard(mtx_);
  address_memevent_.clear();
}

MemEvenRecorder::RecordMemEvent::RecordMemEvent(const Place &place,
                                                size_t bytes)
    : place_(place),
      bytes_(bytes),
      start_ns_(PosixInNsec()),
      alloc_in_(CurAnnotationName()) {
  PushMemEvent(start_ns_, end_ns_, bytes_, place_, alloc_in_);
}

MemEvenRecorder::RecordMemEvent::~RecordMemEvent() {
  DeviceTracer *tracer = GetDeviceTracer();
  end_ns_ = PosixInNsec();

  auto annotation_free = CurAnnotationName();
  if (tracer) {
    tracer->AddMemInfoRecord(start_ns_, end_ns_, bytes_, place_, alloc_in_,
                             annotation_free, g_mem_thread_id);
  }
  PopMemEvent(start_ns_, end_ns_, bytes_, place_, annotation_free);
}

/*RecordRPCEvent::RecordRPCEvent(const std::string &name) {
  if (FLAGS_enable_rpc_profiler) {
    event_.reset(new platform::RecordEvent(name));
  }
}*/

RecordBlock::RecordBlock(int block_id)
    : is_enabled_(false), start_ns_(PosixInNsec()) {
  // lock is not needed, the code below is thread-safe
  if (g_state == ProfilerState::kDisabled) return;
  is_enabled_ = true;
  SetCurBlock(block_id);
  name_ = string::Sprintf("block_%d", block_id);
}

RecordBlock::~RecordBlock() {
  // lock is not needed, the code below is thread-safe
  if (g_state == ProfilerState::kDisabled || !is_enabled_) return;
  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer) {
    // We try to put all blocks at the same nested depth in the
    // same timeline lane. and distinguish the using thread_id.
    tracer->AddCPURecords(name_, start_ns_, PosixInNsec(), BlockDepth(),
                          g_thread_id);
  }
  ClearCurBlock();
}

void PushMemEvent(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                  const Place &place, const std::string &annotation) {
  GetMemEventList().Record(EventType::kPushRange, start_ns, end_ns, bytes,
                           place, g_mem_thread_id, annotation);
}

void PopMemEvent(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                 const Place &place, const std::string &annotation) {
  GetMemEventList().Record(EventType::kPopRange, start_ns, end_ns, bytes, place,
                           g_mem_thread_id, annotation);
}

void Mark(const std::string &name) {
  if (FLAGS_enable_host_event_recorder_hook) {
    HostEventRecorder::GetInstance().RecordEvent(
        name, 0, 0, EventRole::kOrdinary, TracerEventType::NumTypes);
    return;
  }
  GetEventList().Record(EventType::kMark, name, g_thread_id);
}

Event *PushEvent(const std::string &name, const EventRole role,
                 std::string attr) {
  return GetEventList().Record(EventType::kPushRange, name, g_thread_id, role,
                               attr);
}

void PopEvent(const std::string &name, const EventRole role, std::string attr) {
  GetEventList().Record(EventType::kPopRange, name, g_thread_id, role, attr);
}
void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE_NE(state, ProfilerState::kDisabled,
                    platform::errors::InvalidArgument(
                        "Can't enable profiling, since the input state is"
                        "ProfilerState::kDisabled"));
  SynchronizeAllDevice();
  std::lock_guard<std::mutex> l(profiler_mu);
  if (state == g_state) {
    return;
  }
  g_state = state;
  should_send_profile_state = true;
  GetDeviceTracer()->Enable();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (g_state == ProfilerState::kCUDA || g_state == ProfilerState::kAll ||
      g_state == ProfilerState::kCPU) {
    // Generate some dummy events first to reduce the startup overhead.
    DummyKernelAndEvent();
    GetDeviceTracer()->Reset();
  }
#endif
  // Mark the profiling start.
  Mark("_start_profiler_");
}

void ResetProfiler() {
  SynchronizeAllDevice();
  GetDeviceTracer()->Reset();
  MemEvenRecorder::Instance().Flush();
  std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
  for (auto it = g_all_event_lists.begin(); it != g_all_event_lists.end();
       ++it) {
    (*it)->Clear();
  }
  for (auto it = g_all_mem_event_lists.begin();
       it != g_all_mem_event_lists.end(); ++it) {
    (*it)->Clear();
  }
}

static std::map<uint64_t, ThreadEvents> DockHostEventRecorderHostPart();
static void DockHostEventRecorderDevicePart(
    const std::map<uint64_t, ThreadEvents> &thr_events);

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string &profile_path) {
  SynchronizeAllDevice();
  auto thr_events = DockHostEventRecorderHostPart();
  MemEvenRecorder::Instance().Flush();

  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  DealWithShowName();

  DeviceTracer *tracer = GetDeviceTracer();
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
  g_state = ProfilerState::kDisabled;
  g_tracer_option = TracerOption::kDefault;
  should_send_profile_state = true;
}

void CompleteProfilerEvents(proto::Profile *tracer_profile,
                            std::vector<std::vector<Event>> *time_events,
                            std::vector<std::vector<MemEvent>> *mem_events) {
  SynchronizeAllDevice();
  auto thr_events = DockHostEventRecorderHostPart();
  MemEvenRecorder::Instance().Flush();

  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;

  // Mark the profiling stop.
  Mark("_stop_profiler_");

  DeviceTracer *tracer = GetDeviceTracer();
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
  g_state = ProfilerState::kDisabled;
  g_tracer_option = TracerOption::kDefault;
  should_send_profile_state = true;
}

std::vector<std::vector<Event>> GetAllEvents() {
  std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
  std::vector<std::vector<Event>> result;
  for (auto it = g_all_event_lists.begin(); it != g_all_event_lists.end();
       ++it) {
    result.emplace_back((*it)->Reduce());
  }
  return result;
}

bool IsProfileEnabled() { return g_state != ProfilerState::kDisabled; }

bool ShouldSendProfileState() { return should_send_profile_state; }

std::string OpName(const framework::VariableNameMap &name_map,
                   const std::string &type_name) {
  if (platform::GetTracerOption() != platform::TracerOption::kAllOpDetail ||
      !IsProfileEnabled())
    return "";

  std::string ret = type_name + "%";
  for (auto it = name_map.begin(); it != name_map.end(); it++) {
    auto name_outputs = it->second;
    if (!name_outputs.empty()) {
      ret = ret + name_outputs[0];
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
  g_enable_nvprof_hook = true;
}

void NvprofDisableRecordEvent() { g_enable_nvprof_hook = false; }

void EnableHostEventRecorder() { FLAGS_enable_host_event_recorder_hook = true; }

std::string PrintHostEvents() {
  std::ostringstream oss;
  auto host_evt_sec = HostEventRecorder::GetInstance().GatherEvents();
  for (const auto &thr_evt_sec : host_evt_sec.thr_sections) {
    oss << thr_evt_sec.thread_id << std::endl;
    for (const auto &evt : thr_evt_sec.events) {
      oss << "{ " << evt.name << " | " << evt.start_ns << "ns | " << evt.end_ns
          << "ns | " << (evt.end_ns - evt.start_ns) / 1000.000 << "us }"
          << std::endl;
    }
  }
  return oss.str();
}

static void EmulateEventPushAndPop(const HostEventSection &host_sec,
                                   std::map<uint64_t, ThreadEvents> *out) {
  for (const auto &thr_sec : host_sec.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    auto cur_thr_list = std::make_shared<EventList<Event>>();
    g_all_event_lists.emplace_front(cur_thr_list);
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
            prefix = prefix_stk.top() + "/" + prefix;
          }
          prefix_stk.push(prefix);
        }
        ++iter;
      }
      // Record orig event pair
      std::string name =
          prefix_stk.empty() ? evt.name : prefix_stk.top() + "/" + evt.name;
      const char *attr = (evt.attr == nullptr ? "none" : evt.attr);
      Event *orig_evt = cur_thr_list->Record(EventType::kPushRange, name, tid,
                                             evt.role, attr);
      (*out)[tid][evt.end_ns] = std::make_pair(orig_evt, evt.start_ns);
      cur_thr_list->Record(EventType::kPopRange, name, tid, evt.role, attr);
    }
  }
}

static void EmulateCPURecordsAdd(const HostEventSection &host_sec) {
  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer == nullptr) {
    return;
  }
  for (const auto &thr_sec : host_sec.thr_sections) {
    uint64_t tid = thr_sec.thread_id;
    for (const auto &evt : thr_sec.events) {
      tracer->AddCPURecords(evt.name, evt.start_ns, evt.end_ns, BlockDepth(),
                            tid);
    }
  }
}

static void EmulateCorrelation(
    const std::map<uint64_t, ThreadEvents> &thr_events) {
  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer == nullptr) {
    return;
  }
  tracer->AddAnnotations(thr_events);
}

static std::map<uint64_t, ThreadEvents> DockHostEventRecorderHostPart() {
  std::map<uint64_t, ThreadEvents> thr_events;
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return thr_events;
  }
  auto host_evt_sec = HostEventRecorder::GetInstance().GatherEvents();
  EmulateEventPushAndPop(host_evt_sec, &thr_events);
  EmulateCPURecordsAdd(host_evt_sec);
  return std::move(thr_events);
}

static void DockHostEventRecorderDevicePart(
    const std::map<uint64_t, ThreadEvents> &thr_events) {
  if (FLAGS_enable_host_event_recorder_hook == false) {
    return;
  }
  EmulateCorrelation(thr_events);
}

}  // namespace platform
}  // namespace paddle
