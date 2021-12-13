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
#include "paddle/fluid/platform/profiler_helper.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/nvtx.h"
#endif

PADDLE_DEFINE_EXPORTED_bool(enable_rpc_profiler, false,
                            "Enable rpc profiler or not.");

namespace paddle {
namespace platform {

struct DurationEvent {
 public:
  DurationEvent(const char *name, uint64_t start_ns, uint64_t end_ns,
                EventRole role)
      : name(name), start_ns(start_ns), end_ns(end_ns), role(role) {}

  DurationEvent(std::function<void *(size_t)> &arena_allocator,
                const std::string &name_str, uint64_t start_ns, uint64_t end_ns,
                EventRole role, const std::string &attr_str)
      : start_ns(start_ns), end_ns(end_ns), role(role) {
    auto buf = static_cast<char *>(arena_allocator(name_str.length() + 1));
    strncpy(buf, name_str.c_str(), name_str.length() + 1);
    name = buf;
    buf = static_cast<char *>(arena_allocator(attr_str.length() + 1));
    strncpy(buf, attr_str.c_str(), attr_str.length() + 1);
    attr = buf;
  }

  DurationEvent(const std::function<void *(size_t)> &arena_allocator,
                const std::string &name_str, uint64_t start_ns, uint64_t end_ns,
                EventRole role)
      : start_ns(start_ns), end_ns(end_ns), role(role) {
    auto buf = static_cast<char *>(arena_allocator(name_str.length() + 1));
    strncpy(buf, name_str.c_str(), name_str.length() + 1);
    name = buf;
  }

  const char *name = nullptr;  // not owned, designed for performance
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  EventRole role = EventRole::kOrdinary;
  const char *attr = nullptr;  // not owned, designed for performance
};

template <typename HeadType, typename... RestTypes>
struct ContainsStdString
    : std::conditional_t<
          std::is_same<std::string, std::remove_cv_t<std::remove_reference_t<
                                        HeadType>>>::value,
          std::true_type, ContainsStdString<RestTypes...>> {};

template <typename TailType>
struct ContainsStdString<TailType>
    : std::is_same<std::string,
                   std::remove_cv_t<std::remove_reference_t<TailType>>> {};

template <typename EventType>
class EventContainer {
 public:
  EventContainer() {
    event_blocks_ = cur_event_block_ = new EventBlock;
    str_blocks_ = cur_str_block_ = new StringBlock;
  }
  ~EventContainer() {
    Reduce();
    delete event_blocks_;
    for (auto cur = str_blocks_; cur != nullptr;) {
      auto next = cur->next;
      delete cur;
      cur = next;
    }
  }
  DISABLE_COPY_AND_ASSIGN(EventContainer);

 public:
  // Record an event
  template <typename... Args>
  void Record(Args &&... args) {
    DoRecord(ContainsStdString<Args...>(), std::forward<Args>(args)...);
  }

  // Get all events and clear the container
  std::vector<EventType> Reduce();

  // Return a buffer to store the string attribute of Event.
  // HostEventRecorder locates in the static data section.
  // So it's safe to use arena to avoid fragmented allocations.
  char *GetStrBufFromArena(size_t size) { return GetStringStorage(size); }

 private:
  struct EventBlock {
    union InitDeferedEvent {
      InitDeferedEvent() {}
      ~InitDeferedEvent() {}

      EventType event;
    };

    static constexpr size_t kBlockSize = 1 << 24;  // 16 MB
    static constexpr size_t kAvailSize =
        kBlockSize - sizeof(size_t) - sizeof(nullptr);
    static constexpr size_t kNumEvents = kAvailSize / sizeof(InitDeferedEvent);
    static constexpr size_t kPadSize =
        kAvailSize - kNumEvents * sizeof(InitDeferedEvent);
    static constexpr size_t kMinimumEventsPerBlock = 1024;
    static_assert(
        kNumEvents >= kMinimumEventsPerBlock,
        "EventType is too large for kBlockSize, make kBlockSize larger");

    size_t offset = 0;
    EventBlock *next = nullptr;
    InitDeferedEvent events[kNumEvents];
    char padding[kPadSize];
  };
  static_assert(sizeof(EventBlock) == EventBlock::kBlockSize,
                "sizeof EventBlock must equal to kBlockSize");

  struct StringBlock {
    static constexpr size_t kBlockSize = 1 << 22;  // 4 MB
    static constexpr size_t kAvailSize =
        kBlockSize - sizeof(size_t) - sizeof(nullptr);

    size_t offset = 0;
    StringBlock *next = nullptr;
    char storage[kAvailSize];
  };
  static_assert(sizeof(StringBlock) == StringBlock::kBlockSize,
                "sizeof StringBlock must equal to kBlockSize");

  // Record an event with string arguments
  template <typename... Args>
  void DoRecord(std::true_type, Args &&... args) {
    auto *storage = GetEventStorage();
    std::function<void *(size_t)> allocator = [this](size_t size) {
      return GetStrBufFromArena(size);
    };
    new (storage) EventType(allocator, std::forward<Args>(args)...);
  }

  // Record an event without any string argument
  template <typename... Args>
  void DoRecord(std::false_type, Args &&... args) {
    auto *storage = GetEventStorage();
    new (storage) EventType(std::forward<Args>(args)...);
  }

  EventType *GetEventStorage();

  char *GetStringStorage(size_t sz);

  EventBlock *event_blocks_ = nullptr;
  EventBlock *cur_event_block_ = nullptr;
  StringBlock *str_blocks_ = nullptr;
  StringBlock *cur_str_block_ = nullptr;
};

template <typename EventType>
std::vector<EventType> EventContainer<EventType>::Reduce() {
  std::vector<EventType> all_events;
  size_t event_cnt = 0;
  for (auto cur = event_blocks_; cur != nullptr; cur = cur->next) {
    event_cnt += cur->offset;
  }
  all_events.reserve(event_cnt);
  for (auto cur = event_blocks_; cur != nullptr;) {
    for (size_t i = 0; i < cur->offset; ++i) {
      all_events.emplace_back(cur->events[i].event);
    }
    auto next = cur->next;
    delete cur;
    cur = next;
  }
  event_blocks_ = cur_event_block_ = new EventBlock;
  return std::move(all_events);
}

template <typename EventType>
EventType *EventContainer<EventType>::GetEventStorage() {
  if (UNLIKELY(cur_event_block_->offset >=
               EventBlock::kNumEvents)) {  // another block
    cur_event_block_->next = new EventBlock;
    cur_event_block_ = cur_event_block_->next;
  }
  auto &obj = cur_event_block_->events[cur_event_block_->offset].event;
  ++cur_event_block_->offset;
  return &obj;
}

template <typename EventType>
char *EventContainer<EventType>::GetStringStorage(size_t sz) {
  if (UNLIKELY(cur_str_block_->offset + sz >
               StringBlock::kAvailSize)) {  // another block
    cur_str_block_->next = new StringBlock;
    cur_str_block_ = cur_str_block_->next;
  }
  char *storage = cur_str_block_->storage + cur_str_block_->offset;
  cur_str_block_->offset += sz;
  return storage;
}

struct ThreadEventSection {
  std::string thread_name;
  uint64_t thread_id;
  std::vector<DurationEvent> events;
};

class ThreadEventRecorder {
 public:
  ThreadEventRecorder();
  DISABLE_COPY_AND_ASSIGN(ThreadEventRecorder);

 public:
  // Forward call to EventContainer::Record
  template <typename... Args>
  void RecordEvent(Args &&... args) {
    base_evt_cntr_.Record(std::forward<Args>(args)...);
  }

  ThreadEventSection GatherEvents() {
    ThreadEventSection thr_sec;
    thr_sec.thread_name = thread_name_;
    thr_sec.thread_id = thread_id_;
    thr_sec.events = std::move(base_evt_cntr_.Reduce());
    return std::move(thr_sec);
  }

 private:
  uint64_t thread_id_;
  std::string thread_name_;
  EventContainer<DurationEvent> base_evt_cntr_;
};

struct HostEventSection {
  std::string process_name;
  uint64_t process_id;
  std::vector<ThreadEventSection> thr_sections;
};

class HostEventRecorder {
 public:
  // singleton
  static HostEventRecorder &GetInstance() {
    static HostEventRecorder instance;
    return instance;
  }

  // If your string argument has a longer lifetime than the Event,
  // use 'const char*'. e.g.: string literal, op name, etc.
  // Do your best to avoid using 'std::string' as the argument type.
  // It will cause deep-copy to harm performance.
  template <typename... Args>
  void RecordEvent(Args &&... args) {
    GetThreadLocalRecorder().RecordEvent(std::forward<Args>(args)...);
  }

  // Poor performance, call it at the ending
  HostEventSection GatherEvents();

  void RegisterThreadRecorder(uint64_t tid, ThreadEventRecorder *recorder) {
    const std::lock_guard<std::mutex> guard(thread_recorders_lock_);
    thread_recorders_[tid] = recorder;
  }

 private:
  HostEventRecorder() = default;
  DISABLE_COPY_AND_ASSIGN(HostEventRecorder);

  ThreadEventRecorder &GetThreadLocalRecorder() {
    static thread_local ThreadEventRecorder tls_recorder;
    return tls_recorder;
  }

  std::mutex thread_recorders_lock_;
  std::unordered_map<uint64_t, ThreadEventRecorder *> thread_recorders_;
};

static uint64_t GetThreadId() {
  return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

ThreadEventRecorder::ThreadEventRecorder() {
  thread_id_ = GetThreadId();
  HostEventRecorder::GetInstance().RegisterThreadRecorder(thread_id_, this);
}

HostEventSection HostEventRecorder::GatherEvents() {
  HostEventSection host_sec;
  host_sec.thr_sections.reserve(thread_recorders_.size());
  for (auto &kv : thread_recorders_) {
    host_sec.thr_sections.emplace_back(std::move(kv.second->GatherEvents()));
  }
  return std::move(host_sec);
}

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

RecordEvent::RecordEvent(const char *name, const EventRole role) {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name);
    is_pushed_ = true;
  }
#endif
#endif
  if (UNLIKELY(g_enable_host_event_recorder_hook == false)) {
    OriginalConstruct(name, role, "none");
    return;
  }
  shallow_copy_name_ = name;
  role_ = role;
  start_ns_ = PosixInNsec();
}

RecordEvent::RecordEvent(const std::string &name, const EventRole role) {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name.c_str());
    is_pushed_ = true;
  }
#endif
#endif
  if (UNLIKELY(g_enable_host_event_recorder_hook == false)) {
    OriginalConstruct(name, role, "none");
    return;
  }
  name_ = new std::string(name);
  role_ = role;
  start_ns_ = PosixInNsec();
}

RecordEvent::RecordEvent(const std::string &name, const EventRole role,
                         const std::string &attr) {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook) {
    dynload::nvtxRangePushA(name.c_str());
    is_pushed_ = true;
  }
#endif
#endif
  if (UNLIKELY(g_enable_host_event_recorder_hook == false)) {
    OriginalConstruct(name, role, attr);
    return;
  }
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

RecordEvent::~RecordEvent() {
#ifndef _WIN32
#ifdef PADDLE_WITH_CUDA
  if (g_enable_nvprof_hook && is_pushed_) {
    dynload::nvtxRangePop();
  }
#endif
#endif
  uint64_t end_ns = PosixInNsec();
  if (LIKELY(g_enable_host_event_recorder_hook)) {
    if (LIKELY(shallow_copy_name_ != nullptr)) {
      HostEventRecorder::GetInstance().RecordEvent(shallow_copy_name_,
                                                   start_ns_, end_ns, role_);
    } else if (name_ != nullptr) {
      if (attr_ == nullptr) {
        HostEventRecorder::GetInstance().RecordEvent(*name_, start_ns_, end_ns,
                                                     role_);
      } else {
        HostEventRecorder::GetInstance().RecordEvent(*name_, start_ns_, end_ns,
                                                     role_, *attr_);
        delete attr_;
      }
      delete name_;
    }
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

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string &profile_path) {
  SynchronizeAllDevice();
  MemEvenRecorder::Instance().Flush();

  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  DealWithShowName();

  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer->IsEnabled()) {
    tracer->Disable();
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
  MemEvenRecorder::Instance().Flush();

  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;

  // Mark the profiling stop.
  Mark("_stop_profiler_");

  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer->IsEnabled() && tracer_profile != nullptr) {
    tracer->Disable();
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

void EnableHostEventRecorder() { g_enable_host_event_recorder_hook = true; }

std::string PrintHostEvents() {
  std::ostringstream oss;
  auto host_evt_sec = HostEventRecorder::GetInstance().GatherEvents();
  for (const auto &thr_evt_sec : host_evt_sec.thr_sections) {
    oss << thr_evt_sec.thread_id << std::endl;
    for (const auto &evt : thr_evt_sec.events) {
      oss << "{ " << evt.name << " | " << evt.start_ns << " | " << evt.end_ns
          << " }" << std::endl;
    }
  }
  return oss.str();
}

}  // namespace platform
}  // namespace paddle
