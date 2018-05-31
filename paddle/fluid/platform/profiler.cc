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

#include "paddle/fluid/platform/profiler.h"
#include <sys/time.h>
#include <time.h>
#include <algorithm>
#include <iomanip>
#include <map>
#include <mutex>  // NOLINT
#include <string>
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif  // PADDLE_WITH_CUDA
#include "glog/logging.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/platform/device_tracer.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace platform {

struct EventList;

// The profiler state, the initial value is ProfilerState::kDisabled
static ProfilerState g_state = ProfilerState::kDisabled;
// The thread local event list only can be accessed by the specific thread
// The thread index of each thread
static thread_local int32_t g_thread_id;
// The g_next_thread_id is a global counter for threads, by the g_thread_id and
// g_next_thread_id, we can know how many threads have created EventList.
static uint32_t g_next_thread_id = 0;
// The global mutex
static std::mutex g_all_event_lists_mutex;
// The total event lists of all threads
static std::list<std::shared_ptr<EventList>> g_all_event_lists;
// The thread local event list only can be accessed by the specific thread
static thread_local std::shared_ptr<EventList> g_event_list;

struct EventList {
  constexpr static size_t kMB = 1024 * 1024;
  constexpr static size_t kEventBlockSize = 16 * kMB;
  constexpr static size_t kEventSize = sizeof(Event);
  constexpr static size_t kEventAlign = alignof(Event);
  constexpr static size_t kNumBlock =
      kEventBlockSize /
      ((kEventSize + kEventAlign - 1) / kEventAlign * kEventAlign);

  template <typename... Args>
  void Record(Args&&... args) {
    if (event_blocks.empty() || event_blocks.front().size() == kNumBlock) {
      event_blocks.emplace_front();
      event_blocks.front().reserve(kNumBlock);
    }
    event_blocks.front().emplace_back(std::forward<Args>(args)...);
  }

  std::vector<Event> Reduce() {
    std::vector<Event> result;
    for (auto& block : event_blocks) {
      result.insert(result.begin(), std::make_move_iterator(block.begin()),
                    std::make_move_iterator(block.end()));
    }
    event_blocks.clear();
    return result;
  }

  void Clear() { event_blocks.clear(); }

  std::forward_list<std::vector<Event>> event_blocks;
};

inline uint64_t GetTimeInNsec() {
  using clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                 std::chrono::high_resolution_clock,
                                 std::chrono::steady_clock>::type;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

inline uint64_t PosixInNsec() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
}

Event::Event(EventType type, std::string name, uint32_t thread_id,
             const DeviceContext* dev_ctx)
    : type_(type), name_(name), thread_id_(thread_id), has_cuda_(false) {
#ifdef PADDLE_WITH_CUDA
  has_cuda_ = dev_ctx ? platform::is_gpu_place(dev_ctx->GetPlace()) : false;
  if (has_cuda_) {
    auto* cuda_dev_ctx = static_cast<const CUDADeviceContext*>(dev_ctx);
    PADDLE_ENFORCE(cudaGetDevice(&device_));
    PADDLE_ENFORCE(cudaEventCreate(&event_));
    auto stream = cuda_dev_ctx->stream();
    PADDLE_ENFORCE(cudaEventRecord(event_, stream));
  }
#endif
  cpu_ns_ = GetTimeInNsec();
}

const EventType& Event::type() const { return type_; }

double Event::CpuElapsedMs(const Event& e) const {
  return (e.cpu_ns_ - cpu_ns_) / (1000000.0);
}

double Event::CudaElapsedMs(const Event& e) const {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE(e.has_cuda() && has_cuda());
  PADDLE_ENFORCE(e.device() == device());
  PADDLE_ENFORCE(cudaEventSynchronize(event_));
  PADDLE_ENFORCE(cudaEventSynchronize(e.event()));
  float ms;
  PADDLE_ENFORCE(cudaEventElapsedTime(&ms, event_, e.event()));
  return ms;
#else
  PADDLE_THROW("CUDA is not enabled");
#endif
}

#ifdef PADDLE_WITH_CUDA
static void ForEachDevice(std::function<void(int)> func) {
  auto original_device = GetCurrentDeviceId();
  int count = GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    SetDeviceId(i);
    func(i);
  }
  SetDeviceId(original_device);
}
#endif

inline EventList& GetEventList() {
  if (!g_event_list) {
    std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
    g_event_list = std::make_shared<EventList>();
    g_thread_id = g_next_thread_id++;
    g_all_event_lists.emplace_front(g_event_list);
  }
  return *g_event_list;
}

void Mark(const std::string& name, const DeviceContext* dev_ctx) {
  GetEventList().Record(EventType::kMark, name, g_thread_id, dev_ctx);
}

void PushEvent(const std::string& name, const DeviceContext* dev_ctx) {
  GetEventList().Record(EventType::kPushRange, name, g_thread_id, dev_ctx);
}

void PopEvent(const std::string& name, const DeviceContext* dev_ctx) {
  GetEventList().Record(EventType::kPopRange, name, g_thread_id, dev_ctx);
}

RecordEvent::RecordEvent(const std::string& name, const DeviceContext* dev_ctx)
    : start_ns_(PosixInNsec()) {
  if (g_state == ProfilerState::kDisabled) return;
  dev_ctx_ = dev_ctx;
  name_ = name;
  PushEvent(name_, dev_ctx_);
  // Maybe need the same push/pop behavior.
  SetCurAnnotation(name_);
}

RecordEvent::~RecordEvent() {
  if (g_state == ProfilerState::kDisabled) return;
  DeviceTracer* tracer = GetDeviceTracer();
  if (tracer) {
    tracer->AddCPURecords(CurAnnotation(), start_ns_, PosixInNsec(),
                          BlockDepth(), CurThread());
  }
  ClearCurAnnotation();
  PopEvent(name_, dev_ctx_);
}

RecordBlock::RecordBlock(int block_id) : start_ns_(PosixInNsec()) {
  if (g_state == ProfilerState::kDisabled) return;
  SetCurBlock(block_id);
  name_ = string::Sprintf("block_%d", block_id);
}

RecordBlock::~RecordBlock() {
  if (g_state == ProfilerState::kDisabled) return;
  DeviceTracer* tracer = GetDeviceTracer();
  if (tracer) {
    // We try to put all blocks at the same nested depth in the
    // same timeline lane. and distinguish the using thread_id.
    tracer->AddCPURecords(name_, start_ns_, PosixInNsec(), BlockDepth(),
                          CurThread());
  }
  ClearCurBlock();
}

RecordThread::RecordThread(int thread_id) {
  if (g_state == ProfilerState::kDisabled) return;
  SetCurThread(thread_id);
}

RecordThread::~RecordThread() {
  if (g_state == ProfilerState::kDisabled) return;
  ClearCurThread();
}

void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE(state != ProfilerState::kDisabled,
                 "Can't enbale profling, since the input state is ",
                 "ProfilerState::kDisabled");
  PADDLE_ENFORCE(g_state == ProfilerState::kDisabled,
                 "The profiling state should be disabled when calling ",
                 "EnableProfiler.");
  g_state = state;
  if (g_state == ProfilerState::kAll) {
    GetDeviceTracer()->Enable();
  }
#ifdef PADDLE_WITH_CUDA
  if (g_state == ProfilerState::kCUDA) {
    // Generate some dummy events first to reduce the startup overhead.
    for (int i = 0; i < 5; i++) {
      ForEachDevice([](int d) {
        DeviceContext* dev_ctx = new CUDADeviceContext(CUDAPlace(d));
        Mark("_cuda_startup_", dev_ctx);
        dev_ctx->Wait();
        delete dev_ctx;
      });
    }
  }
#endif
  // Mark the profiling start.
  Mark("_start_profiler_", nullptr);
}

void ResetProfiler() {
  std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
  for (auto it = g_all_event_lists.begin(); it != g_all_event_lists.end();
       ++it) {
    (*it)->Clear();
  }
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

// The information of each event given in the profiling report
struct EventItem {
  std::string name;
  int calls;
  double total_time;
  double min_time;
  double max_time;
  double ave_time;
};

// Print results
void PrintProfiler(const std::vector<std::vector<EventItem>>& events_table,
                   const std::string& sorted_domain, const size_t name_width,
                   const size_t data_width) {
  // Output header information
  std::cout << "\n------------------------->"
            << "     Profiling Report     "
            << "<-------------------------\n\n";
  std::string place;
  if (g_state == ProfilerState::kCPU) {
    place = "CPU";
  } else if (g_state == ProfilerState::kCUDA) {
    place = "CUDA";
  } else if (g_state == ProfilerState::kAll) {
    place = "All";
  } else {
    PADDLE_THROW("Invalid profiler state");
  }

  std::cout << "Place: " << place << std::endl;
  std::cout << "Time unit: ms" << std::endl;
  std::cout << "Sorted by " << sorted_domain
            << " in descending order in the same thread\n\n";
  // Output events table
  std::cout.setf(std::ios::left);
  std::cout << std::setw(name_width) << "Event" << std::setw(data_width)
            << "Calls" << std::setw(data_width) << "Total"
            << std::setw(data_width) << "Min." << std::setw(data_width)
            << "Max." << std::setw(data_width) << "Ave." << std::endl;
  for (size_t i = 0; i < events_table.size(); ++i) {
    for (size_t j = 0; j < events_table[i].size(); ++j) {
      const EventItem& event_item = events_table[i][j];
      std::cout << std::setw(name_width) << event_item.name
                << std::setw(data_width) << event_item.calls
                << std::setw(data_width) << event_item.total_time
                << std::setw(data_width) << event_item.min_time
                << std::setw(data_width) << event_item.max_time
                << std::setw(data_width) << event_item.ave_time << std::endl;
    }
  }
  std::cout << std::endl;
}

// Parse the event list and output the profiling report
void ParseEvents(const std::vector<std::vector<Event>>& events,
                 EventSortingKey sorted_by = EventSortingKey::kDefault) {
  if (g_state == ProfilerState::kDisabled) return;

  std::string sorted_domain;
  std::function<bool(const EventItem&, const EventItem&)> sorted_func;
  switch (sorted_by) {
    case EventSortingKey::kCalls:
      sorted_domain = "number of calls";
      sorted_func = [](const EventItem& a, const EventItem& b) {
        return a.calls > b.calls;
      };
      break;
    case EventSortingKey::kTotal:
      sorted_domain = "total time";
      sorted_func = [](const EventItem& a, const EventItem& b) {
        return a.total_time > b.total_time;
      };
      break;
    case EventSortingKey::kMin:
      sorted_domain = "minimum time";
      sorted_func = [](const EventItem& a, const EventItem& b) {
        return a.min_time > b.min_time;
      };
      break;
    case EventSortingKey::kMax:
      sorted_domain = "maximum time";
      sorted_func = [](const EventItem& a, const EventItem& b) {
        return a.max_time > b.max_time;
      };
      break;
    case EventSortingKey::kAve:
      sorted_domain = "average time";
      sorted_func = [](const EventItem& a, const EventItem& b) {
        return a.ave_time > b.ave_time;
      };
      break;
    default:
      sorted_domain = "event first end time";
  }

  std::vector<std::vector<EventItem>> events_table;
  size_t max_name_width = 0;
  for (size_t i = 0; i < events.size(); i++) {
    std::list<Event> pushed_events;
    std::vector<EventItem> event_items;
    std::unordered_map<std::string, int> event_idx;

    for (size_t j = 0; j < events[i].size(); j++) {
      if (events[i][j].type() == EventType::kPushRange) {
        pushed_events.push_back(events[i][j]);
      } else if (events[i][j].type() == EventType::kPopRange) {
        std::list<Event>::reverse_iterator rit = pushed_events.rbegin();
        while (rit != pushed_events.rend() &&
               rit->name() != events[i][j].name()) {
          ++rit;
        }

        if (rit != pushed_events.rend()) {
          double event_time = (g_state == ProfilerState::kCUDA ||
                               g_state == ProfilerState::kAll)
                                  ? rit->CudaElapsedMs(events[i][j])
                                  : rit->CpuElapsedMs(events[i][j]);

          std::string event_name =
              "thread" + std::to_string(rit->thread_id()) + "::" + rit->name();
          max_name_width = std::max(max_name_width, event_name.size());

          if (event_idx.find(event_name) == event_idx.end()) {
            event_idx[event_name] = event_items.size();
            EventItem event_item = {event_name, 1,          event_time,
                                    event_time, event_time, event_time};
            event_items.push_back(event_item);
          } else {
            int index = event_idx[event_name];
            event_items[index].calls += 1;
            // total time
            event_items[index].total_time += event_time;
            // min time
            event_items[index].min_time =
                std::min(event_time, event_items[index].min_time);
            // max time
            event_items[index].max_time =
                std::max(event_time, event_items[index].max_time);
          }

          // remove the push marker from the list
          pushed_events.erase((++rit).base());
        } else {
          LOG(WARNING) << "Cannot find the push marker of event \'"
                       << events[i][j].name()
                       << "\', which will be ignored in profiling report.";
        }
      }
    }
    // average time
    for (auto& item : event_items) {
      item.ave_time = item.total_time / item.calls;
    }
    // sort
    if (sorted_by != EventSortingKey::kDefault) {
      std::sort(event_items.begin(), event_items.end(), sorted_func);
    }

    events_table.push_back(event_items);
    // log warning if there are events with `push` but without `pop`
    std::list<Event>::reverse_iterator rit = pushed_events.rbegin();
    while (rit != pushed_events.rend()) {
      LOG(WARNING) << "Cannot find the pop marker of event \'" << rit->name()
                   << "\', which will be ignored in profiling report.";
      ++rit;
    }
  }

  // Print report
  PrintProfiler(events_table, sorted_domain, max_name_width + 4, 12);
}

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string& profile_path) {
  PADDLE_ENFORCE(g_state != ProfilerState::kDisabled,
                 "Can't disable profiling, since it's not starting.");
  // Mark the profiling stop.
  Mark("_stop_profiler_", nullptr);

  std::vector<std::vector<Event>> all_events = GetAllEvents();
  ParseEvents(all_events, sorted_key);
  ResetProfiler();
  DeviceTracer* tracer = GetDeviceTracer();
  if (g_state == ProfilerState::kAll && tracer && tracer->IsEnabled()) {
    tracer->Disable();
    tracer->GenProfile(profile_path);
  }
  g_state = ProfilerState::kDisabled;
}

}  // namespace platform
}  // namespace paddle
