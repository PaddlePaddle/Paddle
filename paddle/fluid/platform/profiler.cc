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
#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>  // NOLINT
#include <random>
#include <string>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif  // PADDLE_WITH_CUDA

#include "glog/logging.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/platform/device_tracer.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/printf.h"

DEFINE_bool(enable_rpc_profiler, false, "Enable rpc profiler or not.");

namespace paddle {
namespace platform {

static int64_t profiler_lister_id = 0;
static bool should_send_profile_state = false;
std::mutex profiler_mu;

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
static std::list<std::shared_ptr<EventList<Event>>> g_all_event_lists;
// The thread local event list only can be accessed by the specific thread
static thread_local std::shared_ptr<EventList<Event>> g_event_list;

static std::list<std::shared_ptr<EventList<MemEvent>>> g_all_mem_event_lists;
static thread_local std::shared_ptr<EventList<MemEvent>> g_mem_event_list;
static std::mutex g_all_mem_event_lists_mutex;
static thread_local int32_t g_mem_thread_id;
static uint32_t g_mem_next_thread_id = 0;

inline uint64_t GetTimeInNsec() {
  using clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                 std::chrono::high_resolution_clock,
                                 std::chrono::steady_clock>::type;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

Event::Event(EventType type, std::string name, uint32_t thread_id)
    : type_(type), name_(name), thread_id_(thread_id) {
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

inline EventList<MemEvent> &GetMemEventList() {
  if (!g_mem_event_list) {
    g_mem_event_list = std::make_shared<EventList<MemEvent>>();
    std::lock_guard<std::mutex> guard(g_all_mem_event_lists_mutex);
    g_mem_thread_id = g_mem_next_thread_id++;
    g_all_mem_event_lists.emplace_front(g_mem_event_list);
  }
  return *g_mem_event_list;
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

inline EventList<Event> &GetEventList() {
  if (!g_event_list) {
    std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
    g_event_list = std::make_shared<EventList<Event>>();
    g_thread_id = g_next_thread_id++;
    g_all_event_lists.emplace_front(g_event_list);
    RecoreCurThreadId(g_thread_id);
  }
  return *g_event_list;
}

void Mark(const std::string &name) {
  GetEventList().Record(EventType::kMark, name, g_thread_id);
}

Event *PushEvent(const std::string &name) {
  return GetEventList().Record(EventType::kPushRange, name, g_thread_id);
}

void PopEvent(const std::string &name) {
  GetEventList().Record(EventType::kPopRange, name, g_thread_id);
}

RecordEvent::RecordEvent(const std::string &name)
    : is_enabled_(false), start_ns_(PosixInNsec()) {
  if (g_state == ProfilerState::kDisabled) return;
  // lock is not needed, the code below is thread-safe

  is_enabled_ = true;
  name_ = name;
  Event *e = PushEvent(name_);
  // Maybe need the same push/pop behavior.
  SetCurAnnotation(e);
}

RecordEvent::~RecordEvent() {
  if (g_state == ProfilerState::kDisabled || !is_enabled_) return;
  // lock is not needed, the code below is thread-safe
  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer) {
    tracer->AddCPURecords(CurAnnotationName(), start_ns_, PosixInNsec(),
                          BlockDepth(), g_thread_id);
  }
  ClearCurAnnotation();
  PopEvent(name_);
}

MemEvenRecorder MemEvenRecorder::recorder;

void MemEvenRecorder::PushMemRecord(const void *ptr, const Place &place,
                                    size_t size) {
  if (g_state == ProfilerState::kDisabled) return;
  std::lock_guard<std::mutex> guard(mtx_);
  auto &events = address_memevent_[place];
  PADDLE_ENFORCE(events.count(ptr) == 0, "");
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

RecordRPCEvent::RecordRPCEvent(const std::string &name) {
  if (FLAGS_enable_rpc_profiler) {
    event_.reset(new platform::RecordEvent(name));
  }
}

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

void SynchronizeAllDevice() {
#ifdef PADDLE_WITH_CUDA
  int count = GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    SetDeviceId(i);
    PADDLE_ENFORCE(cudaDeviceSynchronize());
  }
#endif
}

void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE(state != ProfilerState::kDisabled,
                 "Can't enable profiling, since the input state is ",
                 "ProfilerState::kDisabled");
  SynchronizeAllDevice();
  std::lock_guard<std::mutex> l(profiler_mu);
  if (state == g_state) {
    return;
  }
  g_state = state;
  should_send_profile_state = true;
  GetDeviceTracer()->Enable();
#ifdef PADDLE_WITH_CUDA
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

std::vector<std::vector<Event>> GetAllEvents() {
  std::lock_guard<std::mutex> guard(g_all_event_lists_mutex);
  std::vector<std::vector<Event>> result;
  for (auto it = g_all_event_lists.begin(); it != g_all_event_lists.end();
       ++it) {
    result.emplace_back((*it)->Reduce());
  }
  return result;
}

std::vector<std::vector<MemEvent>> GetMemEvents() {
  std::lock_guard<std::mutex> guard(g_all_mem_event_lists_mutex);
  std::vector<std::vector<MemEvent>> result;
  for (auto &it : g_all_mem_event_lists) {
    result.emplace_back((*it).Reduce());
  }
  return result;
}

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
};

// Print results
void PrintProfiler(const std::vector<std::vector<EventItem>> &events_table,
                   const std::string &sorted_domain, const size_t name_width,
                   const size_t data_width, bool merge_thread) {
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
    PADDLE_THROW("Invalid profiler state", g_state);
  }

  if (merge_thread) {
    std::cout << "Note! This Report merge all thread info into one."
              << std::endl;
  }
  std::cout << "Place: " << place << std::endl;
  std::cout << "Time unit: ms" << std::endl;
  std::cout << "Sorted by " << sorted_domain
            << " in descending order in the same thread\n\n";
  // Output events table
  std::cout.setf(std::ios::left);
  std::cout << std::setw(name_width) << "Event" << std::setw(data_width)
            << "Calls" << std::setw(data_width) << "Total";
  if (g_state == ProfilerState::kAll) {
    std::cout << std::setw(data_width * 2) << "CPU Time (Ratio)"
              << std::setw(data_width * 2) << "GPU Time (Ratio)";
  }
  std::cout << std::setw(data_width) << "Min." << std::setw(data_width)
            << "Max." << std::setw(data_width) << "Ave."
            << std::setw(data_width) << "Ratio." << std::endl;
  for (size_t i = 0; i < events_table.size(); ++i) {
    for (size_t j = 0; j < events_table[i].size(); ++j) {
      const EventItem &event_item = events_table[i][j];
      std::cout << std::setw(name_width) << event_item.name
                << std::setw(data_width) << event_item.calls
                << std::setw(data_width) << event_item.total_time;
      if (g_state == ProfilerState::kAll) {
        std::cout << std::setw(data_width * 2)
                  << string::Sprintf(
                         "%f (%f)", event_item.cpu_time,
                         (event_item.cpu_time / event_item.total_time))
                  << std::setw(data_width * 2)
                  << string::Sprintf(
                         "%f (%f)", event_item.gpu_time,
                         (event_item.gpu_time / event_item.total_time));
      }
      std::cout << std::setw(data_width) << event_item.min_time
                << std::setw(data_width) << event_item.max_time
                << std::setw(data_width) << event_item.ave_time
                << std::setw(data_width) << event_item.ratio << std::endl;
    }
  }
  std::cout << std::endl;
}

// Parse the event list and output the profiling report
void ParseEvents(const std::vector<std::vector<Event>> &events,
                 bool merge_thread,
                 EventSortingKey sorted_by = EventSortingKey::kDefault) {
  if (g_state == ProfilerState::kDisabled) return;
  if (merge_thread && events.size() < 2) return;

  std::string sorted_domain;
  std::function<bool(const EventItem &, const EventItem &)> sorted_func;
  switch (sorted_by) {
    case EventSortingKey::kCalls:
      sorted_domain = "number of calls";
      sorted_func = [](const EventItem &a, const EventItem &b) {
        return a.calls > b.calls;
      };
      break;
    case EventSortingKey::kTotal:
      sorted_domain = "total time";
      sorted_func = [](const EventItem &a, const EventItem &b) {
        return a.total_time > b.total_time;
      };
      break;
    case EventSortingKey::kMin:
      sorted_domain = "minimum time";
      sorted_func = [](const EventItem &a, const EventItem &b) {
        return a.min_time > b.min_time;
      };
      break;
    case EventSortingKey::kMax:
      sorted_domain = "maximum time";
      sorted_func = [](const EventItem &a, const EventItem &b) {
        return a.max_time > b.max_time;
      };
      break;
    case EventSortingKey::kAve:
      sorted_domain = "average time";
      sorted_func = [](const EventItem &a, const EventItem &b) {
        return a.ave_time > b.ave_time;
      };
      break;
    case EventSortingKey::kGPUTime:
      sorted_domain = "average time";
      sorted_func = [](const EventItem &a, const EventItem &b) {
        return a.gpu_time > b.gpu_time;
      };
      break;
    case EventSortingKey::kCPUTime:
      sorted_domain = "average time";
      sorted_func = [](const EventItem &a, const EventItem &b) {
        return a.cpu_time > b.cpu_time;
      };
      break;
    default:
      sorted_domain = "event first end time";
  }

  const std::vector<std::vector<Event>> *analyze_events;
  std::vector<std::vector<Event>> merged_events_list;
  if (merge_thread) {
    std::vector<Event> merged_events;
    for (size_t i = 0; i < events.size(); ++i) {
      for (size_t j = 0; j < events[i].size(); ++j) {
        merged_events.push_back(events[i][j]);
      }
    }
    merged_events_list.push_back(merged_events);
    analyze_events = &merged_events_list;
  } else {
    analyze_events = &events;
  }

  std::vector<std::vector<EventItem>> events_table;
  size_t max_name_width = 0;
  for (size_t i = 0; i < (*analyze_events).size(); i++) {
    double total = 0.;  // the total time in one thread
    std::list<Event> pushed_events;
    std::vector<EventItem> event_items;
    std::unordered_map<std::string, int> event_idx;

    for (size_t j = 0; j < (*analyze_events)[i].size(); j++) {
      if ((*analyze_events)[i][j].type() == EventType::kPushRange) {
        pushed_events.push_back((*analyze_events)[i][j]);
      } else if ((*analyze_events)[i][j].type() == EventType::kPopRange) {
        std::list<Event>::reverse_iterator rit = pushed_events.rbegin();
        while (rit != pushed_events.rend() &&
               rit->name() != (*analyze_events)[i][j].name()) {
          ++rit;
        }

        if (rit != pushed_events.rend()) {
          double event_time = 0;
          double gpu_time = rit->CudaElapsedMs((*analyze_events)[i][j]);
          double cpu_time = rit->CpuElapsedMs((*analyze_events)[i][j]);
          if (g_state == ProfilerState::kCUDA) {
            event_time = gpu_time;
          } else if (g_state == ProfilerState::kCPU) {
            event_time = cpu_time;
          } else {
            event_time = gpu_time + cpu_time;
          }

          total += event_time;

          std::string event_name;
          if (merge_thread) {
            event_name = rit->name();
            max_name_width = std::max(max_name_width, event_name.size());
          } else {
            event_name = "thread" + std::to_string(rit->thread_id()) + "::" +
                         rit->name();
            max_name_width = std::max(max_name_width, event_name.size());
          }

          if (event_idx.find(event_name) == event_idx.end()) {
            event_idx[event_name] = event_items.size();
            EventItem event_item = {event_name, 1,          event_time,
                                    event_time, event_time, event_time,
                                    gpu_time,   cpu_time,   0.};
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
            event_items[index].gpu_time += gpu_time;
            event_items[index].cpu_time += cpu_time;
          }

          // remove the push marker from the list
          pushed_events.erase((++rit).base());
        } else {
          LOG(WARNING) << "Cannot find the push marker of event \'"
                       << (*analyze_events)[i][j].name()
                       << "\', which will be ignored in profiling report.";
        }
      }
    }
    // average time
    for (auto &item : event_items) {
      item.ave_time = item.total_time / item.calls;
      item.ratio = item.total_time / total;
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
  PrintProfiler(events_table, sorted_domain, max_name_width + 4, 12,
                merge_thread);
}

struct MemoryProfierReport {
  size_t alloc_times{0};
  size_t alloc_size{0};
  size_t free_times{0};
  size_t free_size{0};
};

// Print results
void PrintMemProfiler(
    const std::map<Place, std::unordered_map<std::string, MemoryProfierReport>>
        &annotation_report,
    const size_t name_width, const size_t data_width) {
  // Output header information
  std::cout << "\n------------------------->"
            << "    Memory Profiling Report     "
            << "<-------------------------\n\n";

  // Output events table
  std::cout.setf(std::ios::left);
  std::cout << std::setw(name_width) << "Event" << std::setw(data_width)
            << "Alloc Calls" << std::setw(data_width) << "Size(MB)"
            << std::setw(data_width) << "Free Calls" << std::setw(data_width)
            << "Size(MB)" << std::endl;

  for (auto &tmp : annotation_report) {
    for (auto &e : tmp.second) {
      auto event_name = string::Sprintf("%s:%s", tmp.first, e.first);
      std::cout << std::setw(name_width) << event_name;
      std::cout << std::setw(data_width) << e.second.alloc_times;
      std::cout << std::setw(data_width)
                << e.second.alloc_size / (1024.0 * 1024.0);
      std::cout << std::setw(data_width) << e.second.free_times;
      std::cout << std::setw(data_width)
                << e.second.free_size / (1024.0 * 1024.0) << std::endl;
    }
  }
  std::cout << std::endl;
}

// parse memory events
void ParseMemEvents(const std::vector<std::vector<MemEvent>> &events) {
  if (g_state == ProfilerState::kDisabled) return;
  // place, annotation, alloc times,  alloc size
  std::map<Place, std::unordered_map<std::string, MemoryProfierReport>>
      annotation_report;

  for (auto &tmp : events) {
    for (auto &e : tmp) {
      if (e.type() == EventType::kPushRange) {
        annotation_report[e.place()][e.annotation()].alloc_times += 1;
        annotation_report[e.place()][e.annotation()].alloc_size += e.bytes();
      } else if (e.type() == EventType::kPopRange) {
        annotation_report[e.place()][e.annotation()].free_times += 1;
        annotation_report[e.place()][e.annotation()].free_size += e.bytes();
      }
    }
  }
  PrintMemProfiler(annotation_report, 55, 18);
}

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string &profile_path) {
  SynchronizeAllDevice();
  MemEvenRecorder::Instance().Flush();

  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;
  // Mark the profiling stop.
  Mark("_stop_profiler_");

  DeviceTracer *tracer = GetDeviceTracer();
  if (tracer->IsEnabled()) {
    tracer->Disable();
    tracer->GenProfile(profile_path);
    tracer->GenEventKernelCudaElapsedTime();
  }

  std::vector<std::vector<Event>> all_events = GetAllEvents();
  ParseEvents(all_events, true, sorted_key);
  ParseEvents(all_events, false, sorted_key);
  if (VLOG_IS_ON(5)) {
    std::vector<std::vector<MemEvent>> all_mem_events = GetMemEvents();
    ParseMemEvents(all_mem_events);
  }

  ResetProfiler();
  g_state = ProfilerState::kDisabled;
  should_send_profile_state = true;
}

bool IsProfileEnabled() { return g_state != ProfilerState::kDisabled; }
bool ShouldSendProfileState() { return should_send_profile_state; }

void SetProfileListener() {
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(
      1, std::numeric_limits<int>::max());
  profiler_lister_id = dist6(rng);
}
int64_t ListenerId() { return profiler_lister_id; }

}  // namespace platform
}  // namespace paddle
