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
#include <stack>
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

static TracerOption g_tracer_option = TracerOption::kDefault;
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

RecordEvent::RecordEvent(const std::string &name, const RecordRole role)
    : is_enabled_(false), start_ns_(PosixInNsec()), role_(role) {
  if (g_state == ProfilerState::kDisabled || name.empty()) return;
  // lock is not needed, the code below is thread-safe
  is_enabled_ = true;
  Event *e = PushEvent(name);
  // Maybe need the same push/pop behavior.
  SetCurAnnotation(e);
  name_ = e->name();
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

struct OverHead {
  bool print = false;
  double total_time = 0.;
  float compute_ratio = 0.0f;
  float framework_ratio = 0.0f;
  EventItem memcpy_item;
  std::vector<EventItem> sub_memcpy_items;
};

// Print results
void PrintProfiler(const std::vector<std::vector<EventItem>> &events_table,
                   const std::multimap<std::string, EventItem> &child_map,
                   const OverHead &overhead, const std::string &sorted_domain,
                   const size_t name_width, const size_t data_width,
                   bool merge_thread, int print_depth, int remove_len) {
  if (print_depth == 0) {
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
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Except profiler state must to be one of ['CPU', 'GPU' 'ALL'], but "
          "received Invalid profiler state"));
    }

    if (merge_thread) {
      std::cout << "Note! This Report merge all thread info into one."
                << std::endl;
    }
    std::cout << "Place: " << place << std::endl;
    std::cout << "Time unit: ms" << std::endl;
    std::cout << "Sorted by " << sorted_domain
              << " in descending order in the same thread\n\n";

    if (overhead.print) {
      double compute_time = overhead.total_time * overhead.compute_ratio;
      double framework_time = overhead.total_time * overhead.framework_ratio;
      std::cout.setf(std::ios::left);
      std::cout << "Total time: " << overhead.total_time << std::endl;
      std::cout << std::setw(25) << "  Computation time"
                << "Total: " << std::setw(data_width) << compute_time
                << "Ratio: " << overhead.compute_ratio * 100 << "%"
                << std::endl;
      std::cout << std::setw(25) << "  Framework overhead"
                << "Total: " << std::setw(data_width) << framework_time
                << "Ratio: " << overhead.framework_ratio * 100 << "%"
                << std::endl;

      std::cout << "\n-------------------------"
                << "     GpuMemCpy Summary     "
                << "-------------------------\n\n";
      std::cout << std::setw(25) << "GpuMemcpy"
                << "Calls: " << std::setw(data_width)
                << overhead.memcpy_item.calls
                << "Total: " << std::setw(data_width)
                << overhead.memcpy_item.total_time
                << "Ratio: " << overhead.memcpy_item.ratio * 100 << "%"
                << std::endl;
      for (size_t i = 0; i < overhead.sub_memcpy_items.size(); ++i) {
        EventItem item = overhead.sub_memcpy_items[i];
        if (item.calls != 0) {
          std::cout << std::setw(25) << "  " + item.name
                    << "Calls: " << std::setw(data_width) << item.calls
                    << "Total: " << std::setw(data_width) << item.total_time
                    << "Ratio: " << item.ratio * 100 << "%" << std::endl;
        }
      }
    }
    std::cout << "\n-------------------------"
              << "       Event Summary       "
              << "-------------------------\n\n";
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
  }

  if (events_table.size() <= 0) return;

  for (size_t i = 0; i < events_table.size(); ++i) {
    for (size_t j = 0; j < events_table[i].size(); ++j) {
      auto event_item = events_table[i][j];
      std::vector<std::vector<EventItem>> child_table;
      std::vector<EventItem> table;
      for (auto it = child_map.begin(); it != child_map.end(); it++) {
        if (it->first == event_item.name) {
          table.push_back(it->second);
        }
      }
      child_table.push_back(table);

      auto name_len = event_item.name.length();
      std::string print_name = event_item.name.substr(remove_len, name_len);
      std::string delimiter;
      for (int i = 0; i < print_depth; i++) {
        delimiter = "  " + delimiter;
      }
      print_name = delimiter + print_name;

      std::cout << std::setw(name_width) << print_name << std::setw(data_width)
                << event_item.calls << std::setw(data_width)
                << event_item.total_time;
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
      PrintProfiler(child_table, child_map, overhead, sorted_domain, name_width,
                    data_width, merge_thread, print_depth + 1, 0);
    }
  }
}

std::function<bool(const EventItem &, const EventItem &)> SetSortedFunc(
    EventSortingKey sorted_by, std::string *domain) {
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
  *domain = sorted_domain;
  return sorted_func;
}

void SetEvent(bool merge_thread, Event analyze_event, size_t *max_name_width,
              std::list<Event> *pushed_events,
              std::vector<EventItem> *event_items,
              std::unordered_map<std::string, int> *event_idx) {
  if (analyze_event.type() == EventType::kPushRange) {
    pushed_events->push_back(analyze_event);
  } else if (analyze_event.type() == EventType::kPopRange) {
    std::list<Event>::reverse_iterator rit = pushed_events->rbegin();
    while (rit != pushed_events->rend() &&
           rit->name() != analyze_event.name()) {
      ++rit;
    }
    // to find the father name event name

    if (rit != pushed_events->rend()) {
      double event_time = 0;
      double gpu_time = 0.0f;
#ifdef PADDLE_WITH_CUDA
      gpu_time = rit->CudaElapsedMs(analyze_event);
#endif
      double cpu_time = rit->CpuElapsedMs(analyze_event);
      if (g_state == ProfilerState::kCUDA) {
        event_time = gpu_time;
      } else if (g_state == ProfilerState::kCPU) {
        event_time = cpu_time;
      } else {
        event_time = gpu_time + cpu_time;
      }

      std::string event_name;
      if (merge_thread) {
        event_name = rit->name();
        *max_name_width = std::max(*max_name_width, event_name.size());
      } else {
        event_name =
            "thread" + std::to_string(rit->thread_id()) + "::" + rit->name();
        *max_name_width = std::max(*max_name_width, event_name.size());
      }

      if (event_idx->find(event_name) == event_idx->end()) {
        event_idx->insert({event_name, event_items->size()});
        EventItem event_item = {event_name, 1,          event_time,
                                event_time, event_time, event_time,
                                cpu_time,   gpu_time,   0.};
        event_items->push_back(event_item);
      } else {
        int index = event_idx->at(event_name);
        event_items->at(index).calls += 1;
        // total time
        event_items->at(index).total_time += event_time;
        // min time
        event_items->at(index).min_time =
            std::min(event_time, event_items->at(index).min_time);
        // max time
        event_items->at(index).max_time =
            std::max(event_time, event_items->at(index).max_time);
        event_items->at(index).gpu_time += gpu_time;
        event_items->at(index).cpu_time += cpu_time;
      }

      // remove the push marker from the list
      pushed_events->erase((++rit).base());
    } else {
      LOG(WARNING) << "Cannot find the push marker of event \'"
                   << analyze_event.name()
                   << "\', which will be ignored in profiling report.";
    }
  }
}

void ComputeOverhead(const std::multimap<std::string, EventItem> &sub_child_map,
                     OverHead *overhead) {
  EventItem memcpy_async = {"GpuMemcpyAsync", 0, 0., 0., 0., 0., 0., 0., 0.0f};
  EventItem memcpy_sync = {"GpuMemcpySync", 0, 0., 0., 0., 0., 0., 0., 0.0f};
  for (auto it = sub_child_map.begin(); it != sub_child_map.end(); it++) {
    if (it->second.name.find("compute") != std::string::npos) {
      overhead->compute_ratio += it->second.ratio;
    }
    if (it->second.name.find("GpuMemcpyAsync") != std::string::npos) {
      memcpy_async.calls += it->second.calls;
      memcpy_async.total_time += it->second.total_time;
      memcpy_async.ratio += it->second.ratio;
    } else if (it->second.name.find("GpuMemcpySync") != std::string::npos) {
      memcpy_sync.calls += it->second.calls;
      memcpy_sync.total_time += it->second.total_time;
      memcpy_sync.ratio += it->second.ratio;
    }
  }
  overhead->framework_ratio = 1.0f - overhead->compute_ratio;
  overhead->memcpy_item.calls = memcpy_async.calls + memcpy_sync.calls;
  overhead->memcpy_item.total_time =
      memcpy_async.total_time + memcpy_sync.total_time;
  overhead->memcpy_item.ratio = memcpy_async.ratio + memcpy_sync.ratio;
  overhead->sub_memcpy_items = {memcpy_async, memcpy_sync};
}

// When TracerOption is KDefault, OpDetail will be recorded but only default
// profile result will be printed.
// GpuMemcpy should be printed in kDefault setting, however it offten occurs
// during 'compute' or 'prepare data' process, so the elements of sub_child_map
// need to be changed before being inserted into child_map. for instance:
// it->first: OpType/compute => OpType
// it->second.name: OpType/compute/GpuMemcpyAsync => OpType/GpuMemcpyAsync.
void GetChildMap(const std::multimap<std::string, EventItem> &sub_child_map,
                 std::multimap<std::string, EventItem> *child_map) {
  if (platform::GetTracerOption() != TracerOption::kDefault) {
    for (auto it = sub_child_map.begin(); it != sub_child_map.end(); it++) {
      child_map->insert(
          std::pair<std::string, EventItem>(it->first, it->second));
    }
  } else {
    for (auto it = sub_child_map.begin(); it != sub_child_map.end(); it++) {
      if (it->second.name.find("GpuMemcpy") != std::string::npos) {
        std::string parent_name = it->first;
        auto left_pos = it->first.find("/");
        if (left_pos != std::string::npos) {
          parent_name = it->first.substr(0, left_pos);
        }
        auto item = it->second;
        auto right_pos = item.name.rfind("/");
        if (right_pos != std::string::npos) {
          std::string child_name = item.name.substr(
              right_pos + 1, item.name.length() - right_pos - 1);
          item.name = parent_name + "/" + child_name;
        }
        child_map->insert(std::pair<std::string, EventItem>(parent_name, item));
      }
    }
  }
}

// Parse the event list and output the profiling report
void ParseEvents(const std::vector<std::vector<Event>> &events,
                 bool merge_thread,
                 EventSortingKey sorted_by = EventSortingKey::kDefault) {
  if (g_state == ProfilerState::kDisabled) return;
  if (merge_thread && events.size() < 2) return;

  std::string sorted_domain;
  std::function<bool(const EventItem &, const EventItem &)> sorted_func;
  sorted_func = SetSortedFunc(sorted_by, &sorted_domain);

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
  std::multimap<std::string, EventItem> child_map;
  size_t max_name_width = 0;
  OverHead overhead;

  for (size_t i = 0; i < (*analyze_events).size(); i++) {
    double total = 0.;  // the total time in one thread
    std::list<Event> pushed_events;
    std::vector<EventItem> event_items;
    std::vector<EventItem> main_event_items;
    std::unordered_map<std::string, int> event_idx;
    std::multimap<std::string, EventItem> sub_child_map;

    for (size_t j = 0; j < (*analyze_events)[i].size(); j++) {
      Event analyze_event = (*analyze_events)[i][j];
      SetEvent(merge_thread, analyze_event, &max_name_width, &pushed_events,
               &event_items, &event_idx);
    }

    auto table_size = event_items.size();
    std::vector<int> child_index(table_size, 0);
    for (size_t j = 0; j < table_size; ++j) {
      std::string fname = event_items[j].name;
      std::string grad_name = event_items[j].name + "_grad";
      for (size_t k = 0; k < table_size; ++k) {
        std::string cname = event_items[k].name;
        bool condition = cname.length() > fname.length() &&
                         cname.rfind(fname, 0) == 0 &&
                         !cname.rfind(grad_name, 0) == 0 &&
                         (cname[fname.length()] == '/' &&
                          cname.rfind('/') == fname.length());
        if (condition) {
          sub_child_map.insert(
              std::pair<std::string, EventItem>(fname, event_items[k]));
          child_index[k] = 1;
        }
      }
    }

    for (size_t j = 0; j < table_size; ++j) {
      if (child_index[j] == 0) {
        main_event_items.push_back(event_items[j]);
        total += event_items[j].total_time;
      }
    }

    // average time
    for (auto &item : main_event_items) {
      item.ave_time = item.total_time / item.calls;
      item.ratio = item.total_time / total;
    }
    for (auto it = sub_child_map.begin(); it != sub_child_map.end(); it++) {
      it->second.ratio = it->second.total_time / total;
      it->second.ave_time = it->second.total_time / it->second.calls;
    }

    // When multi-threaded, overhead are printed only if merge_thread is true
    if ((*analyze_events).size() == 1) {
      overhead.total_time = total;
      overhead.print = true;
      ComputeOverhead(sub_child_map, &overhead);
    }

    // sort
    if (sorted_by != EventSortingKey::kDefault) {
      std::sort(main_event_items.begin(), main_event_items.end(), sorted_func);
    }

    events_table.push_back(main_event_items);
    // log warning if there are events with `push` but without `pop`
    std::list<Event>::reverse_iterator rit = pushed_events.rbegin();
    while (rit != pushed_events.rend()) {
      LOG(WARNING) << "Cannot find the pop marker of event \'" << rit->name()
                   << "\', which will be ignored in profiling report.";
      ++rit;
    }

    GetChildMap(sub_child_map, &child_map);
  }

  // Print report
  PrintProfiler(events_table, child_map, overhead, sorted_domain,
                max_name_width + 8, 12, merge_thread, 0, 0);
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

void DealWithShowName() {
  std::unordered_map<std::string, std::vector<std::string>> profiler_name_info;
  for (auto it = g_all_event_lists.begin(); it != g_all_event_lists.end();
       ++it) {
    for (auto &block : (*it)->event_blocks) {
      for (auto &r : block) {
        auto event_name = r.name();
        size_t start = event_name.find('%', 0);
        size_t end = event_name.find('%', start + 1);
        std::string prefix_str = event_name.substr(0, start);
        while (start != std::string::npos && end != std::string::npos) {
          auto search_str = event_name.substr(start, end - start + 1);
          std::string replace_str = "";
          int replace_index = 0;

          auto it = profiler_name_info.find(prefix_str);
          if (it == profiler_name_info.end()) {
            std::vector<std::string> op_name_vector{search_str};
            profiler_name_info[prefix_str] = op_name_vector;
          } else {
            auto op_name_vector = it->second;
            auto iter =
                find(op_name_vector.begin(), op_name_vector.end(), search_str);
            if (iter == op_name_vector.end()) {
              replace_index = it->second.size();
              it->second.push_back(search_str);
            } else {
              replace_index = it->second.size() - 1;
            }
          }
          replace_str = std::to_string(replace_index);
          event_name.replace(start, end - start + 1, replace_str);
          start = start + 1;
          start = event_name.find('%', start);
          end = event_name.find('%', start + 1);
          prefix_str = event_name.substr(0, start);
        }
        r.set_name(event_name);
      }
    }
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

std::string OpName(const framework::VariableNameMap &name_map,
                   const std::string &type_name) {
  if (platform::GetTracerOption() != platform::TracerOption::kAllOpDetail)
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
}  // namespace platform
}  // namespace paddle
