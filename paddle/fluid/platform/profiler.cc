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

#include <float.h>
#include <gflags/gflags.h>
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
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/printf.h"

DEFINE_bool(enable_rpc_profiler, false, "Enable rpc profiler or not.");

namespace paddle {
namespace platform {

struct EventList;
struct MemEventList;

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
static std::list<std::shared_ptr<EventList>> g_all_event_lists;
// The thread local event list only can be accessed by the specific thread
static thread_local std::shared_ptr<EventList> g_event_list;

std::mutex profiler_mem;  // record memory mutex
static std::list<std::shared_ptr<MemEventList>> g_all_mem_event_lists;
static thread_local std::shared_ptr<MemEventList> g_mem_event_list;
static std::mutex g_all_mem_event_lists_mutex;
static thread_local int32_t g_mem_thread_id;
static uint32_t g_mem_next_thread_id = 0;

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

struct MemEventList {
  constexpr static size_t kMB = 1024 * 1024;
  constexpr static size_t kEventBlockSize = 16 * kMB;
  constexpr static size_t kEventSize = sizeof(MemEvent);
  constexpr static size_t kEventAlign = alignof(MemEvent);
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

  std::vector<MemEvent> Reduce() {
    std::vector<MemEvent> result;
    for (auto& block : event_blocks) {
      result.insert(result.begin(), std::make_move_iterator(block.begin()),
                    std::make_move_iterator(block.end()));
    }
    event_blocks.clear();
    return result;
  }

  void Clear() { event_blocks.clear(); }
  std::forward_list<std::vector<MemEvent>> event_blocks;
};

inline uint64_t GetTimeInNsec() {
  using clock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                 std::chrono::high_resolution_clock,
                                 std::chrono::steady_clock>::type;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

Event::Event(EventType type, std::string name, uint32_t thread_id,
             const DeviceContext* dev_ctx)
    : type_(type), name_(name), thread_id_(thread_id), has_cuda_(false) {
#ifdef PADDLE_WITH_CUDA
  has_cuda_ = dev_ctx ? platform::is_gpu_place(dev_ctx->GetPlace()) : false;
  if (has_cuda_) {
    auto* cuda_dev_ctx = static_cast<const CUDADeviceContext*>(dev_ctx);
    PADDLE_ENFORCE(cudaSetDevice(
        boost::get<platform::CUDAPlace>(cuda_dev_ctx->GetPlace()).device));
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
  if (!has_cuda_) return 0.0;
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

MemEvent::MemEvent(EventType type, uint64_t start_ns, uint64_t end_ns,
                   size_t bytes, Place place, int64_t thread_id)
    : type_(type),
      start_ns_(start_ns),
      end_ns_(end_ns),
      bytes_(bytes),
      place_(place),
      thread_id_(thread_id) {}

inline MemEventList& GetMemEventList() {
  if (!g_mem_event_list) {
    std::lock_guard<std::mutex> guard(g_all_mem_event_lists_mutex);
    g_mem_event_list = std::make_shared<MemEventList>();
    g_mem_thread_id = g_mem_next_thread_id++;
    g_all_mem_event_lists.emplace_front(g_mem_event_list);
  }
  return *g_mem_event_list;
}

void PushMemEvent(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                  Place place) {
  GetMemEventList().Record(EventType::kPushRange, start_ns, end_ns, bytes,
                           place, g_mem_thread_id);
}

void PopMemEvent(uint64_t start_ns, uint64_t end_ns, size_t bytes,
                 Place place) {
  GetMemEventList().Record(EventType::kPopRange, start_ns, end_ns, bytes, place,
                           g_mem_thread_id);
}

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
    : is_enabled_(false), start_ns_(PosixInNsec()) {
  if (g_state == ProfilerState::kDisabled) return;
  std::lock_guard<std::mutex> l(profiler_mu);

  is_enabled_ = true;
  dev_ctx_ = dev_ctx;
  name_ = name;
  PushEvent(name_, dev_ctx_);
  // Maybe need the same push/pop behavior.
  SetCurAnnotation(name_);
}

RecordEvent::~RecordEvent() {
  if (g_state == ProfilerState::kDisabled || !is_enabled_) return;
  std::lock_guard<std::mutex> l(profiler_mu);
  DeviceTracer* tracer = GetDeviceTracer();
  if (tracer) {
    tracer->AddCPURecords(CurAnnotation(), start_ns_, PosixInNsec(),
                          BlockDepth(), g_thread_id);
  }
  ClearCurAnnotation();
  PopEvent(name_, dev_ctx_);
}

RecordMemEvent::RecordMemEvent()
    : is_enabled_(false), start_ns_(PosixInNsec()) {}

void RecordMemEvent::InitRecordMem(size_t bytes, Place place) {
  if (g_state == ProfilerState::kDisabled) return;
  std::lock_guard<std::mutex> l(profiler_mem);
  is_enabled_ = true;
  place_ = place;
  bytes_ = bytes;
}

void RecordMemEvent::DelRecordMem() {
  if (g_state == ProfilerState::kDisabled || !is_enabled_) return;
  std::lock_guard<std::mutex> l(profiler_mem);
  DeviceTracer* tracer = GetDeviceTracer();
  end_ns_ = PosixInNsec();
  PushMemEvent(start_ns_, end_ns_, bytes_, place_);
  if (tracer) {
    tracer->AddMemInfoRecord(start_ns_, end_ns_, bytes_, place_,
                             g_mem_thread_id);
  }
  PopMemEvent(start_ns_, end_ns_, bytes_, place_);
}

RecordRPCEvent::RecordRPCEvent(const std::string& name,
                               const DeviceContext* dev_ctx) {
  if (FLAGS_enable_rpc_profiler) {
    event_.reset(new platform::RecordEvent(name, dev_ctx));
  }
}

RecordBlock::RecordBlock(int block_id)
    : is_enabled_(false), start_ns_(PosixInNsec()) {
  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;
  is_enabled_ = true;
  SetCurBlock(block_id);
  name_ = string::Sprintf("block_%d", block_id);
}

RecordBlock::~RecordBlock() {
  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled || !is_enabled_) return;
  DeviceTracer* tracer = GetDeviceTracer();
  if (tracer) {
    // We try to put all blocks at the same nested depth in the
    // same timeline lane. and distinguish the using thread_id.
    tracer->AddCPURecords(name_, start_ns_, PosixInNsec(), BlockDepth(),
                          g_thread_id);
  }
  ClearCurBlock();
}

void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE(state != ProfilerState::kDisabled,
                 "Can't enable profiling, since the input state is ",
                 "ProfilerState::kDisabled");

  std::lock_guard<std::mutex> l(profiler_mu);
  if (state == g_state) {
    return;
  }
  g_state = state;
  should_send_profile_state = true;
  GetDeviceTracer()->Enable();
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
  for (auto& it : g_all_mem_event_lists) {
    result.emplace_back((*it).Reduce());
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
  float ratio;
};

struct MemEventItem {
  double peak;
  double valley;
  double average;
  double max;
  double min;
  Place place;
};

// Print results
void PrintProfiler(const std::vector<std::vector<EventItem>>& events_table,
                   const std::string& sorted_domain, const size_t name_width,
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
            << "Calls" << std::setw(data_width) << "Total"
            << std::setw(data_width) << "Min." << std::setw(data_width)
            << "Max." << std::setw(data_width) << "Ave."
            << std::setw(data_width) << "Ratio." << std::endl;
  for (size_t i = 0; i < events_table.size(); ++i) {
    for (size_t j = 0; j < events_table[i].size(); ++j) {
      const EventItem& event_item = events_table[i][j];
      std::cout << std::setw(name_width) << event_item.name
                << std::setw(data_width) << event_item.calls
                << std::setw(data_width) << event_item.total_time
                << std::setw(data_width) << event_item.min_time
                << std::setw(data_width) << event_item.max_time
                << std::setw(data_width) << event_item.ave_time
                << std::setw(data_width) << event_item.ratio << std::endl;
    }
  }
  std::cout << std::endl;
}
// print the memory information
void PrintMemProfiler(const std::vector<MemEventItem>& events_table,
                      const size_t name_width, const size_t data_width) {
  VLOG(1) << "\n";
  bool cpu_involved = false;
  bool gpu_involved = false;
  bool cudapin_involved = false;
  for (size_t i = 0; i < events_table.size(); ++i) {
    VLOG(1) << "Place: ";
    Place a;
    if (is_cpu_place(events_table[i].place)) {
      VLOG(1) << "CPU";
      cpu_involved = true;
    } else if (is_gpu_place(events_table[i].place)) {
      VLOG(1) << "GPU:"
              << boost::get<platform::CUDAPlace>(events_table[i].place)
                     .GetDeviceId()
              << "\n";
      gpu_involved = true;
    } else {
      VLOG(1) << "CUDAPinnedPlace";
      cudapin_involved = true;
    }
    VLOG(1) << "Memory unit: MB\n";
    // Output events table
    VLOG(1) << "Event"
            << "\tPeak"
            << "\tValley"
            << "\tAverage"
            << "\tMaximum"
            << "\tMinimum"
            << "\n";
    VLOG(1) << "Memory"
            << "\t" << events_table[i].peak << "\t" << events_table[i].valley
            << "\t" << events_table[i].average << "\t" << events_table[i].max
            << "\t" << events_table[i].min << "\n";
  }
  VLOG(1) << "\n";
  if (!cpu_involved) {
    VLOG(1) << "Place: CPU\n";
    VLOG(1) << "Memory unit: MB\n";
    VLOG(1) << "Event"
            << "\tPeak"
            << "\tValley"
            << "\tAverage"
            << "\tMaximum"
            << "\tMinimum"
            << "\n";
    VLOG(1) << "Memory"
            << "\t0\t0\t0\t0\t0\n";
    VLOG(1) << "\n";
  }
  if (!gpu_involved) {
    VLOG(1) << "Place: GPU\n";
    VLOG(1) << "Memory unit: MB\n";
    VLOG(1) << "Event"
            << "\tPeak"
            << "\tValley"
            << "\tAverage"
            << "\tMaximum"
            << "\tMinimum"
            << "\n";
    VLOG(1) << "Memory"
            << "\t0\t0\t0\t0\t0\n";
    VLOG(1) << "\n";
  }
  if (!cudapin_involved) {
    VLOG(1) << "Place: CUDAPinnedPlace\n";
    VLOG(1) << "Memory unit: MB\n";
    VLOG(1) << "Event"
            << "\tPeak"
            << "\tValley"
            << "\tAverage"
            << "\tMaximum"
            << "\tMinimum"
            << "\n";
    VLOG(1) << "Memory"
            << "\t0\t0\t0\t0\n";
    VLOG(1) << "\n";
  }
}

struct MemResult {
  uint64_t time;
  double size_;
};

// parse memory events
void ParseMemEvents(const std::vector<std::vector<MemEvent>>& events) {
  if (g_state == ProfilerState::kDisabled) return;

  static std::function<bool(const MemResult&, const MemResult&)> sort_func = [](
      const MemResult& a, const MemResult& b) { return a.time < b.time; };
  // combine the start_ns and end_ns and parse record
  std::vector<MemEventItem> event_info(events.size());
  std::vector<std::vector<MemResult>> mem_timeline(events.size());
  int base = 1024 * 1024;
  for (size_t i = 0; i < events.size(); ++i) {
    event_info[i].place = events[i][0].place();
    double total = 0.0;
    event_info[i].peak = 0.0;
    event_info[i].valley = DBL_MAX;
    event_info[i].max = 0.0;
    event_info[i].min = DBL_MAX;

    std::vector<MemResult> tmp_mem_results;
    for (size_t k = 1; k < events[i].size() - 1; ++k) {
      double individual_size = static_cast<double>(events[i][k].bytes()) / base;
      event_info[i].max = std::max(event_info[i].max, individual_size);
      event_info[i].min = std::min(event_info[i].min, individual_size);
      total += individual_size;

      tmp_mem_results.push_back({events[i][k].start_ns(), individual_size});
      tmp_mem_results.push_back({events[i][k].end_ns(), -individual_size});
    }
    sort(tmp_mem_results.begin(), tmp_mem_results.end(), sort_func);
    double crt_size = 0.;
    for (size_t j = 0; j < tmp_mem_results.size(); ++j) {
      crt_size += tmp_mem_results[j].size_;
      while (j < tmp_mem_results.size() - 1 &&
             tmp_mem_results[j].time == tmp_mem_results[j + 1].time)
        crt_size += tmp_mem_results[++j].size_;
      mem_timeline[i].push_back({tmp_mem_results[j].time, crt_size});

      event_info[i].peak = std::max(event_info[i].peak, crt_size);
      event_info[i].valley = std::min(event_info[i].valley, crt_size);
    }
    event_info[i].peak =
        std::max(event_info[i].peak, mem_timeline[i].rbegin()->size_);
    event_info[i].valley =
        std::min(event_info[i].valley, mem_timeline[i].rbegin()->size_);
    event_info[i].average = total / mem_timeline[i].size();
  }

  PrintMemProfiler(event_info, 10, 12);
}

// Parse the event list and output the profiling report
void ParseEvents(const std::vector<std::vector<Event>>& events,
                 bool merge_thread,
                 EventSortingKey sorted_by = EventSortingKey::kDefault) {
  if (g_state == ProfilerState::kDisabled) return;
  if (merge_thread && events.size() < 2) return;

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

  const std::vector<std::vector<Event>>* analyze_events;
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
          double event_time = (g_state == ProfilerState::kCUDA ||
                               g_state == ProfilerState::kAll)
                                  ? rit->CudaElapsedMs((*analyze_events)[i][j])
                                  : rit->CpuElapsedMs((*analyze_events)[i][j]);
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
                                    0.};
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
                       << (*analyze_events)[i][j].name()
                       << "\', which will be ignored in profiling report.";
        }
      }
    }
    // average time
    for (auto& item : event_items) {
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

void DisableProfiler(EventSortingKey sorted_key,
                     const std::string& profile_path) {
  std::lock_guard<std::mutex> l(profiler_mu);
  if (g_state == ProfilerState::kDisabled) return;
  // Mark the profiling stop.
  Mark("_stop_profiler_", nullptr);

  std::vector<std::vector<Event>> all_events = GetAllEvents();
  std::vector<std::vector<MemEvent>> all_mem_events = GetMemEvents();
  ParseEvents(all_events, true, sorted_key);
  ParseEvents(all_events, false, sorted_key);
  ParseMemEvents(all_mem_events);
  ResetProfiler();
  DeviceTracer* tracer = GetDeviceTracer();
  if (tracer->IsEnabled()) {
    tracer->Disable();
    tracer->GenProfile(profile_path);
  }
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
