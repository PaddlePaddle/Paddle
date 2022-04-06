// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/profiler/executor_statistics.h"
#include <fstream>
#include <functional>
#include <map>
#include <ostream>
#include <set>
#include <unordered_map>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/platform/profiler/utils.h"

namespace paddle {
namespace platform {

/*
uint64_t CalcOverlapGaps(const TraceEventCollector& collector) {
  std::map<uint64_t, int> m;
  for (const auto& evt : collector.HostEvents()) {
    if (evt.type != TracerEventType::Operator) {
      continue;
    }
    ++m[evt.start_ns];
    --m[evt.end_ns];
  }
  uint64_t gaps = 0;
  int cur = 0;
  uint64_t gap_start = 0;
  for (const auto& kv : m) {
    cur += kv.second;
    // VLOG(1) << kv.first << ":" << kv.second << "cnt:" << cur;
  :  if (cur == 0) {
      gap_start = kv.first;
    } else if (cur > 0) {
      if (gap_start > 0) {
        gaps += (kv.first - gap_start);
        // VLOG(1) << "gap:" << gap_start << "->" << kv.first << " add " <<
        // kv.first - gap_start;
        gap_start = 0;
      }
    } else {
      VLOG(1) << "cnt < 0, fatal error";
    }
  }
  return gaps;
}

uint64_t CalcOverlapTotal(const TraceEventCollector& collector,
                          std::function<bool(const HostTraceEvent&)> filter) {
  std::map<uint64_t, int> m;
  for (const auto& evt : collector.HostEvents()) {
    if (filter(evt) == false) {
      continue;
    }
    ++m[evt.start_ns];
    --m[evt.end_ns];
  }
  uint64_t total = 0;
  int cur = 0;
  uint64_t overlap_start = 0;
  for (const auto& kv : m) {
    cur += kv.second;
    if (cur > 0) {
      if (overlap_start == 0) {
        overlap_start = kv.first;
      }
    } else if (cur == 0) {
      total += (kv.first - overlap_start);
      // VLOG(1) << "slice:" << overlap_start << "->" << kv.first << " add "
      //         << kv.first - overlap_start;
      overlap_start = 0;
    } else {
      VLOG(1) << "cnt < 0, fatal error";
    }
  }
  return total;
}
*/

class StatisticsEngine {
 public:
  int Apply(const NodeTrees& tree);

  void Log(const std::string& full_filename);

 private:
  // type
  struct EventStat {
    uint64_t total_time = 0;
    size_t count = 0;
    uint64_t normalization_time = 0;
  };

  struct Priority {
    // use a smaller number to denote higher priority
    int innerthread_priority = 0;
    int interthread_priority = 0;
  };

  struct StdEvent {
    size_t evt_idx;
    uint64_t start_ns;
    uint64_t end_ns;

    StdEvent(size_t idx, uint64_t start, uint64_t end)
        : evt_idx(idx), start_ns(start), end_ns(end) {}
  };

  using Filter = std::function<bool(const HostTraceEventNode&)>;

  int Init(
      const std::map<uint64_t, std::vector<HostTraceEventNode*>>& thread2nodes);

  int Stat(
      const std::map<uint64_t, std::vector<HostTraceEventNode*>>& thread2nodes);

  void InitStdEvents();

  void InitInnerthreadPriorityForStdEvents();

  void InitInterthreadPriorityForStdEvents();

  int InitFiltersForExecutor();

  int InitFiltersForParallelExecutor();

  int InitFiltersForInterpreterCore();

  int RegisterEventFilter(const std::string& std_event, Filter filter) {
    auto iter = name2idx_.find(std_event);
    if (iter == name2idx_.end()) {
      LOG(WARNING) << "Unsupported std_event " << std_event;
      return -1;
    }
    auto idx = iter->second;
    if (filters_[idx]) {
      LOG(WARNING) << "Duplicate registration for std_event(" << std_event
                   << ")";
      return -1;
    }
    filters_[idx] = std::move(filter);
    return 0;
  }

  int MergeInnerthreadEvents(std::vector<std::vector<StdEvent>>* all_evts);

  int MergeInterthreadEvents(std::vector<std::vector<StdEvent>>* all_evts);

  int StatNormalizationTime(const std::vector<std::vector<StdEvent>>& all_evts);

  bool inited_ = false;
  std::vector<std::string> names_;
  std::vector<Filter> filters_;
  std::vector<Priority> priorities_;
  std::vector<EventStat> statistics_;
  std::unordered_map<std::string, size_t> name2idx_;
};

int StatisticsEngine::Apply(const NodeTrees& tree) {
  auto thread2nodes = tree.Traverse(true);
  return Init(thread2nodes) || Stat(thread2nodes);
}

int StatisticsEngine::Init(
    const std::map<uint64_t, std::vector<HostTraceEventNode*>>& thread2nodes) {
  if (inited_) {
    LOG(WARNING) << "Duplicate initialization for StatisticsEngine";
    return -1;
  }
  inited_ = true;
  InitStdEvents();
  InitInnerthreadPriorityForStdEvents();
  InitInterthreadPriorityForStdEvents();
  // determine executor type
  for (const auto& kv : thread2nodes) {
    for (const auto& evt : kv.second) {
      const auto& name = evt->Name();
      if (name.find("Executor::") == 0) {
        VLOG(10) << "type: Executor";
        return InitFiltersForExecutor();
      } else if (name.find("ParallelExecutor::") == 0) {
        VLOG(10) << "type: ParallelExecutor";
        return InitFiltersForParallelExecutor();
      } else if (name.find("StandaloneExecutor::") == 0) {
        VLOG(10) << "type: InterpreterCore";
        return InitFiltersForInterpreterCore();
      }
    }
  }
  LOG(WARNING) << "Unsupported Executor";
  return -1;
}

void StatisticsEngine::InitStdEvents() {
  name2idx_["Total"] = names_.size();
  names_.push_back("Total");
  name2idx_["PythonEnd"] = names_.size();
  names_.push_back("PythonEnd");
  name2idx_["CplusplusEnd"] = names_.size();
  names_.push_back("CplusplusEnd");
  name2idx_["RunOp"] = names_.size();
  names_.push_back("RunOp");
  name2idx_["LuanchKernel"] = names_.size();
  names_.push_back("LuanchKernel");
  name2idx_["OpCompute"] = names_.size();
  names_.push_back("OpCompute");
  name2idx_["OpInfershape"] = names_.size();
  names_.push_back("OpInfershape");
  name2idx_["DataTransform"] = names_.size();
  names_.push_back("DataTransform");
  name2idx_["GarbageCollect"] = names_.size();
  names_.push_back("GarbageCollect");
  name2idx_["CalcNextOp"] = names_.size();
  names_.push_back("CalcNextOp");
  name2idx_["AllocateDeviceMem"] = names_.size();
  names_.push_back("AllocateDeviceMem");
  name2idx_["FreeDeviceMem"] = names_.size();
  names_.push_back("FreeDeviceMem");
  name2idx_["ThreadpoolAddTask"] = names_.size();
  names_.push_back("ThreadpoolAddTask");

  size_t n = names_.size();
  filters_.resize(n);
  priorities_.resize(n);
  statistics_.resize(n);
}

void StatisticsEngine::InitInnerthreadPriorityForStdEvents() {
  int prio = 0;
  priorities_[name2idx_["AllocateDeviceMem"]].innerthread_priority = ++prio;
  priorities_[name2idx_["FreeDeviceMem"]].innerthread_priority = prio;
  priorities_[name2idx_["ThreadpoolAddTask"]].innerthread_priority = prio;

  priorities_[name2idx_["CalcNextOp"]].innerthread_priority = ++prio;
  priorities_[name2idx_["GarbageCollect"]].innerthread_priority = prio;
  priorities_[name2idx_["OpCompute"]].innerthread_priority = prio;
  priorities_[name2idx_["OpInfershape"]].innerthread_priority = prio;
  priorities_[name2idx_["DataTransform"]].innerthread_priority = prio;

  priorities_[name2idx_["RunOp"]].innerthread_priority = ++prio;

  priorities_[name2idx_["CplusplusEnd"]].innerthread_priority = ++prio;

  priorities_[name2idx_["Total"]].innerthread_priority = ++prio;
}

void StatisticsEngine::InitInterthreadPriorityForStdEvents() {
  int prio = 0;
  priorities_[name2idx_["LuanchKernel"]].interthread_priority = ++prio;
  priorities_[name2idx_["AllocateDeviceMem"]].interthread_priority = ++prio;
  priorities_[name2idx_["FreeDeviceMem"]].interthread_priority = ++prio;
  priorities_[name2idx_["ThreadpoolAddTask"]].interthread_priority = ++prio;

  priorities_[name2idx_["CalcNextOp"]].interthread_priority = ++prio;
  priorities_[name2idx_["GarbageCollect"]].interthread_priority = ++prio;
  priorities_[name2idx_["OpInfershape"]].interthread_priority = ++prio;
  priorities_[name2idx_["DataTransform"]].interthread_priority = ++prio;

  priorities_[name2idx_["RunOp"]].interthread_priority = ++prio;
  priorities_[name2idx_["CplusplusEnd"]].interthread_priority = ++prio;
  priorities_[name2idx_["PythonEnd"]].interthread_priority = prio;
}

int StatisticsEngine::InitFiltersForExecutor() {
  return RegisterEventFilter("Total",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name().find("ProfileStep") == 0;
                             }) ||
         RegisterEventFilter("CplusplusEnd",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() ==
                                      "Executor::RunPartialPreparedContext";
                             }) ||
         RegisterEventFilter("RunOp",
                             [](const HostTraceEventNode& evt) {
                               return evt.Type() == TracerEventType::Operator;
                             }) ||
         RegisterEventFilter("OpCompute",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "compute" &&
                                      evt.Type() ==
                                          TracerEventType::OperatorInner;
                             }) ||
         RegisterEventFilter("OpInfershape",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "infer_shape" &&
                                      evt.Type() ==
                                          TracerEventType::OperatorInner;
                             }) ||
         RegisterEventFilter("GarbageCollect",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "CheckGC";
                             }) ||
         RegisterEventFilter("AllocateDeviceMem",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() ==
                                      "AutoGrowthBestFitAllocator::Allocate";
                             }) ||
         RegisterEventFilter("FreeDeviceMem",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() ==
                                      "AutoGrowthBestFitAllocator::Free";
                             }) ||
         RegisterEventFilter(
             "DataTransform", [](const HostTraceEventNode& evt) {
               return evt.Name() == "prepare_data" &&
                      evt.Type() == TracerEventType::OperatorInner;
             });
}

int StatisticsEngine::InitFiltersForParallelExecutor() { return 0; }

int StatisticsEngine::InitFiltersForInterpreterCore() {
  return RegisterEventFilter("Total",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name().find("ProfileStep") == 0;
                             }) ||
         RegisterEventFilter("CplusplusEnd",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "StandaloneExecutor::run";
                             }) ||
         RegisterEventFilter("RunOp",
                             [](const HostTraceEventNode& evt) {
                               return evt.Type() == TracerEventType::Operator;
                             }) ||
         RegisterEventFilter("OpCompute",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "compute" &&
                                      evt.Type() ==
                                          TracerEventType::OperatorInner;
                             }) ||
         RegisterEventFilter("OpInfershape",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "infer_shape" &&
                                      evt.Type() ==
                                          TracerEventType::OperatorInner;
                             }) ||
         RegisterEventFilter("GarbageCollect",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "CheckGC" ||
                                      evt.Name() == "RecordStreamForGC";
                             }) ||
         RegisterEventFilter("AllocateDeviceMem",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() ==
                                      "StreamSafeCUDAAllocator::Allocate";
                             }) ||
         RegisterEventFilter("FreeDeviceMem",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() ==
                                      "StreamSafeCUDAAllocator::Free";
                             }) ||
         RegisterEventFilter("CalcNextOp",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "RunNextInstructions";
                             }) ||
         RegisterEventFilter("ThreadpoolAddTask",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "WorkQueue::AddTask";
                             });
}

int StatisticsEngine::Stat(
    const std::map<uint64_t, std::vector<HostTraceEventNode*>>& thread2nodes) {
  // build StdEvent
  std::vector<std::vector<StdEvent>> all_evts;
  for (const auto& nodes : thread2nodes) {
    if (nodes.second.size() == 0) {
      continue;
    }
    std::vector<StdEvent> thr_evts;
    thr_evts.reserve(nodes.second.size());
    for (const auto evt : nodes.second) {
      for (size_t idx = 0; idx < filters_.size(); ++idx) {
        if (!filters_[idx]) {
          continue;
        }
        if (filters_[idx](*evt)) {
          thr_evts.emplace_back(idx, evt->StartNs(), evt->EndNs());
          VLOG(10) << "name:" << evt->Name() << " type:" << names_[idx];
          break;
        }
      }
    }
    if (thr_evts.size() == 0) {
      continue;
    }
    std::sort(thr_evts.begin(), thr_evts.end(),
              [](const StdEvent& e1, const StdEvent& e2) {
                return e1.start_ns < e2.start_ns;
              });
    all_evts.push_back(std::move(thr_evts));
  }
  if (all_evts.size() == 0) {
    LOG(WARNING) << "No profiler events";
    return -1;
  }

  // statistic total_time/count
  for (const auto& thr_evts : all_evts) {
    for (const auto& evt : thr_evts) {
      auto& evt_stat = statistics_[evt.evt_idx];
      evt_stat.total_time += evt.end_ns - evt.start_ns;
      evt_stat.count += 1;
    }
  }
  auto& python_end = statistics_[name2idx_["PythonEnd"]];
  const auto& totol = statistics_[name2idx_["Total"]];
  const auto& cplusplus_end = statistics_[name2idx_["CplusplusEnd"]];
  python_end.total_time = totol.total_time - cplusplus_end.total_time;
  python_end.count = cplusplus_end.total_time + 1;

  auto& luanch_kernel = statistics_[name2idx_["LuanchKernel"]];
  const auto& op_compute = statistics_[name2idx_["OpCompute"]];
  const auto& allocate = statistics_[name2idx_["AllocateDeviceMem"]];
  luanch_kernel.total_time = op_compute.total_time - allocate.total_time;
  luanch_kernel.count = op_compute.count;

  // statistic normalization_time
  return MergeInnerthreadEvents(&all_evts) ||
         MergeInterthreadEvents(&all_evts) || StatNormalizationTime(all_evts);
}

int StatisticsEngine::MergeInnerthreadEvents(
    std::vector<std::vector<StdEvent>>* all_evts) {
  for (auto& thr_evts : *all_evts) {
    std::list<StdEvent> merge_evts;
    merge_evts.push_back(thr_evts[0]);
    auto m_iter = merge_evts.begin();
    for (size_t c = 1; c < thr_evts.size();) {
      const auto& cur = thr_evts[c];
      VLOG(10) << "cur:" << names_[cur.evt_idx] << "|" << cur.start_ns << "|"
               << cur.end_ns;  //
      if (m_iter == merge_evts.end()) {
        merge_evts.push_back(cur);
        ++c;
        continue;
      }
      const auto& merge = *m_iter;
      VLOG(10) << "merg:" << names_[merge.evt_idx] << "|" << merge.start_ns
               << "|" << merge.end_ns;  //
      if (cur.start_ns >= merge.end_ns) {
        ++m_iter;
        continue;
      }
      if (cur.end_ns > merge.end_ns) {
        LOG(WARNING) << "Event " << names_[cur.evt_idx]
                     << " starts and ends after Event "
                     << names_[merge.evt_idx];
        return -1;
      }
      auto cur_prio = priorities_[cur.evt_idx].innerthread_priority;
      auto merge_prio = priorities_[merge.evt_idx].innerthread_priority;
      VLOG(10) << cur_prio << " vs " << merge_prio;  //
      if (cur_prio > merge_prio) {
        ++c;
        continue;
      } else if (cur_prio == merge_prio) {
        LOG(WARNING) << "Sub event has lower priority";
        return -1;
      }

      StdEvent prev{merge.evt_idx, merge.start_ns, cur.start_ns};
      StdEvent post{merge.evt_idx, cur.end_ns, merge.end_ns};
      merge_evts.insert(m_iter, prev);
      *(m_iter) = cur;
      if (post.start_ns < post.end_ns) {
        auto pos = m_iter;
        merge_evts.insert(++pos, post);
      }
      ++c;
    }
    for (auto& evt : merge_evts) {
      if (names_[evt.evt_idx] == "Total") {
        evt.evt_idx = name2idx_["PythonEnd"];
      } else if (names_[evt.evt_idx] == "OpCompute") {
        evt.evt_idx = name2idx_["LuanchKernel"];
      }
    }

    VLOG(10) << "new thread";
    for (const auto& evt : merge_evts) {
      VLOG(10) << names_[evt.evt_idx] << " " << evt.start_ns << " "
               << evt.end_ns;
    }  ////

    thr_evts.assign(merge_evts.begin(), merge_evts.end());
  }
  return 0;
}

int StatisticsEngine::MergeInterthreadEvents(
    std::vector<std::vector<StdEvent>>* all_evts) {
  std::vector<StdEvent> base_list;
  base_list.swap((*all_evts)[0]);
  for (size_t i = 1; i < all_evts->size(); ++i) {
    std::vector<StdEvent> merge;
    auto& cur_list = (*all_evts)[i];
    for (size_t c = 0, m = 0; c < cur_list.size(); ++c) {
      VLOG(10) << i << c << m;
      continue;
    }
  }
  all_evts->resize(1);
  (*all_evts)[0].swap(base_list);
  VLOG(10) << (*all_evts)[0].size();  /////
  return 0;
}

int StatisticsEngine::StatNormalizationTime(
    const std::vector<std::vector<StdEvent>>& all_evts) {
  if (all_evts.size() != 1) {
    LOG(WARNING) << "Invalid argument";
    return -1;
  }
  for (const auto& evt : all_evts[0]) {
    statistics_[evt.evt_idx].normalization_time += evt.end_ns - evt.start_ns;
  }
  return 0;
}

void StatisticsEngine::Log(const std::string& full_filename) {
  std::ofstream ofs;
  ofs.open(full_filename, std::ofstream::out | std::ofstream::trunc);
  if (!ofs) {
    LOG(WARNING) << "Unable to open file " << full_filename
                 << " for writing data.";
    return;
  }
  LOG(INFO) << "writing statistics data to " << full_filename;
  ofs << "[";
  for (size_t idx = 0; idx < statistics_.size(); ++idx) {
    const auto& evt_stat = statistics_[idx];
    ofs << string_format(std::string(R"JSON(
  { 
    "statistical item" : "%s", 
    "total time(ns)" : %llu, 
    "total number of times" : %llu,
    "normalization time(ns)" : %llu
  },)JSON"),
                         names_[idx].c_str(), evt_stat.total_time,
                         evt_stat.count, evt_stat.normalization_time);
  }
  ofs.seekp(-1, std::ios_base::end);
  ofs << "]";
  ofs.close();
}

void ExecutorStatistics(const std::string& file_name, const NodeTrees& tree) {
  StatisticsEngine engine;
  if (engine.Apply(tree) == 0) {
    engine.Log(file_name);
  }
  /*REG_EVENT(AutoGrowthBestFitAllocator::Allocate);
  REG_EVENT(AutoGrowthBestFitAllocator::Free);
  REG_EVENT(StreamSafeCUDAAllocator::Allocate);
  REG_EVENT(StreamSafeCUDAAllocator::Free);
  REG_EVENT(WorkQueue::AddTask);
  REG_EVENT(StandaloneExecutor::run);
  REG_EVENT(ParallelExecutor::Run);*/
  /*
  } else if (executor_type == 2) {
    total_cost = GET_EVENT_TATALTIME(ProfileStep);
    cplusplus_cost = GET_EVENT_TATALTIME(ParallelExecutor::Run);
    python_cost = total_cost - cplusplus_cost;
    run_op_cost = CalcOverlapTotal(collector, [](const HostTraceEvent& evt) {
      return evt.type == TracerEventType::Operator;
    });
    infershape_cost = CalcOverlapTotal(collector, [](const HostTraceEvent&
evt)
{ return evt.name == "infer_shape"; });
    allocator_cost = CalcOverlapTotal(collector, [](const HostTraceEvent& evt)
{
return evt.name == "AutoGrowthBestFitAllocator::Allocat"; });
  } else if (executor_type == 3) {
    total_cost = GET_EVENT_TATALTIME(ProfileStep);
    cplusplus_cost = GET_EVENT_TATALTIME(StandaloneExecutor::run);
    python_cost = total_cost - cplusplus_cost;
    run_op_cost = CalcOverlapTotal(collector, [](const HostTraceEvent& evt) {
      return evt.type == TracerEventType::Operator;
    });
    infershape_cost = CalcOverlapTotal(collector, [](const HostTraceEvent&
evt)
{ return evt.name == "infer_shape"; });
    allocator_alloc = CalcOverlapTotal(collector, [](const HostTraceEvent&
evt)
{ return evt.name == "StreamSafeCUDAAllocator::Allocate"; });

  }
uint64_t allocator_cost = 0;
if (executor_type <= 2) {
  VLOG(1) << "thread model==========";
  VLOG(1) << "threadpool AddTask: " << GET_EVENT_TATALTIME(WorkQueue::AddTask)
          << "ns " << GET_EVENT_COUNT(WorkQueue::AddTask) << "times";
  allocator_cost = GET_EVENT_TATALTIME(StreamSafeCUDAAllocator::Allocate) +
                   GET_EVENT_TATALTIME(StreamSafeCUDAAllocator::Free);
  VLOG(1) << "Allocator: " << allocator_cost << "ns "
          << GET_EVENT_COUNT(StreamSafeCUDAAllocator::Allocate) +
                 GET_EVENT_COUNT(StreamSafeCUDAAllocator::Free)
          << "times";
  uint64_t autogrowth_allocator_cost =
      GET_EVENT_TATALTIME(AutoGrowthBestFitAllocator::Allocate) +
      GET_EVENT_TATALTIME(AutoGrowthBestFitAllocator::Free);
  VLOG(1) << "AutoGrowth Allocator: " << autogrowth_allocator_cost << "ns "
          << GET_EVENT_COUNT(AutoGrowthBestFitAllocator::Allocate) +
                 GET_EVENT_COUNT(AutoGrowthBestFitAllocator::Free)
          << "times";
  uint64_t kernel_luanch = CalcOverlapTotal(collector,
                                            [](const HostTraceEvent& evt) {
                                              return evt.name == "compute";
                                            }) -
                           allocator_cost;
  VLOG(1) << "kernel luanch: " << kernel_luanch << "ns ";
} else {  // old executor
  VLOG(1) << "static kernel=========";
  VLOG(1) << "data transform: " << GET_EVENT_TATALTIME(prepare_data) << "ns "
          << GET_EVENT_COUNT(prepare_data) << "times";
  VLOG(1) << "thread model==========";
  VLOG(1) << "threadpool AddTask: " << GET_EVENT_TATALTIME(WorkQueue::AddTask)
          << "ns " << GET_EVENT_COUNT(WorkQueue::AddTask) << "times";
  allocator_cost = GET_EVENT_TATALTIME(AutoGrowthBestFitAllocator::Allocate) +
                   GET_EVENT_TATALTIME(AutoGrowthBestFitAllocator::Free);
  VLOG(1) << "Allocator: " << allocator_cost << "ns "
          << GET_EVENT_COUNT(AutoGrowthBestFitAllocator::Allocate) +
                 GET_EVENT_COUNT(AutoGrowthBestFitAllocator::Free)
          << "times";
}
// common
VLOG(1) << "kernel luanch(overlap): " << kernel_luanch << "ns ";
VLOG(1) << "op gaps: " << CalcOverlapGaps(collector);*/
}

}  // namespace platform
}  // namespace paddle
