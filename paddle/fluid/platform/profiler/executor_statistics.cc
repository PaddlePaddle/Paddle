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

  void MergeEvents(std::function<size_t(size_t, size_t)> merger,
                   std::vector<StdEvent>* in_out_evts);

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

int StatisticsEngine::InitFiltersForParallelExecutor() {
  return RegisterEventFilter("Total",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name().find("ProfileStep") == 0;
                             }) ||
         RegisterEventFilter("CplusplusEnd",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "ParallelExecutor::Run";
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
                               return evt.Name() == "eager_deletion";
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
         RegisterEventFilter("DataTransform",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "prepare_data" &&
                                      evt.Type() ==
                                          TracerEventType::OperatorInner;
                             }) ||
         RegisterEventFilter("ThreadpoolAddTask",
                             [](const HostTraceEventNode& evt) {
                               return evt.Name() == "WorkQueue::AddTask";
                             });
}

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
  python_end.count = cplusplus_end.count + 1;

  auto& luanch_kernel = statistics_[name2idx_["LuanchKernel"]];
  const auto& op_compute = statistics_[name2idx_["OpCompute"]];
  const auto& allocate = statistics_[name2idx_["AllocateDeviceMem"]];
  luanch_kernel.total_time = op_compute.total_time - allocate.total_time;
  luanch_kernel.count = op_compute.count;

  // statistic normalization_time
  return MergeInnerthreadEvents(&all_evts) ||
         MergeInterthreadEvents(&all_evts) || StatNormalizationTime(all_evts);
}

void StatisticsEngine::MergeEvents(std::function<size_t(size_t, size_t)> merger,
                                   std::vector<StdEvent>* in_out_evts) {
  auto evts = *in_out_evts;
  std::sort(evts.begin(), evts.end(),
            [](const StdEvent& e1, const StdEvent& e2) {
              return e1.start_ns < e2.start_ns;
            });

  std::list<StdEvent> merged;
  auto iter = merged.begin();
  for (size_t i = 0; i < evts.size();) {
    if (iter == merged.end()) {
      iter = merged.insert(iter, evts[i]);
      ++i;
    } else if (iter->end_ns <= evts[i].start_ns) {
      ++iter;
    } else if (iter->evt_idx == evts[i].evt_idx) {
      iter->end_ns = std::max(iter->end_ns, evts[i].end_ns);
      ++i;
    } else {
      auto merged_type = merger(iter->evt_idx, evts[i].evt_idx);
      if (merged_type == iter->evt_idx) {
        if (evts[i].end_ns > iter->end_ns) {
          evts[i].start_ns = iter->end_ns;
          ++iter;
        } else {
          ++i;
        }
      } else {
        StdEvent back = *iter;
        if (back.start_ns != evts[i].start_ns) {
          merged.insert(iter, {back.evt_idx, back.start_ns, evts[i].start_ns});
        }
        *iter = evts[i];
        if (back.end_ns > evts[i].end_ns) {
          auto pos = iter;
          merged.insert(++pos, {back.evt_idx, evts[i].end_ns, back.end_ns});
        }
        ++i;
      }
    }
  }
  in_out_evts->assign(merged.begin(), merged.end());
}

int StatisticsEngine::MergeInnerthreadEvents(
    std::vector<std::vector<StdEvent>>* all_evts) {
  auto merger = [& priorities = priorities_](size_t idx1, size_t idx2) {
    return priorities[idx1].innerthread_priority <=
                   priorities[idx2].innerthread_priority
               ? idx1
               : idx2;
  };
  for (auto& thr_evts : *all_evts) {
    MergeEvents(merger, &thr_evts);
    for (auto& evt : thr_evts) {
      if (names_[evt.evt_idx] == "Total") {
        evt.evt_idx = name2idx_["PythonEnd"];
      } else if (names_[evt.evt_idx] == "OpCompute") {
        evt.evt_idx = name2idx_["LuanchKernel"];
      }
    }
  }
  return 0;
}

int StatisticsEngine::MergeInterthreadEvents(
    std::vector<std::vector<StdEvent>>* all_evts) {
  auto merger = [& priorities = priorities_](size_t idx1, size_t idx2) {
    return priorities[idx1].interthread_priority <=
                   priorities[idx2].interthread_priority
               ? idx1
               : idx2;
  };
  // K-way merge, just simplest impl
  std::vector<StdEvent> base_list;
  base_list.swap(all_evts->at(0));
  for (size_t i = 1; i < all_evts->size(); ++i) {
    auto& cur_list = all_evts->at(i);
    base_list.reserve(base_list.size() + cur_list.size());
    base_list.insert(base_list.end(), cur_list.begin(), cur_list.end());
    MergeEvents(merger, &base_list);
  }
  all_evts->resize(1);
  (*all_evts)[0].swap(base_list);
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
  // verify
  uint64_t total = statistics_[name2idx_["Total"]].total_time;
  uint64_t normalization_sum = 0;
  for (size_t idx = 0; idx < statistics_.size(); ++idx) {
    normalization_sum += statistics_[idx].normalization_time;
  }
  if (total - normalization_sum != 0) {
    LOG(WARNING) << "total: " << total
                 << "is greater than normalization_sum:" << normalization_sum;
    return -1;
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
}

}  // namespace platform
}  // namespace paddle
