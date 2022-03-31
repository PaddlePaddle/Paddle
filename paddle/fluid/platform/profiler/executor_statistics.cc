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
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include "glog/logging.h"

namespace paddle {
namespace platform {

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
    if (cur == 0) {
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

void StatisticsHostEvents(const TraceEventCollector& collector) {
  std::unordered_map<std::string, int> name2idx;
  std::vector<size_t> idx2cnt;
  std::vector<uint64_t> idx2total_ns;
  std::vector<std::set<uint64_t>> idx2threads;
#define REG_EVENT(event)             \
  name2idx[#event] = idx2cnt.size(); \
  idx2cnt.push_back(0);              \
  idx2total_ns.push_back(0);         \
  idx2threads.push_back({})
  REG_EVENT(AutoGrowthBestFitAllocator::Allocate);
  REG_EVENT(AutoGrowthBestFitAllocator::Free);
  REG_EVENT(StreamSafeCUDAAllocator::Allocate);
  REG_EVENT(StreamSafeCUDAAllocator::Free);
  REG_EVENT(WorkQueue::AddTask);
  REG_EVENT(prepare_data);
  REG_EVENT(infer_shape);
  REG_EVENT(compute);
  REG_EVENT(ProfileStep);
  REG_EVENT(StandaloneExecutor::run);
  REG_EVENT(Executor::RunPartialPreparedContext);
  REG_EVENT(ParallelExecutor::Run);
#undef REG_EVENT

  // statistics
  for (const auto& evt : collector.HostEvents()) {
    // VLOG(1) << "name: " << evt.name;
    std::string prefix = evt.name;
    size_t split = evt.name.find('#');
    if (split != std::string::npos) {
      prefix = evt.name.substr(0, split);
    }
    auto iter = name2idx.find(prefix);
    if (iter != name2idx.end()) {
      int idx = iter->second;
      ++idx2cnt[idx];
      idx2total_ns[idx] += evt.end_ns - evt.start_ns;
      idx2threads[idx].insert(evt.thread_id);
    }
  }

  // Events Statistics
  VLOG(1) << "=========Events Statistics==========";
  for (const auto& kv : name2idx) {
    VLOG(1) << kv.first << " cnt:" << idx2cnt[kv.second]
            << " total(ns):" << idx2total_ns[kv.second]
            << " threads:" << idx2threads[kv.second].size();
  }

  uint64_t total_cost = 0;
  uint64_t python_cost = 0;
  uint64_t cplusplus_cost = 0;
  uint64_t run_op_cost = 0;
/*uint64_t infershape_cost = 0;
uint64_t luanch_kernel_cost = 0;
uint64_t gc_cost = 0;
uint64_t nexts_cost = 0;
uint64_t allocator_cost = 0;
uint64_t add_task_cost = 0;
uint64_t data_transform = 0;*/

#define GET_EVENT_TATALTIME(event) idx2total_ns[name2idx[#event]]
#define GET_EVENT_COUNT(event) idx2cnt[name2idx[#event]]
  std::vector<std::string> executor_type_names{
      "Unknown", "Executor", "ParallelExecutor", "StandaloneExecutor"};
  int executor_type = 0;
  if (GET_EVENT_COUNT(Executor::RunPartialPreparedContext) > 0) {
    executor_type = 1;  // executor
  } else if (GET_EVENT_COUNT(ParallelExecutor::Run) > 0) {
    executor_type = 2;  // PE
  } else if (GET_EVENT_COUNT(StandaloneExecutor::run) > 0) {
    executor_type = 3;  // new executor
  }

  VLOG(1) << "========Executor analysis(" << executor_type_names[executor_type]
          << ")========";
  if (executor_type == 1) {
    total_cost = GET_EVENT_TATALTIME(ProfileStep);
    cplusplus_cost = GET_EVENT_TATALTIME(Executor::RunPartialPreparedContext);
    python_cost = total_cost - cplusplus_cost;
    run_op_cost = CalcOverlapTotal(collector, [](const HostTraceEvent& evt) {
      return evt.type == TracerEventType::Operator;
    });
  } else if (executor_type == 2) {
    total_cost = GET_EVENT_TATALTIME(ProfileStep);
    cplusplus_cost = GET_EVENT_TATALTIME(ParallelExecutor::Run);
    python_cost = total_cost - cplusplus_cost;
    run_op_cost = CalcOverlapTotal(collector, [](const HostTraceEvent& evt) {
      return evt.type == TracerEventType::Operator;
    });
  } else if (executor_type == 3) {
    total_cost = GET_EVENT_TATALTIME(ProfileStep);
    cplusplus_cost = GET_EVENT_TATALTIME(StandaloneExecutor::run);
    python_cost = total_cost - cplusplus_cost;
    run_op_cost = CalcOverlapTotal(collector, [](const HostTraceEvent& evt) {
      return evt.type == TracerEventType::Operator;
    });
  }
  VLOG(1) << "total_cost"
          << "\t"
          << "python_cost"
          << "\t"
          << "cplusplus_cost"
          << "\t"
          << "run_op_cost";

  VLOG(1) << total_cost << "\t" << python_cost << "\t" << cplusplus_cost << "\t"
          << run_op_cost;
/*uint64_t allocator_cost = 0;
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
uint64_t op_run = CalcOverlapTotal(collector, [](const HostTraceEvent& evt) {
  return evt.type == TracerEventType::Operator;
});
VLOG(1) << "run op(overlap): " << op_run << "ns ";
uint64_t infer_shape = CalcOverlapTotal(
    collector,
    [](const HostTraceEvent& evt) { return evt.name == "infer_shape"; });
VLOG(1) << "infershape(overlap): " << infer_shape << "ns ";
uint64_t kernel_luanch = CalcOverlapTotal(collector,
                                          [](const HostTraceEvent& evt) {
                                            return evt.name == "compute";
                                          }) -
                         allocator_cost;
VLOG(1) << "kernel luanch(overlap): " << kernel_luanch << "ns ";
VLOG(1) << "op count: " << GET_EVENT_COUNT(compute);
VLOG(1) << "op gaps: " << CalcOverlapGaps(collector);*/
#undef GET_EVENT_TATALTIME
#undef GET_EVENT_COUNT
}

}  // namespace platform
}  // namespace paddle
