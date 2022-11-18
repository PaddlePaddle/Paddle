// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/interpreter/stream_analyzer.h"

#include <future>
#include <unordered_set>

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace interpreter {

using DeviceContext = platform::DeviceContext;
using DeviceEvent = platform::DeviceEvent;

class ContextManager {
 public:
  using DeviceContextMap =
      std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>;

  static ContextManager& Instance() {
    static ContextManager* ctx_manager = new ContextManager;
    return *ctx_manager;
  }

  std::shared_future<std::unique_ptr<DeviceContext>> Get(
      const std::string& type, const platform::Place& place) {
    std::lock_guard<std::mutex> lk(ctx_mtx_);
    VLOG(6) << "Get dev_ctx for " << type << " - " << place;

    DeviceContextMap& ctxs = ctx_pool_[type];
    if (ctxs.find(place) == ctxs.end()) {
      platform::EmplaceDeviceContexts(
          &ctxs,
          {place},
          /*disable_setting_default_stream_for_allocator=*/true);
    }
    return ctxs[place];
  }

 private:
  ContextManager() {}
  DISABLE_COPY_AND_ASSIGN(ContextManager);

  std::mutex ctx_mtx_;
  std::unordered_map<std::string, DeviceContextMap> ctx_pool_;
};

void StreamAnalyzer::ConstructEvents(
    const DependencyBuilder& dependency_builder,
    std::vector<Instruction>* instructions) const {
  // parse startup op
  const std::map<size_t, std::set<size_t>>& downstream_map =
      dependency_builder.OpDownstreamMap();
  const size_t instr_num = instructions->size();
  std::vector<bool> is_startup_instrs(instr_num, true);
  std::vector<size_t> startup_instrs;
  for (size_t instr_id = 0; instr_id < instr_num; ++instr_id) {
    auto it = downstream_map.find(instr_id);
    if (it != downstream_map.end()) {
      for (size_t next_instr_id : it->second) {
        is_startup_instrs[next_instr_id] = false;
      }
    }
  }
  for (size_t instr_id = 0; instr_id < instr_num; ++instr_id) {
    if (is_startup_instrs[instr_id]) {
      startup_instrs.push_back(instr_id);
    }
  }

  std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>
      event_info_map;  // DeviceContext -> waiter_instr_id -> recorder_instr_ids
  AnalyseAllRunType(downstream_map, startup_instrs, instructions);
  AnalyseAllEventInfo(
      *instructions, startup_instrs, dependency_builder, &event_info_map);
  ShrinkEventInfo(dependency_builder, &event_info_map);

  // Construct events
  std::map<size_t, std::shared_ptr<DeviceEvent>> instr2event;
  for (auto& context_item : event_info_map) {
    for (auto& waiter_item : context_item.second) {
      size_t waiter_instr_id = waiter_item.first;
      std::set<size_t>& recorder_instr_ids = waiter_item.second;
      for (size_t recorder_instr_id : recorder_instr_ids) {
        Instruction& recorder_instr = instructions->at(recorder_instr_id);
        Instruction& waiter_instr = instructions->at(waiter_instr_id);
        platform::DeviceType waiter_type = GetWaiterType(waiter_instr);

        if (instr2event.find(recorder_instr_id) == instr2event.end()) {
          std::shared_ptr<DeviceEvent> device_event =
              std::make_shared<DeviceEvent>(
                  recorder_instr.DeviceContext().GetPlace(),
                  platform::GenerateDeviceEventFlag());
          recorder_instr.AddEventToRecord(device_event,
                                          platform::kCUDA /*unused*/);
          instr2event.emplace(recorder_instr_id, device_event);
        }

        waiter_instr.AddEventToWait(
            recorder_instr_id, instr2event.at(recorder_instr_id), waiter_type);
        VLOG(6) << "Add event: " << recorder_instr.OpBase()->Type() << "("
                << recorder_instr_id << ") -> " << waiter_instr.OpBase()->Type()
                << "(" << waiter_instr_id << "), waiter type = " << waiter_type;
      }
    }
  }
}

DeviceContext* StreamAnalyzer::ParseDeviceContext(
    const OpFuncNode& op_func_node) const {
  auto& op = op_func_node.operator_base_;
  auto& op_type = op->Type();
  const std::string& execution_stream = op_func_node.execution_stream_;
  ContextManager& ctx_manager = ContextManager::Instance();

  // only gpu/npu need update. xpu not need, because xpu memcpy op kernel is
  // synchronous.
  if (platform::is_gpu_place(place_) || platform::is_npu_place(place_) ||
      platform::is_custom_place(place_)) {
    VLOG(6) << "Parse DeviceContext for " << op_type
            << ", execution stream = " << execution_stream;
    if (execution_stream != kDefaultStream) {
      return ctx_manager
          .Get(std::string(kCustomStream) + "-" + execution_stream, place_)
          .get()
          .get();
    }

    if (op_type == interpreter::kMemcpyD2H) {
      return ctx_manager.Get(std::string(kD2HStream), place_).get().get();
    } else if (op_type == interpreter::kMemcpyH2D) {
      return ctx_manager.Get(std::string(kH2DStream), place_).get().get();
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    // NOTE(Ruibiao): Here supports multi-stream overlap for c_allreduce_sum
    // with use_cal_stream==false by returning a device context getting from the
    // global NCCLCommContext instance. Because when use_calc_stream==false, in
    // OP kernel, the NCCL communication will be launched to the stream directly
    // getting from the global NCCLCommContext instance rather than the
    // DeviceContext passed from executor (see CAllReduceOpCUDAKernel in
    // c_allreduce_op.h). Now it is just a temporary solution for ONLY
    // c_allreduce_sum which is used in ResNet50 distributed training.
    if (op_type == "c_allreduce_sum" &&
        op->Attr<bool>("use_calc_stream") == false) {
      int ring_id = op->Attr<int>("ring_id");
      return platform::NCCLCommContext::Instance()
          .Get(ring_id, place_)
          ->dev_context();
    }
#endif
  }

  return op_func_node.dev_ctx_;
}

bool StreamAnalyzer::HasDataDependency(const Instruction& cur_instr,
                                       const Instruction& next_instr) const {
  auto no_need_buffer_ins =
      [](const Instruction& instr) -> const std::unordered_set<std::string> {
    auto* op = instr.OpBase();
    auto& inferer = op->Info().NoNeedBufferVarsInferer();
    if (inferer) {
      return inferer(op->Inputs(), op->Outputs(), op->Attrs());
    }
    return std::unordered_set<std::string>();
  };

  // cur_instr->var->next_instr
  std::unordered_set<size_t> cur_var_ids;
  for (auto& item : cur_instr.Outputs()) {
    cur_var_ids.insert(item.second.begin(), item.second.end());
  }

  const std::unordered_set<std::string> next_instr_no_need_buffer_ins =
      no_need_buffer_ins(next_instr);

  for (auto& item : next_instr.Inputs()) {
    if (next_instr_no_need_buffer_ins.find(item.first) !=
        next_instr_no_need_buffer_ins.end()) {
      continue;
    }
    for (auto next_var_id : item.second) {
      if (cur_var_ids.find(next_var_id) != cur_var_ids.end()) {
        VLOG(6) << "Found data dependency from " << cur_instr.OpBase()->Type()
                << "(" << cur_instr.Id() << ") to "
                << next_instr.OpBase()->Type() << "(" << next_instr.Id()
                << ") at variable " << item.first << "(" << next_var_id << ")";
        return true;
      }
    }
  }

  // cur_instr->var && next_instr->var
  // var->cur_instr && next_instr->var
  const std::unordered_set<std::string> cur_instr_no_need_buffer_ins =
      no_need_buffer_ins(cur_instr);
  for (auto& item : cur_instr.Inputs()) {
    if (cur_instr_no_need_buffer_ins.find(item.first) ==
        cur_instr_no_need_buffer_ins.end()) {
      cur_var_ids.insert(item.second.begin(), item.second.end());
    }
  }

  for (auto& item : next_instr.Outputs()) {
    for (auto next_var_id : item.second) {
      if (cur_var_ids.find(next_var_id) != cur_var_ids.end()) {
        VLOG(6) << "Found data dependency from " << cur_instr.OpBase()->Type()
                << "(" << cur_instr.Id() << ") to "
                << next_instr.OpBase()->Type() << "(" << next_instr.Id()
                << ") at variable " << item.first << "(" << next_var_id << ")";
        return true;
      }
    }
  }

  return false;
}

void StreamAnalyzer::AnalyseAllEventInfo(
    const std::vector<Instruction>& instructions,
    const std::vector<size_t>& startup_instrs,
    const DependencyBuilder& dependency_builder,
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
        event_info_map) const {
  std::queue<size_t> queue;
  std::vector<bool> is_visited_instrs(instructions.size(), false);
  for (size_t startup_instr_id : startup_instrs) {
    queue.push(startup_instr_id);
    is_visited_instrs[startup_instr_id] = true;
  }

  while (!queue.empty()) {
    size_t cur_instr_id = queue.front();
    queue.pop();

    const Instruction& cur_instr = instructions[cur_instr_id];
    const NextInstructionList& next_instructions = cur_instr.NextInstructions();
    std::set<size_t> waiter_instr_ids;

    std::vector<size_t> next_instr_ids = next_instructions.SyncRunIds();
    next_instr_ids.insert(next_instr_ids.end(),
                          next_instructions.EventRunIds().begin(),
                          next_instructions.EventRunIds().end());
    for (size_t next_instr_id : next_instr_ids) {
      AnalyseEventInfoForTwoInstructions(
          instructions, cur_instr_id, next_instr_id, &waiter_instr_ids);
    }

    for (size_t waiter_instr_id : waiter_instr_ids) {
      (*event_info_map)[&(cur_instr.DeviceContext())][waiter_instr_id].insert(
          cur_instr_id);
    }

    next_instr_ids.insert(next_instr_ids.end(),
                          next_instructions.DirectRunIds().begin(),
                          next_instructions.DirectRunIds().end());
    for (size_t next_instr_id : next_instr_ids) {
      if (!is_visited_instrs[next_instr_id]) {
        queue.push(next_instr_id);
        is_visited_instrs[next_instr_id] = true;
      }
    }
  }
}

void StreamAnalyzer::AnalyseAllRunType(
    const std::map<size_t, std::set<size_t>>& downstream_map,
    const std::vector<size_t>& startup_instrs,
    std::vector<Instruction>* instructions) const {
  std::queue<size_t> queue;
  std::vector<bool> is_visited_instrs(instructions->size(), false);
  for (size_t startup_instr_id : startup_instrs) {
    queue.push(startup_instr_id);
    is_visited_instrs[startup_instr_id] = true;
  }

  while (!queue.empty()) {
    size_t cur_instr_id = queue.front();
    queue.pop();
    auto it = downstream_map.find(cur_instr_id);
    if (it != downstream_map.end()) {
      Instruction& cur_instr = instructions->at(cur_instr_id);
      NextInstructionList& next_instructions = cur_instr.NextInstructions();
      for (size_t next_instr_id : it->second) {
        Instruction& next_instr = instructions->at(next_instr_id);
        DownstreamRunType run_type =
            AnalyseRunTypeForTwoInstructions(cur_instr, next_instr);
        switch (run_type) {
          case DownstreamRunType::kDirectRun:
            next_instructions.AddDirectRun(next_instr_id);
            VLOG(6) << "DirectRun: " << cur_instr.OpBase()->Type() << "("
                    << cur_instr_id << ") -> " << next_instr.OpBase()->Type()
                    << "(" << next_instr_id << ")";
            break;
          case DownstreamRunType::kSyncRun:
            next_instructions.AddSyncRun(next_instr_id);
            VLOG(6) << "SyncRun: " << cur_instr.OpBase()->Type() << "("
                    << cur_instr_id << ") -> " << next_instr.OpBase()->Type()
                    << "(" << next_instr_id << ")";
            break;
          case DownstreamRunType::kEventRun:
            next_instructions.AddEventRun(next_instr_id);
            VLOG(6) << "EventRun: " << cur_instr.OpBase()->Type() << "("
                    << cur_instr_id << ") -> " << next_instr.OpBase()->Type()
                    << "(" << next_instr_id << ")";
            break;
          default:
            PADDLE_THROW(phi::errors::Unavailable(
                "Unrecognized DownstreamRunType from %s(%d) to %s(%d).",
                cur_instr.OpBase()->Type(),
                cur_instr_id,
                next_instr.OpBase()->Type(),
                next_instr_id));
        }

        if (!is_visited_instrs[next_instr_id]) {
          queue.push(next_instr_id);
          is_visited_instrs[next_instr_id] = true;
        }
      }
    }
  }
}

// The caller should guarantee cur_instr and next_instr is kSyncRun or kEventRun
void StreamAnalyzer::AnalyseEventInfoForTwoInstructions(
    const std::vector<Instruction>& instructions,
    const size_t cur_instr_id,
    const size_t next_instr_id,
    std::set<size_t>* waiter_instr_ids) const {
  if (HasDataDependency(instructions[cur_instr_id],
                        instructions[next_instr_id])) {
    waiter_instr_ids->insert(next_instr_id);
    return;
  }

  // NOTE(Ruibiao): If no data dependency from cur_instr to next_instr, we try
  // to recursively add events between cur_instr and next_instr's
  // direct-run-instrs. When next_instr has too many direct-run-instrs, it may
  // perform worse than add event directly between cur_instr and next_instr.
  for (size_t instr_id :
       instructions[next_instr_id].NextInstructions().DirectRunIds()) {
    AnalyseEventInfoForTwoInstructions(
        instructions, cur_instr_id, instr_id, waiter_instr_ids);
  }
}

// waiter instr should only wait events for the last recorder instrs in each
// stream
void StreamAnalyzer::ShrinkEventInfo(
    const DependencyBuilder& dependency_builder,
    std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
        event_info_map) const {
  for (auto& context_item : *event_info_map) {
    for (auto& waiter_item : context_item.second) {
      size_t waiter_instr_id = waiter_item.first;
      std::set<size_t>& recorder_instr_ids = waiter_item.second;
      std::set<size_t> unnecessary_recorder_instr_ids;
      for (size_t cur_instr_id : recorder_instr_ids) {
        for (size_t next_instr_id : recorder_instr_ids) {
          if (dependency_builder.OpHappensBefore(cur_instr_id, next_instr_id)) {
            unnecessary_recorder_instr_ids.insert(cur_instr_id);
            break;
          }
        }
      }

      for (size_t unnecessary_recorder_instr_id :
           unnecessary_recorder_instr_ids) {
        VLOG(8) << "Shrink event : " << unnecessary_recorder_instr_id << " -> "
                << waiter_instr_id;
        recorder_instr_ids.erase(unnecessary_recorder_instr_id);
      }
    }
  }
}

platform::DeviceType StreamAnalyzer::GetWaiterType(
    const Instruction& instr) const {
  if (instr.KernelType() == OpFuncType::kCpuSync) {
    return platform::kCPU;
  } else {
    if (platform::is_xpu_place(place_)) {
      return platform::kXPU;
    } else if (platform::is_npu_place(place_)) {
      return platform::kNPU;
    } else if (platform::is_custom_place(place_)) {
      return platform::kCUSTOM_DEVICE;
    }
    return platform::kCUDA;
  }
}

DownstreamRunType StreamAnalyzer::AnalyseRunTypeForTwoInstructions(
    const Instruction& cur_instr, const Instruction& next_instr) const {
  // xpu&ipu memcpy kerenl is synchronous.
  if (platform::is_ipu_place(place_) || platform::is_xpu_place(place_)) {
    return DownstreamRunType::kDirectRun;
  }

  // npu d2h kernel is asynchronous.
  if (platform::is_npu_place(place_) || platform::is_custom_place(place_)) {
    if (interpreter::IsCpuOp(cur_instr) ||
        interpreter::IsMemcpyH2D(next_instr)) {
      return DownstreamRunType::kDirectRun;
    }
  }

  if (cur_instr.KernelType() == OpFuncType::kGpuAsync) {
    if (next_instr.KernelType() == OpFuncType::kCpuSync) {
      return DownstreamRunType::kSyncRun;
    } else {
      // cross-stream: kGpuAsync -> kGpuSync, kGpuAsync -> kGpuSync
      if (&cur_instr.DeviceContext() != &next_instr.DeviceContext()) {
        return DownstreamRunType::kEventRun;
      }
    }
  }

  return DownstreamRunType::kDirectRun;
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
