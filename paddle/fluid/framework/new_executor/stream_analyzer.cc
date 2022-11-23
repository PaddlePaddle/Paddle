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

#include "paddle/fluid/framework/new_executor/stream_analyzer.h"

#include <future>
#include <unordered_set>

#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace {
std::map<Place, std::shared_future<std::unique_ptr<platform::DeviceContext>>>*
    d2h_ctxs = nullptr;
std::map<Place, std::shared_future<std::unique_ptr<platform::DeviceContext>>>*
    h2d_ctxs = nullptr;
std::mutex ctx_mtx;
}  // namespace

StreamAnalyzer::StreamAnalyzer(const platform::Place& place) : place_(place) {
  if (platform::is_gpu_place(place) || platform::is_npu_place(place) ||
      platform::is_custom_place(place)) {
    std::lock_guard<std::mutex> lk(ctx_mtx);
    if (d2h_ctxs == nullptr) {
      d2h_ctxs = new std::map<
          Place,
          std::shared_future<std::unique_ptr<platform::DeviceContext>>>();
      h2d_ctxs = new std::map<
          Place,
          std::shared_future<std::unique_ptr<platform::DeviceContext>>>();
    }
    if (d2h_ctxs->find(place) == d2h_ctxs->end()) {
      platform::EmplaceDeviceContexts(
          d2h_ctxs,
          {place},
          /*disable_setting_default_stream_for_allocator=*/true);
      platform::EmplaceDeviceContexts(
          h2d_ctxs,
          {place},
          /*disable_setting_default_stream_for_allocator=*/true);
    }
    d2h_ctx_ = (*d2h_ctxs)[place];
    h2d_ctx_ = (*h2d_ctxs)[place];
  }
}

/*
 * Parse the var_ids that need to be associated with an event.
 * The caller should guarantee front_op and back_op satisfy the
 * following conditions:
 *   1. kQueueSync -> kQueueAsync
 *   2. kQueueAsync -> kQueueSync
 *
 * For example: matmul(gpu) -> out_var -> memcpy_d2h
 * out_var should be associated with an event.
 *
 * NOTE(zhiqiu): There are two special case that no event is needed:
 *  1. the variable is marked as NoDataTransformVar
 *  2. the variable is marked as NoNeedDataBuffer
 */
std::vector<size_t> StreamAnalyzer::GetNeedEventVarIds(
    const Instruction& cur_instr, const Instruction& next_instr) {
  std::unordered_set<size_t> unique_var_ids;
  for (auto& item : cur_instr.Outputs()) {
    unique_var_ids.insert(item.second.begin(), item.second.end());
  }

  auto is_no_need_buffer = [&next_instr](std::string name) {
    auto* op = next_instr.OpBase();
    auto& inferer = op->Info().NoNeedBufferVarsInferer();
    if (inferer) {
      auto no_need_buffer_ins =
          inferer(op->Inputs(), op->Outputs(), op->Attrs());
      return no_need_buffer_ins.count(name) != 0;
    }
    return false;
  };

  std::vector<size_t> need_event_var_ids;
  for (auto& item : next_instr.Inputs()) {
    for (auto var_id : item.second) {
      if (unique_var_ids.count(var_id) > 0) {
        if (next_instr.NoDataTransformVars().count(var_id)) {
          VLOG(4) << "Skip inserting event at variable " << item.first
                  << " of operator " << next_instr.OpBase()->Type()
                  << " since it is NoDataTransform";
          continue;
        }
        if (is_no_need_buffer(item.first)) {
          VLOG(4) << "Skip inserting event at variable " << item.first
                  << " of operator " << next_instr.OpBase()->Type()
                  << " since it is NoNeedBufferVar";
          continue;
        }

        need_event_var_ids.push_back(var_id);
      }
    }
  }
  return need_event_var_ids;
}

void StreamAnalyzer::ConstructEventForVar(
    const std::vector<size_t>& new_event_var_id,
    Instruction* next_instr,
    platform::DeviceType waiter_type,
    const platform::Place& place) {
  for (auto var_id : new_event_var_id) {
    if (var_id2event_.count(var_id) == 0) {
      auto device_event = std::make_shared<platform::DeviceEvent>(
          place, platform::GenerateDeviceEventFlag());
      var_id2event_.emplace(var_id, std::move(device_event));
    }
    // Add events for next_instr.inputs
    next_instr->AddInputEvent(var_id, var_id2event_.at(var_id), waiter_type);
  }
}

void StreamAnalyzer::Schedule(const std::vector<size_t>& downstream_ops,
                              std::vector<Instruction>* instructions,
                              size_t op_index) {
  auto& cur_instr = instructions->at(op_index);
  auto& next_instruction = cur_instr.NextInstructions();
  std::vector<size_t> event_var_ids;
  for (auto next_op_id : downstream_ops) {
    auto& next_instr = instructions->at(next_op_id);
    if (IsDirectRun(cur_instr, next_instr)) {
      VLOG(4) << "DirectRun: " << cur_instr.OpBase()->Type() << "->"
              << next_instr.OpBase()->Type();
      next_instruction.AddDirectRun(next_op_id);
    } else {
      // Always insert events between different stream
      auto need_event_var_ids = GetNeedEventVarIds(cur_instr, next_instr);
      event_var_ids.insert(event_var_ids.end(),
                           need_event_var_ids.begin(),
                           need_event_var_ids.end());

      auto waiter_type = GetWaiterType(next_instr);
      ConstructEventForVar(need_event_var_ids,
                           &next_instr,
                           waiter_type,
                           cur_instr.DeviceContext().GetPlace());

      if (waiter_type == platform::kCPU) {  // GPU -> CPU
        VLOG(4) << "SyncRun: " << cur_instr.OpBase()->Type() << "->"
                << next_instr.OpBase()->Type();
        next_instruction.AddSyncRun(next_op_id);
      } else {  // GPU -> GPU(different stream)
        VLOG(4) << "EventRun: " << cur_instr.OpBase()->Type() << "->"
                << next_instr.OpBase()->Type();
        next_instruction.ADDEventRun(next_op_id);
      }
    }
  }
  // Create events for these cross-stream vars
  VLOG(3) << cur_instr.OpBase()->Type()
          << " event_var_ids.size: " << event_var_ids.size();
  for (auto var_id : event_var_ids) {
    cur_instr.AddOutputEvent(
        var_id, var_id2event_.at(var_id), platform::kCUDA /*not used*/);
  }
}

platform::DeviceContext* StreamAnalyzer::ParseDeviceContext(
    const OpFuncNode& op_func_node) {
  auto& op_type = op_func_node.operator_base_->Type();
  auto* dev_ctx = op_func_node.dev_ctx_;
  // only gpu/npu need update. xpu not need, because xpu memcpy op kernel is
  // synchronous.
  if (platform::is_gpu_place(place_) || platform::is_npu_place(place_) ||
      platform::is_custom_place(place_)) {
    if (op_type == interpreter::kMemcpyD2H) {
      VLOG(3) << "Get dev_ctx from d2h_context_pool_";
      dev_ctx = d2h_ctx_.get().get();
    } else if (op_type == interpreter::kMemcpyH2D) {
      VLOG(3) << "Get dev_ctx from h2d_context_pool_";
      dev_ctx = h2d_ctx_.get().get();
    }
  }
  return dev_ctx;
}

/*
 * NOTE(dev): The following cases are considered as directly run:
 *
 *  0. in XPU place. because xpu memcpy op kernel is synchronous.
 *  1. with same dev_ctx_, such as: CPU -> CPU, GPU -> GPU
 *  2. CPU -> any (it is possible: CPU op->VAR->GPU op, when var is no need
 * buffer or no need data transform)
 *  3. D2H -> CPU
 *  4. CPU -> H2D
 */
bool StreamAnalyzer::IsDirectRun(Instruction& cur_instr,
                                 const Instruction& next_instr) {
  if (&cur_instr.DeviceContext() == &next_instr.DeviceContext()) return true;

  // xpu&ipu memcpy kerenl is synchronous.
  if (platform::is_ipu_place(place_) || platform::is_xpu_place(place_))
    return true;

  // npu d2h kernel is asynchronous.
  if (platform::is_npu_place(place_) || platform::is_custom_place(place_)) {
    return interpreter::IsCpuOp(cur_instr) ||
           interpreter::IsMemcpyH2D(next_instr);
  }
  // gpu or cpu
  return interpreter::IsCpuOp(cur_instr) ||
         interpreter::IsMemcpyD2H(cur_instr) ||
         interpreter::IsMemcpyH2D(next_instr);
}

platform::DeviceType StreamAnalyzer::GetWaiterType(const Instruction& instr) {
  if (instr.KernelType() == OpFuncType::kQueueSync) {
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

}  // namespace framework
}  // namespace paddle
