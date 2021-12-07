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
#include <unordered_set>

namespace paddle {
namespace framework {

/*
 * Parse the var_ids that need to be associated with an event.
 * The caller should guarantee front_op and back_op satisfy the
 * following conditions:
 *   1. kQueueAsync -> kQueueAsync
 *   2. kQueueAsync -> kQueueSync
 *
 * For example: matmul(gpu) -> out_var -> memcpy_d2h
 * out_var should be associated with an event.
 */
std::vector<size_t> StreamAnalyzer::ParseEventVarIds(
    const Instruction& cur_instr, const Instruction& next_instr) {
  std::unordered_set<size_t> unique_var_ids;
  for (auto& item : cur_instr.output_index_) {
    unique_var_ids.insert(item.second.begin(), item.second.end());
  }

  std::vector<size_t> new_event_var_ids;
  for (auto& item : next_instr.input_index_) {
    for (auto var_id : item.second) {
      if (unique_var_ids.count(var_id) > 0) {
        new_event_var_ids.push_back(var_id);
      }
    }
  }
  return new_event_var_ids;
}

void StreamAnalyzer::AssociateInputWithEvents(
    const std::vector<size_t>& new_event_var_id, Instruction* next_instr,
    bool is_sync) {
  for (auto var_id : new_event_var_id) {
    if (var_id2event_.count(var_id) == 0) {
      auto device_event = std::make_shared<platform::DeviceEvent>(
          place_, platform::GenerateDeviceEventFlag());
      var_id2event_.emplace(var_id, std::move(device_event));
    }
    // Add events for next_instr.inputs
    next_instr->intput_events_.emplace_back(var_id, var_id2event_.at(var_id),
                                            is_sync);
  }
}

void StreamAnalyzer::Schedule(const std::vector<OpFuncNode>& op_func_nodes,
                              const std::vector<size_t>& downstream_ops,
                              size_t op_index,
                              std::vector<Instruction>* instructions) {
  auto& op_func_type = op_func_nodes[op_index].type_;
  auto& cur_instr = instructions->at(op_index);
  auto& next_instruction = cur_instr.next_instruction_;

  if (op_func_type == OpFuncType::kQueueSync) {
    // all downstream ops of kQueueSync can directly run, such as CPU -> Any
    next_instruction.direct_run_ = downstream_ops;
  } else {  // kQueueAsync
    std::vector<size_t> event_var_ids;
    for (auto next_op_id : downstream_ops) {
      auto& next_instr = instructions->at(next_op_id);
      // case 1: GPU -> GPU(same stream)
      if (cur_instr.dev_ctx_ == next_instr.dev_ctx_) {
        next_instruction.direct_run_.emplace_back(next_op_id);
        continue;
      }
      // Always insert events between different stream
      auto new_event_var_ids = ParseEventVarIds(cur_instr, next_instr);
      event_var_ids.insert(event_var_ids.end(), new_event_var_ids.begin(),
                           new_event_var_ids.end());

      bool is_sync =
          (op_func_nodes[next_op_id].type_ == OpFuncType::kQueueSync);
      AssociateInputWithEvents(new_event_var_ids, &next_instr, is_sync);

      if (is_sync) {  // GPU -> CPU
        next_instruction.synchronize_run_.emplace_back(next_op_id);
      } else {  // GPU -> GPU(different stream)
        next_instruction.event_wait_run_.emplace_back(next_op_id);
      }
    }
    // Create events for these cross-stream vars
    VLOG(3) << cur_instr.kernel_func_.operator_base_->Type()
            << " event_var_ids.size: " << event_var_ids.size();
    for (auto var_id : event_var_ids) {
      cur_instr.output_events_.emplace_back(var_id, var_id2event_.at(var_id),
                                            false /*not used*/);
    }
  }
}

platform::DeviceContext* StreamAnalyzer::ParseDeviceContext(
    const OpFuncNode& op_func_node, const OperatorBase& op_base) {
  auto& op_type = op_base.Type();
  auto* dev_ctx = op_func_node.dev_ctx_;
  if (op_type == interpretercore::kMemcpyH2D) {
    VLOG(3) << "Get dev_ctx from d2h_context_pool_";
    dev_ctx = d2h_ctx_pool_.Get(place_);
  } else if (op_type == interpretercore::kMemcpyD2H) {
    VLOG(3) << "Get dev_ctx from h2d_context_pool_";
    dev_ctx = h2d_ctx_pool_.Get(place_);
  }

  return dev_ctx;
}

}  // namespace framework
}  // namespace paddle
