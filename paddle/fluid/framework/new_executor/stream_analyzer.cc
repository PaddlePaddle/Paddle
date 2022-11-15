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

#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class ContextManager {
 public:
  using DeviceContextMap =
      std::map<Place,
               std::shared_future<std::unique_ptr<platform::DeviceContext>>>;

  static ContextManager& Instance() {
    static ContextManager* ctx_manager = new ContextManager;
    return *ctx_manager;
  }

  std::shared_future<std::unique_ptr<platform::DeviceContext>> Get(
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

  auto is_shape_op = [](std::string op_name) {
    if (op_name == "shape") {
      return true;
    }
    return false;
  };

  bool is_memcpy =
      interpreter::IsMemcpyOp(cur_instr) || interpreter::IsMemcpyOp(next_instr);

  std::vector<size_t> need_event_var_ids;
  for (auto& item : next_instr.Inputs()) {
    for (auto var_id : item.second) {
      if (unique_var_ids.count(var_id) > 0) {
        if (is_memcpy || is_shape_op(next_instr.OpBase()->Type())) {
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
  auto& op = op_func_node.operator_base_;
  auto& op_type = op->Type();
  const std::string& execution_stream = op_func_node.execution_stream_;
  ContextManager& ctx_manager = ContextManager::Instance();

  // only gpu/npu need update. xpu not need, because xpu memcpy op kernel is
  // synchronous.
  if (platform::is_gpu_place(place_) || platform::is_npu_place(place_) ||
      platform::is_custom_place(place_)) {
    VLOG(7) << "Parse DeviceContext for " << op_type
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
  if ((cur_instr.KernelType() == OpFuncType::kQueueSync) &&
      (next_instr.KernelType() == OpFuncType::kQueueSync)) {
    return true;
  }

  if (cur_instr.KernelType() == next_instr.KernelType() &&
      (&cur_instr.DeviceContext() == &next_instr.DeviceContext())) {
    return true;
  }

  // xpu&ipu memcpy kerenl is synchronous.
  if (platform::is_ipu_place(place_) || platform::is_xpu_place(place_)) {
    return true;
  }

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
