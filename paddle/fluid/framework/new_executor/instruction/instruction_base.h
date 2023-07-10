// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace framework {

// struct OpFuncNode {
//   int stream_priority_{0};  // lower value, higher priority
//   // fit for phi kernel
//   phi::Kernel* phi_kernel_{nullptr};  // not owned
//   platform::DeviceContext* dev_ctx_;  // not owned

//   std::map<int, int> inplace_back_map;

//   std::map<std::string, std::vector<int>> input_index;
//   std::map<std::string, std::vector<int>> output_index;

//   // TODO(zhiqiu): Better make it unique_ptr
//   std::shared_ptr<OperatorBase> operator_base_{nullptr};
//   std::string execution_stream_{kDefaultStream};

//   OpFuncType type_;
//   OpKernelComputeFunc kernel_func_;

//   SchedulingPriority scheduling_priority_{0};  // lower value, higher
//   priority

//   // the next only for new IR
//   phi::KernelContext kernel_context_;
//   phi::InferMetaContext infer_meta_context_;
//   std::string phi_op_name_;
//   paddle::dialect::InferMetaInterface::Concept*
//   infer_meta_interface_{nullptr};
// };

class InstructionBase {
 public:
  explicit InstructionBase(size_t id);

  bool IsArtificial() const { return is_artificial_; }

  //   const std::vector<size_t>& NextInstrsInDifferenceThread() const {
  //     return next_instrs_in_different_thread;
  //   }

  //   const std::vector<size_t>& NextInstrsInSameThread() const {
  //     return next_instrs_in_same_thread;
  //   }

  //   size_t Id() const { return id_; }

  //   void AddEventToRecord(std::shared_ptr<platform::DeviceEvent> event,
  //                         platform::DeviceType waiter_type) {
  //     event_to_record_ = std::make_shared<EventInter>(id_, event,
  //     waiter_type);
  //   }

  //   void AddEventToWait(size_t instr_id,
  //                       std::shared_ptr<platform::DeviceEvent> event,
  //                       platform::DeviceType waiter_type) {
  //     events_to_wait_.emplace_back(instr_id, event, waiter_type);
  //   }

  //   const std::vector<EventInter>& EventsToWait() const {
  //     return events_to_wait_;
  //   }

  //   void AddNextInstrInDifferentThread(size_t id) {
  //     next_instrs_in_different_thread.push_back(id);
  //   }

  //   void AddNextInstrInSameThread(size_t id) {
  //     next_instrs_in_same_thread.push_back(id);
  //   }

  //   void RecordEvent(const Place& place) const;

  //   void WaitEvent(const Place& place) const;

  //   const std::map<std::string, std::vector<int>>& Inputs() const;

  //   const std::map<std::string, std::vector<int>>& Outputs() const;

  //   const std::unordered_set<int>& NoDataTransformVars() const;

  //   OpKernelComputeFunc KernelFunc() const;

  //   phi::Kernel* PhiKernel() const;

  //   OpFuncType KernelType() const;

  //   const std::map<int, int>& InplaceBackMap() const;

  //   OperatorBase* OpBase() const;

  //   bool OpBaseValid() const;

  //   void AddGCCheckVar(size_t id);

  //   const std::vector<size_t>& GCCheckVars() const;

  //   void ResetContext(const VariableValueMap& in_vars,
  //                     const VariableValueMap& out_vars);

  //   void ResetContextWithScope(const VariableValueMap& in_vars,
  //                              const VariableValueMap& out_vars,
  //                              const framework::Scope& scope);

  //   std::shared_ptr<RuntimeContext> InnerRuntimeContext() const;

  //   std::shared_ptr<RuntimeInferShapeContext> InnerInferShapeContext() const;

  //   std::shared_ptr<ExecutionContext> InnerExecutionContext() const;

  //   const platform::DeviceContext& DeviceContext() const;

  //   const std::vector<std::pair<Variable*, Variable*>>& InplaceInfo() const;

  //   void AddInplace(Variable* in, Variable* out);

  //   void ClearInplace();

  //   SchedulingPriority GetSchedulingPriority() const {
  //     return op_func_node_.scheduling_priority_;
  //   }

  //   bool PreDefineContext() const { return pre_define_context_; }

  //   const OpFuncNode* OpFunc() const { return &op_func_node_; }

 private:
  size_t id_;
  bool is_artificial_;  // Instruction is artificial means that it is only used
                        // to assist scheduling and no need to be executed.

  //   std::vector<size_t> next_instrs_in_different_thread;
  //   std::vector<size_t> next_instrs_in_same_thread;

  //   std::shared_ptr<EventInter> event_to_record_;
  //   std::vector<EventInter> events_to_wait_;

  //   OpFuncNode op_func_node_;
  //   const platform::DeviceContext& dev_ctx_;  // not owned

  //   std::shared_ptr<RuntimeContext> runtime_ctx_;
  //   std::shared_ptr<RuntimeInferShapeContext> infershape_ctx_;
  //   std::shared_ptr<ExecutionContext> execution_ctx_;

  //   std::vector<size_t> gc_check_vars_;

  //   std::vector<std::pair<Variable*, Variable*>> vec_inplace_in_to_out_;

  //   bool pre_define_context_{false};
};

}  // namespace framework
}  // namespace paddle
