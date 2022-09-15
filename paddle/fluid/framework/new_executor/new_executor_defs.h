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
#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/fluid/platform/event.h"
#include "paddle/phi/core/utils/rw_lock.h"

#define SCOPE_VARS_READER_LOCK AutoRDLock auto_lock(&vars_lock_);
#define SCOPE_VARS_WRITER_LOCK AutoWRLock auto_lock(&vars_lock_);

namespace paddle {
namespace framework {

using OpKernelComputeFunc = std::function<void(const ExecutionContext&)>;
using OpKernelMap =
    std::unordered_map<OpKernelType, OpKernelComputeFunc, OpKernelType::Hash>;

constexpr int kEmptyVarIndex = 0;

class InterpretercoreInferShapeContext : public InferShapeContext {
 public:
  InterpretercoreInferShapeContext(const OperatorBase& op,
                                   const RuntimeContext& ctx);

  bool HasInput(const std::string& name) const override;

  bool HasOutput(const std::string& name) const override;

  bool HasAttr(const std::string& name) const override;

  bool HasInputs(const std::string& name) const override;

  bool HasOutputs(const std::string& name,
                  bool allow_null = false) const override;

  AttrReader Attrs() const override;

  std::vector<std::string> Inputs(const std::string& name) const override;

  std::vector<std::string> Outputs(const std::string& name) const override;

  std::string GetInputNameByIdx(size_t idx) const override;

  std::string GetOutputNameByIdx(size_t idx) const override;

  void ShareDim(const std::string& in,
                const std::string& out,
                size_t i = 0,
                size_t j = 0) override;

  void ShareAllLoD(const std::string& in,
                   const std::string& out) const override;

  void ShareLoD(const std::string& in,
                const std::string& out,
                size_t i = 0,
                size_t j = 0) const override;

  int32_t GetLoDLevel(const std::string& in, size_t i = 0) const override;

  void SetLoDLevel(const std::string& out,
                   int32_t lod_level,
                   size_t j = 0) const override;

  bool IsRuntime() const override;

  bool IsRunMKLDNNKernel() const override;

  // TODO(paddle-dev): Can this be template?
  paddle::small_vector<InferShapeVarPtr, phi::kInputSmallVectorSize>
  GetInputVarPtrs(const std::string& name) const override;

  paddle::small_vector<InferShapeVarPtr, phi::kOutputSmallVectorSize>
  GetOutputVarPtrs(const std::string& name) const override;

  DDim GetInputDim(const std::string& name) const override;

  std::vector<DDim> GetInputsDim(const std::string& name) const override;

  proto::VarType::Type GetInputVarType(const std::string& name) const override;

  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string& name) const override;

  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string& name) const override;

  void SetOutputDim(const std::string& name, const DDim& dim) override;

  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override;

  const phi::ArgumentMappingFn* GetPhiArgumentMappingFn() const override;

  const phi::KernelSignature* GetPhiDefaultKernelSignature() const override;

  void SetSkipLoD(bool skip);

 protected:
  DDim GetDim(Variable* var) const;

  std::vector<DDim> GetDims(const std::vector<Variable*>& vars) const;

  std::vector<DDim> GetRepeatedDims(const std::string& name) const override;

  void SetDim(Variable* var, const DDim& dim);

  void SetDims(const std::vector<Variable*>& vars,
               const std::vector<DDim>& dims);

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override;

  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<Variable*>& vars) const;

  proto::VarType::Type GetVarType(Variable* var) const;

 private:
  const std::vector<Variable*>& InputVars(const std::string& name) const;

  const std::vector<Variable*>& OutputVars(const std::string& name) const;

  const OperatorBase& op_;
  const RuntimeContext& ctx_;
  bool can_skip_lod_;
};

struct OpKernelFunc {
  OpKernelComputeFunc compute_func_;
};

struct VariableMetaInfo {
  int var_ref_count_{0};
  framework::VarDesc* var_desc_{nullptr};
  bool sikp_inplace_{false};

  VariableMetaInfo() {}
  VariableMetaInfo(int var_ref_count, framework::VarDesc* var_desc)
      : var_ref_count_(var_ref_count), var_desc_(var_desc) {}
};

class VariableScope {
 public:
  explicit VariableScope(Scope* scope);

  Scope* GetMutableScope() const;

  Scope* GetMutableLocalScope() const;

  void SetScope(Scope* scope);

  void SetLocalScope(Scope* local_scope);

  ~VariableScope();

  // Get variable id by name, return -1 if not found
  int GetIdByName(const std::string& name) const;

  // Get variable name by id, return "" if not found
  std::string GetNameById(int id) const;

  bool HasVar(const std::string& name) const;

  int VarId(const std::string& name) const;

  size_t VarSize() const;

  void AddVar(const std::string& name, VarDesc* var_desc);

  Variable* VarRef(int id) const;

  void SetVarDesc(const std::string& name, framework::VarDesc* var_desc);

  paddle::framework::VarDesc* VarDesc(const std::string& name) const;

  paddle::framework::VarDesc* VarDesc(int id) const;

  void CheckExist(int id) const;

  void CheckExist(const std::string& name) const;

  std::vector<VariableMetaInfo>& MutableVecMetaInfo() { return vec_meta_info_; }

  const std::vector<VariableMetaInfo>& VecMetaInfo() const {
    return vec_meta_info_;
  }

  const std::vector<std::pair<std::string, int>>& DataTransferAddedVars()
      const {
    return data_transfer_added_vars_;
  }

  std::vector<std::pair<std::string, int>>& MutableDataTransferAddedVars() {
    return data_transfer_added_vars_;
  }

  std::vector<Variable*>& MutableVarList() { return var_list_; }

  void SetVarSikpInplace(const std::string& name, bool skip);

  bool GetVarSikpInplace(int id) const;

 private:
  // not owned, better remove it since all vars should be
  // accessed by Scope instead of VariableScope
  std::vector<Variable*> var_list_;

  std::map<std::string, int> name2id_;
  std::vector<VariableMetaInfo> vec_meta_info_;

  Scope* scope_{nullptr};
  // TODO(zhiqiu): find a better way to support local scope.
  Scope* local_scope_{nullptr};
  // mutable RWLock vars_lock_;

  // var_name -> var_type
  std::vector<std::pair<std::string, int>> data_transfer_added_vars_;
};

class NextInstruction {
 public:
  void AddDirectRun(size_t id) { direct_run_.push_back(id); }

  void ADDEventRun(size_t id) { event_wait_run_.push_back(id); }

  void AddSyncRun(size_t id) { synchronize_run_.push_back(id); }

  const std::vector<size_t>& DirectRunIds() const { return direct_run_; }

  const std::vector<size_t>& EventRunIds() const { return event_wait_run_; }

  const std::vector<size_t>& SyncRunIds() const { return synchronize_run_; }

 private:
  std::vector<size_t> direct_run_;
  std::vector<size_t> event_wait_run_;
  std::vector<size_t> synchronize_run_;
};

struct EventInter {
  explicit EventInter(size_t var_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type)
      : var_id_(var_id), event_(event), waiter_type_(waiter_type) {}
  size_t var_id_;
  std::shared_ptr<platform::DeviceEvent> event_;
  platform::DeviceType waiter_type_;
};

struct InstructionInfo {
  std::vector<size_t> dependecy_count_;
};

enum class OpFuncType {
  kQueueSync = 0,   // CPU kernel, block host
  kQueueAsync = 1,  // GPU、XPU Kernel or d2h, h2d, send, recv, broadcast
};
class RuntimeInferShapeContext;

struct OpFuncNode {
  // TODO(zhiqiu): Better make it unique_ptr
  std::shared_ptr<OperatorBase> operator_base_;
  std::map<std::string, std::vector<int>> input_index;
  std::map<std::string, std::vector<int>> output_index;
  std::unordered_set<int> no_data_transform_index;

  std::map<int, int> inplace_back_map;

  OpKernelComputeFunc kernel_func_;
  platform::DeviceContext* dev_ctx_;  // not owned

  // fit for phi kernel
  phi::Kernel* phi_kernel_{nullptr};  // not owned

  OpFuncType type_;
};

class Instruction {
 public:
  Instruction(size_t id,
              OpFuncNode&& op_func_node,
              const platform::DeviceContext& dev_ctx);

  size_t Id() const;

  const std::map<std::string, std::vector<int>>& Inputs() const;

  const std::map<std::string, std::vector<int>>& Outputs() const;

  const std::unordered_set<int>& NoDataTransformVars() const;

  OpKernelComputeFunc KernelFunc() const;

  phi::Kernel* PhiKernel() const;

  OpFuncType KernelType() const;

  const std::map<int, int>& InplaceBackMap() const;

  OperatorBase* OpBase() const;

  NextInstruction& NextInstructions();

  const NextInstruction& NextInstructions() const;

  void AddGCCheckVar(size_t id);

  const std::vector<size_t>& GCCheckVars() const;

  void ResetContext(const VariableValueMap& in_vars,
                    const VariableValueMap& out_vars);

  void ResetContextWithScope(const VariableValueMap& in_vars,
                             const VariableValueMap& out_vars,
                             const framework::Scope& scope);

  std::shared_ptr<RuntimeContext> InnerRuntimeContext() const;

  std::shared_ptr<InterpretercoreInferShapeContext> InnerInferShapeContext()
      const;

  std::shared_ptr<ExecutionContext> InnerExecutionContext() const;

  const platform::DeviceContext& DeviceContext() const;

  const std::vector<std::pair<Variable*, Variable*>>& InplaceInfo() const;

  void AddInplace(Variable* in, Variable* out);

  void ClearInplace();

  const std::vector<EventInter>& InputEvents() const;

  const std::vector<EventInter>& OutputEvents() const;

  void AddInputEvent(size_t var_id,
                     std::shared_ptr<platform::DeviceEvent> event,
                     platform::DeviceType waiter_type);

  void AddOutputEvent(size_t var_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type);

 private:
  size_t id_;
  OpFuncNode op_func_node_;
  const platform::DeviceContext& dev_ctx_;  // not owned

  std::shared_ptr<RuntimeContext> runtime_ctx_;
  std::shared_ptr<InterpretercoreInferShapeContext> infershape_ctx_;
  std::shared_ptr<ExecutionContext> execution_ctx_;

  std::vector<size_t> gc_check_var_list_;
  NextInstruction next_instruction_;

  std::vector<EventInter> intput_events_;
  std::vector<EventInter> output_events_;

  std::vector<std::pair<Variable*, Variable*>> vec_inplace_in_to_out_;
};

namespace interpreter {
static constexpr char kMemcpyH2D[] = "memcpy_h2d";
static constexpr char kMemcpyD2H[] = "memcpy_d2h";
static constexpr char kFetchVarName[] = "fetch";

static bool IsMemcpyH2D(const Instruction& instr) {
  return instr.OpBase()->Type() == kMemcpyH2D;
}

static bool IsMemcpyD2H(const Instruction& instr) {
  return instr.OpBase()->Type() == kMemcpyD2H;
}

static bool IsCpuOp(const Instruction& instr) {
  return platform::is_cpu_place(instr.DeviceContext().GetPlace());
}

// is supported heterogeneous place
static bool IsSupportedHetePlace(const phi::Place& place) {
  return platform::is_gpu_place(place) || platform::is_npu_place(place) ||
         platform::is_xpu_place(place) || platform::is_ipu_place(place);
}

}  // namespace interpreter

}  // namespace framework
}  // namespace paddle
