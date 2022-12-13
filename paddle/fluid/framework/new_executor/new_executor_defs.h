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

constexpr const char* kCoalesceTensor = "coalesce_tensor";

// stream types
constexpr const char* kCustomStream = "CustromStream";
constexpr const char* kDefaultStream = "DefaultStream";
constexpr const char* kD2HStream = "D2HStream";
constexpr const char* kH2DStream = "H2DStream";

constexpr int kEmptyVarIndex = 0;

enum class Priority { kLowest, kNormal };

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

struct EventInter {
  explicit EventInter(size_t instr_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type)
      : instr_id_(instr_id), event_(event), waiter_type_(waiter_type) {}
  size_t instr_id_;
  std::shared_ptr<platform::DeviceEvent> event_;
  platform::DeviceType waiter_type_;
};

enum class OpFuncType {
  kCpuSync,  // CPU kernel, block host
  kGpuSync,  // GPU or other device kernel without asynchronous operation
  kGpuAsync  // GPU or other device kernel with asynchronous operation
};
class RuntimeInferShapeContext;

struct OpFuncNode {
  // TODO(zhiqiu): Better make it unique_ptr
  std::shared_ptr<OperatorBase> operator_base_;
  std::string execution_stream_{kDefaultStream};
  std::map<std::string, std::vector<int>> input_index;
  std::map<std::string, std::vector<int>> output_index;

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
              const platform::DeviceContext& dev_ctx,
              const Priority priority);

  bool IsArtificial() const { return is_artificial_; }

  const std::vector<size_t>& NextInstrsInDifferenceThread() const {
    return next_instrs_in_different_thread;
  }

  const std::vector<size_t>& NextInstrsInSameThread() const {
    return next_instrs_in_same_thread;
  }

  size_t Id() const { return id_; }

  void AddEventToRecord(std::shared_ptr<platform::DeviceEvent> event,
                        platform::DeviceType waiter_type) {
    event_to_record_ = std::make_shared<EventInter>(id_, event, waiter_type);
  }

  void AddEventToWait(size_t instr_id,
                      std::shared_ptr<platform::DeviceEvent> event,
                      platform::DeviceType waiter_type) {
    events_to_wait_.emplace_back(instr_id, event, waiter_type);
  }

  const std::vector<EventInter>& EventsToWait() const {
    return events_to_wait_;
  }

  void AddNextInstrInDifferentThread(size_t id) {
    next_instrs_in_different_thread.push_back(id);
  }

  void AddNextInstrInSameThread(size_t id) {
    next_instrs_in_same_thread.push_back(id);
  }

  void RecordEvent(const Place& place) const;

  void WaitEvent(const Place& place) const;

  const std::map<std::string, std::vector<int>>& Inputs() const;

  const std::map<std::string, std::vector<int>>& Outputs() const;

  const std::unordered_set<int>& NoDataTransformVars() const;

  OpKernelComputeFunc KernelFunc() const;

  phi::Kernel* PhiKernel() const;

  OpFuncType KernelType() const;

  const std::map<int, int>& InplaceBackMap() const;

  OperatorBase* OpBase() const;

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

  Priority GetPriority() const { return priority_; }

 private:
  bool is_artificial_;  // Instruction is artificial means that it is only used
                        // to assist scheduling and no need to be executed.

  size_t id_;

  std::vector<size_t> next_instrs_in_different_thread;
  std::vector<size_t> next_instrs_in_same_thread;

  std::shared_ptr<EventInter> event_to_record_;
  std::vector<EventInter> events_to_wait_;

  OpFuncNode op_func_node_;
  const platform::DeviceContext& dev_ctx_;  // not owned
  const Priority priority_;

  std::shared_ptr<RuntimeContext> runtime_ctx_;
  std::shared_ptr<InterpretercoreInferShapeContext> infershape_ctx_;
  std::shared_ptr<ExecutionContext> execution_ctx_;

  std::vector<size_t> gc_check_vars_;

  std::vector<std::pair<Variable*, Variable*>> vec_inplace_in_to_out_;
};

namespace interpreter {
static constexpr char kMemcpyH2D[] = "memcpy_h2d";
static constexpr char kMemcpyD2H[] = "memcpy_d2h";
static constexpr char kFetchVarName[] = "fetch";

// static_ref_ is the numer of last live ops calculated to statically after
// `build` the Instructions. dynamic_ref_  is the runtime version ref which will
// be decreased by one dynamiclly after the execution of an op (in last ops
// list). var_ is the related variable

// The dynamic_ref_ is initialized to static_ref_ first, and is decreased to 1
// during interpretercore's execution, after the interpretercore run, it `reset`
// all dynamic_ref_, i.e., dynamic_ref_ = static_ref_ see ResetAtomicGuard for
// details
class VarRefInfo {
 public:
  explicit VarRefInfo(size_t ref, Variable* var)
      : static_ref_(ref), dynamic_ref_(ref), var_(var) {}
  size_t DynamicRef() { return dynamic_ref_; }
  Variable* Var() { return var_; }
  void ResetDynamicRef() {
    if (static_ref_ != 1) {
      dynamic_ref_ = static_ref_;
    }
  }
  void ResetVariable(Variable* new_var) { var_ = new_var; }
  bool CheckAndDecrease() {
    return static_ref_ == 1 || (dynamic_ref_.fetch_sub(1) == 1);
  }

 private:
  const size_t static_ref_;
  std::atomic<size_t> dynamic_ref_;
  Variable* var_;
};

// static_dep_ is the numer of dependencies (ops that must run before it) of
// each op which is calculated to statically. static_dep_  is the runtime
// version dep which will be decreased by one dynamiclly after the execution of
// one dependency op.

// The dynamic_dep_ is initialized to static_dep_ first, and is decreased to 1
// during interpretercore's execution, after the interpretercore run, it `reset`
// all dynamic_dep_, i.e., dynamic_dep_ = static_dep_ see ResetAtomicGuard for
// details

class OpDepInfo {
 public:
  explicit OpDepInfo(size_t dep) : static_dep_(dep), dynamic_dep_(dep) {}
  size_t DynamicDep() { return dynamic_dep_; }
  void ResetDynamicDep() {
    if (static_dep_ != 1) {
      dynamic_dep_ = static_dep_;
    }
  }
  bool CheckAndDecrease() {
    return static_dep_ == 1 || (dynamic_dep_.fetch_sub(1) == 1);
  }

 private:
  const size_t static_dep_;
  std::atomic<size_t> dynamic_dep_;
};

class ResetAtomicGuard {
 public:
  ResetAtomicGuard(std::vector<std::shared_ptr<OpDepInfo>>* deps,
                   std::vector<std::shared_ptr<VarRefInfo>>* refs)
      : deps_(deps), refs_(refs) {}

  ~ResetAtomicGuard() {
    VLOG(10) << "Reset DynamicDep";
    for (auto&& dep : *deps_) {
      dep->ResetDynamicDep();
    }
    VLOG(10) << "Reset DynamicRef";
    for (auto&& ref : *refs_) {
      ref->ResetDynamicRef();
    }
  }

 private:
  std::vector<std::shared_ptr<OpDepInfo>>* deps_;
  std::vector<std::shared_ptr<VarRefInfo>>* refs_;
};

}  // namespace interpreter

}  // namespace framework
}  // namespace paddle
