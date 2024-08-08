//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_set>
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace framework {

class Scope;

namespace details {
struct VarHandleBase;
}  // namespace details
namespace ir {
class Node;
}  // namespace ir

namespace details {
using DeviceType = paddle::platform::DeviceType;
namespace p = paddle::platform;
// Wraps ir::Node and provide helper utilities.
// It's responsible for populating necessary fields of ir::Node.
class OpHandleBase {
 public:
  /**
   * NOTE(zjl): Some op should have higher priority than others.
   * The higher priority op would run first without switching
   * threads in Executor.
   */
  enum Priority { kHighest = 0, kNormal = 1 };

  // Owned by `node`. No need to be deleted explicitly.
  explicit OpHandleBase(ir::Node *node) : node_(node) {
    node_->WrappedBy(this);
  }

  TEST_API virtual ~OpHandleBase() PADDLE_MAY_THROW;

  std::string DebugString() const;

  virtual Priority GetPriority() const { return kNormal; }

  virtual bool GetSkipRunning() const { return skip_running_; }

  virtual void SetSkipRunning(bool skip_running) {
    skip_running_ = skip_running;
  }

  virtual std::string Name() const = 0;

  TEST_API void Run(DeviceType use_device);

  TEST_API virtual void RecordWaitEventOnCtx(phi::DeviceContext *waited_ctx);

  TEST_API void AddInput(VarHandleBase *in);

  TEST_API void AddOutput(VarHandleBase *out);

  // This method adds the wait events of all the input on the specified device
  // context.
  // NOTE: This Wait is asynchronous operation.
  TEST_API virtual void WaitInputVarGenerated(const phi::Place &place);

  TEST_API virtual bool NeedWait(VarHandleBase *in_var);

  // If the Op involves data transfer of multiple devices that
  // will likely block other computations.
  virtual bool IsMultiDeviceTransfer() { return false; }

  const phi::DeviceContext *DeviceContext(phi::Place place) {
    auto it = dev_ctxes_.find(place);
    return it != dev_ctxes_.end() ? it->second : nullptr;
  }
  const std::map<phi::Place, phi::DeviceContext *> &DeviceContext() {
    return dev_ctxes_;
  }

  void SetDeviceContext(phi::Place place, phi::DeviceContext *ctx_) {
    dev_ctxes_[place] = ctx_;
  }

  const std::vector<VarHandleBase *> &Inputs() const { return inputs_; }

  size_t NoDupInputSize() const {
    std::unordered_set<VarHandleBase *> res;
    for (auto *var : inputs_) {
      res.emplace(var);
    }
    return res.size();
  }

  size_t NotReadyInputSize() const;

  const std::vector<VarHandleBase *> &Outputs() const { return outputs_; }

  size_t NoDummyInputSize() const;

  ir::Node *Node() { return node_; }

  const ir::Node *Node() const { return node_; }

  TEST_API void SetLocalExecScopes(
      const std::unordered_map<Scope *, Scope *> &scope_map);

  void SetIsVariantScope(bool is_variant_scope) {
    is_variant_scope_ = is_variant_scope;
  }

 protected:
  virtual std::vector<Scope *> GetLocalScopes() = 0;

  void RunAndRecordEvent(const std::function<void()> &callback);

  void RunAndRecordEvent(phi::Place p, const std::function<void()> &callback);

  virtual void RunImpl() = 0;

  TEST_API virtual void InitCUDA();
  TEST_API virtual void InitXPU();

  ir::Node *node_;
  std::vector<VarHandleBase *> inputs_;
  std::vector<VarHandleBase *> outputs_;
  std::map<phi::Place, phi::DeviceContext *> dev_ctxes_;

  std::vector<Scope *> local_exec_scopes_;
  bool skip_running_ = false;
  // NOTE(Aurelius84): Indicate whether scope held in OpHandle is changeable.
  // Ophandle's scope normally keep same in most cases, except running
  // run_program_op from @to_static.
  // The scope may be changed while each training iteration.
  // See https://github.com/PaddlePaddle/Paddle/pull/32283
  bool is_variant_scope_ = false;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::unordered_map<int, gpuEvent_t> events_;
#endif

  DISABLE_COPY_AND_ASSIGN(OpHandleBase);
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
