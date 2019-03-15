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
#include <vector>
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
namespace details {

constexpr char kLocalExecScopeName[] = "@LOCAL_SCOPE@";

// Wraps ir::Node and provide helper utilities.
// It's responsible for populating necessary fields of ir::Node.
class OpHandleBase {
 public:
  // Owned by `node`. No need to be deleted explicitly.
  explicit OpHandleBase(ir::Node *node) : node_(node) {
    node_->WrappedBy(this);
  }

  virtual ~OpHandleBase();

  std::string DebugString() const;

  virtual std::string Name() const = 0;

  void Run(bool use_cuda);

  virtual void RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx);

  virtual void RecordWaitEventOnCtx2(const std::vector<VarHandle *> &in_vars,
                                     platform::DeviceContext *waited_ctx);

  void AddInput(VarHandleBase *in);

  void AddOutput(VarHandleBase *out);

  // This method adds the wait events of all the input on all the device
  // context.
  // NODE: This Wait is asynchronous operation.
  virtual void WaitInputVarGenerated();

  // This method adds the wait events of all the input on the specified device
  // context.
  // NODE: This Wait is asynchronous operation.
  virtual void WaitInputVarGenerated(const platform::Place &place);

  virtual bool NeedWait(VarHandleBase *in_var);

  // If the Op involves data transfer of multiple devices that
  // will likely block other computations.
  virtual bool IsMultiDeviceTransfer() const { return false; }

  const platform::DeviceContext *DeviceContext(platform::Place place) {
    auto it = dev_ctxes_.find(place);
    return it != dev_ctxes_.end() ? it->second : nullptr;
  }

  const std::map<platform::Place, platform::DeviceContext *> &DeviceContext() {
    return dev_ctxes_;
  }

  void SetDeviceContext(platform::Place place, platform::DeviceContext *ctx_) {
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

 protected:
  void RunAndRecordEvent(const std::function<void()> &callback);

  void RunAndRecordEvent(platform::Place p,
                         const std::function<void()> &callback);

  virtual void RunImpl() = 0;

  ir::Node *node_;
  std::vector<VarHandleBase *> inputs_;
  std::vector<VarHandleBase *> outputs_;
  std::map<platform::Place, platform::DeviceContext *> dev_ctxes_;

#ifdef PADDLE_WITH_CUDA
  std::unordered_map<int, cudaEvent_t> events_;
#endif

  DISABLE_COPY_AND_ASSIGN(OpHandleBase);
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
