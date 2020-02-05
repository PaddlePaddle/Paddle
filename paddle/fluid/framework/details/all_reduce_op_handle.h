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

#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/framework/details/nccl_op_handle.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

#if defined(PADDLE_WITH_NCCL)
class AllReduceOpHandle : public NCCLOpHandleBase {
 public:
  AllReduceOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                    const std::vector<platform::Place> &places,
                    const platform::NCCLCommunicator *ctxs);
#else
class AllReduceOpHandle : public OpHandleBase {
 public:
  AllReduceOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                    const std::vector<platform::Place> &places);
#endif
  std::string Name() const override;

  // Delay and buffer nccl_all_reduce together can significantly increase
  // performance. Disable this feature by returning false.
  bool IsMultiDeviceTransfer() override { return true; };

 protected:
  void RunImpl() override;

  std::vector<Scope *> GetLocalScopes() override { return local_scopes_; }

  std::vector<Scope *> local_scopes_;

#ifndef PADDLE_WITH_NCCL
  // NCCLOpHandleBase already have these attributes.
  // Will polish it by class inheritance framework.
  std::vector<platform::Place> places_;
#endif

#if defined(PADDLE_WITH_NCCL)
  void NCCLAllReduceFunc(
      const std::vector<std::function<void()>> &all_reduce_calls);

  void SyncNCCLAllReduce();
#endif

  void AllReduceImpl(const std::vector<VarHandle *> &in_var_handles,
                     const std::vector<VarHandle *> &out_var_handles);

  void AllReduceFunc(std::vector<const void *> lod_tensor_data,
                     const framework::proto::VarType::Type &dtype,
                     int64_t numel, const std::vector<platform::Place> &places,
                     const std::vector<std::string> &out_var_handles);
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
