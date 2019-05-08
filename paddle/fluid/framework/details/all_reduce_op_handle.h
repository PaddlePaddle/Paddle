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
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/framework/details/nccl_op_handle.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
class AllReduceOpHandle : public NCCLOpHandleBase {
 public:
  AllReduceOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                    const std::vector<platform::Place> &places,
                    const platform::NCCLContextMap *ctxs);
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
  void SetNCCLContextMap(const platform::NCCLContextMap *ctxs) override {
    nccl_ctxs_ = ctxs;
    for (auto &p : places_) {
      this->SetDeviceContext(p, nccl_ctxs_->DevCtx(p));
    }
  }

 protected:
  void RunImpl() override;

  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  void RunAllReduceFuncs(
      const std::vector<std::function<void()>> &all_reduce_calls);
// const platform::NCCLContextMap *nccl_ctxs_;
#endif
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
