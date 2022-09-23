//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/fused_all_reduce_op_handle.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
namespace platform {
class NCCLCommunicator;
}  // namespace platform
}  // namespace paddle
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/framework/details/nccl_op_handle.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
namespace details {

class GradMergeAllReduceOpHandle : public AllReduceOpHandle {
 public:
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  GradMergeAllReduceOpHandle(ir::Node *node,
                             const std::vector<Scope *> &local_scopes,
                             const std::vector<platform::Place> &places,
                             const std::string &grad_merge_cond_name,
                             const platform::NCCLCommunicator *ctxs);
#elif defined(PADDLE_WITH_XPU_BKCL)
  GradMergeAllReduceOpHandle(ir::Node *node,
                             const std::vector<Scope *> &local_scopes,
                             const std::vector<platform::Place> &places,
                             const std::string &grad_merge_cond_name,
                             const platform::BKCLCommunicator *ctxs);
#else
  GradMergeAllReduceOpHandle(ir::Node *node,
                             const std::vector<Scope *> &local_scopes,
                             const std::vector<platform::Place> &places,
                             const std::string &grad_merge_cond_name);
#endif
  std::string Name() const override;

  std::string GradMergeCondName() { return grad_merge_cond_name_; }

 protected:
  void RunImpl() override;

 private:
  std::string grad_merge_cond_name_;
};

class FusedGradMergeAllReduceOpHandle : public FusedAllReduceOpHandle {
 public:
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  FusedGradMergeAllReduceOpHandle(ir::Node *node,
                                  const std::vector<Scope *> &local_scopes,
                                  const std::vector<platform::Place> &places,
                                  const size_t num_of_all_reduce,
                                  const std::string &grad_merge_cond_name,
                                  const platform::NCCLCommunicator *ctxs);
#elif defined(PADDLE_WITH_XPU_BKCL)
  FusedGradMergeAllReduceOpHandle(ir::Node *node,
                                  const std::vector<Scope *> &local_scopes,
                                  const std::vector<platform::Place> &places,
                                  const size_t num_of_all_reduce,
                                  const std::string &grad_merge_cond_name,
                                  const platform::BKCLCommunicator *ctxs);
#else
  FusedGradMergeAllReduceOpHandle(ir::Node *node,
                                  const std::vector<Scope *> &local_scopes,
                                  const std::vector<platform::Place> &places,
                                  const size_t num_of_all_reduce,
                                  const std::string &grad_merge_cond_name);
#endif

  std::string Name() const override;

  std::string GradMergeCondName() { return grad_merge_cond_name_; }

 protected:
  void RunImpl() override;

 private:
  std::string grad_merge_cond_name_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
