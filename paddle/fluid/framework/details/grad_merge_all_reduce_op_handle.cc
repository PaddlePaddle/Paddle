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
#include "paddle/fluid/framework/details/grad_merge_all_reduce_op_handle.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
DECLARE_bool(sync_nccl_allreduce);
#endif

namespace paddle {
namespace framework {
namespace details {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
GradMergeAllReduceOpHandle::GradMergeAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    const std::string &grad_merge_cond_name,
    const platform::NCCLCommunicator *ctxs)
    : AllReduceOpHandle(node, local_scopes, places, ctxs),
      grad_merge_cond_name_(grad_merge_cond_name) {}
#elif defined(PADDLE_WITH_XPU_BKCL)
GradMergeAllReduceOpHandle::GradMergeAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    const std::string &grad_merge_cond_name,
    const platform::BKCLCommunicator *ctxs)
    : AllReduceOpHandle(node, local_scopes, places, ctxs),
      grad_merge_cond_name_(grad_merge_cond_name) {}
#else
GradMergeAllReduceOpHandle::GradMergeAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    const std::string &grad_merge_cond_name)
    : AllReduceOpHandle(node, local_scopes, places),
      grad_merge_cond_name_(grad_merge_cond_name) {}
#endif

void GradMergeAllReduceOpHandle::RunImpl() {
  platform::RecordEvent record_event(
      Name(), platform::TracerEventType::Communication, 1);
  PADDLE_ENFORCE_GT(local_scopes_.size(), 0,
                    platform::errors::PreconditionNotMet(
                        "The number of local scope should be > 0, but got %zu.",
                        local_scopes_.size()));

  auto *local_scope = local_exec_scopes_[0];
  auto cond_var = local_scope->FindVar(grad_merge_cond_name_);
  PADDLE_ENFORCE_NOT_NULL(
      cond_var, platform::errors::NotFound("Variable %s is not found in scope.",
                                           cond_var));
  bool cond = *cond_var->Get<LoDTensor>().data<bool>();

  if (cond) {
    AllReduceOpHandle::RunImpl();
  }
}

std::string GradMergeAllReduceOpHandle::Name() const {
  return "grad_merge_all_reduce";
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
FusedGradMergeAllReduceOpHandle::FusedGradMergeAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, const size_t num_of_all_reduce,
    const std::string &grad_merge_cond_name,
    const platform::NCCLCommunicator *ctxs)
    : FusedAllReduceOpHandle(node, local_scopes, places, num_of_all_reduce,
                             ctxs),
      grad_merge_cond_name_(grad_merge_cond_name) {}
#elif defined(PADDLE_WITH_XPU_BKCL)
FusedGradMergeAllReduceOpHandle::FusedGradMergeAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, const size_t num_of_all_reduce,
    const std::string &grad_merge_cond_name,
    const platform::BKCLCommunicator *ctxs)
    : FusedAllReduceOpHandle(node, local_scopes, places, num_of_all_reduce,
                             ctxs),
      grad_merge_cond_name_(grad_merge_cond_name) {}
#else
FusedGradMergeAllReduceOpHandle::FusedGradMergeAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, const size_t num_of_all_reduce,
    const std::string &grad_merge_cond_name)
    : FusedAllReduceOpHandle(node, local_scopes, places, num_of_all_reduce),
      grad_merge_cond_name_(grad_merge_cond_name) {}
#endif

void FusedGradMergeAllReduceOpHandle::RunImpl() {
  platform::RecordEvent record_event(
      Name(), platform::TracerEventType::Communication, 1);
  PADDLE_ENFORCE_GT(local_scopes_.size(), 0,
                    platform::errors::PreconditionNotMet(
                        "The number of local scope should be > 0, but got %zu.",
                        local_scopes_.size()));

  auto *local_scope = local_exec_scopes_[0];
  auto cond_var = local_scope->FindVar(grad_merge_cond_name_);
  PADDLE_ENFORCE_NOT_NULL(
      cond_var, platform::errors::NotFound("Variable %s is not found in scope.",
                                           cond_var));
  bool cond = *cond_var->Get<LoDTensor>().data<bool>();

  if (cond) {
    VLOG(10) << "run fused grad merge all reduce";
    FusedAllReduceOpHandle::RunImpl();
  }
}

std::string FusedGradMergeAllReduceOpHandle::Name() const {
  return "fused_grad_merge_all_reduce";
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
