/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/parallel_executor.h"

#include "ThreadPool.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"

namespace paddle {
namespace framework {

class ParallelExecutorPrivate {
 public:
  explicit ParallelExecutorPrivate(const std::vector<platform::Place> &places)
      : places_(places) {}

  std::vector<platform::Place> places_;
  std::vector<Scope *> local_scopes_;
  Scope *global_scope_;
  std::unique_ptr<details::SSAGraphExecutor> executor_;

#ifdef PADDLE_WITH_CUDA
  std::unique_ptr<platform::NCCLContextMap> nccl_ctxs_;
#endif
};

ParallelExecutor::ParallelExecutor(
    size_t num_threads, bool use_event,
    const std::vector<platform::Place> &places,
    const std::unordered_set<std::string> &params,
    const ProgramDesc &startup_program, const ProgramDesc &main_program,
    const std::string &loss_var_name, Scope *scope)
    : member_(new ParallelExecutorPrivate(places)) {
  member_->global_scope_ = scope;

  // Step 1. RunStartupProgram and Bcast the params to devs.
  Executor exe(places[0]);
  exe.Run(startup_program, scope, 0);
  // Create local scopes
  for (size_t i = 0; i < member_->places_.size(); ++i) {
    member_->local_scopes_.push_back(&scope->NewScope());
  }

// Bcast Parameters to all GPUs
#ifdef PADDLE_WITH_CUDA
  member_->nccl_ctxs_.reset(new platform::NCCLContextMap(member_->places_));
#endif
  if (platform::is_gpu_place(places[0]) &&
      member_->local_scopes_.size() != 1) {  // Is CUDA
    BCastParamsToGPUs(startup_program);
  }
// Startup Program has been run. All local scopes has correct parameters.

// Step 2. Convert main_program to SSA form and dependency graph. Also, insert
// ncclOp
#ifdef PADDLE_WITH_CUDA
  details::MultiDevSSAGraphBuilder builder(member_->places_, loss_var_name,
                                           params, member_->local_scopes_,
                                           member_->nccl_ctxs_.get());
#else
  details::MultiDevSSAGraphBuilder builder(member_->places_, loss_var_name,
                                           params, member_->local_scopes_);
#endif
  auto graph = builder.Build(main_program);

  member_->executor_.reset(new details::ThreadedSSAGraphExecutor(
      num_threads, use_event, member_->local_scopes_, places,
      std::move(graph)));

  // Step 3. Create vars in each scope;
  for (auto *scope : member_->local_scopes_) {
    for (auto *var : main_program.Block(0).AllVars()) {
      if (scope->FindVar(var->Name()) != nullptr) {
        continue;
      }

      InitializeVariable(scope->Var(var->Name()), var->GetType());
    }
  }
}

void ParallelExecutor::BCastParamsToGPUs(
    const ProgramDesc &startup_program) const {
#ifdef PADDLE_WITH_CUDA
  auto *main_scope = member_->local_scopes_[0];

  for (auto *var_desc : startup_program.Block(0).AllVars()) {
    if (var_desc->GetType() == proto::VarType::LOD_TENSOR) {
      auto &main_tensor =
          main_scope->FindVar(var_desc->Name())->Get<LoDTensor>();
      ncclDataType_t data_type = platform::ToNCCLDataType(main_tensor.type());
      auto &dims = main_tensor.dims();
      size_t numel = main_tensor.numel();

      platform::NCCLGroupGuard guard;

      for (size_t i = 0; i < member_->places_.size(); ++i) {
        auto place = member_->places_[i];
        void *buffer;
        if (i == 0) {
          buffer = const_cast<void *>(main_tensor.data<void>());
        } else {
          auto local_scope = member_->local_scopes_[i];
          auto *t = local_scope->Var(var_desc->Name())->GetMutable<LoDTensor>();
          t->Resize(dims);
          buffer = t->mutable_data(place, main_tensor.type());
        }

        auto &nccl_ctx = member_->nccl_ctxs_->at(place);
        platform::dynload::ncclBcast(buffer, numel, data_type, 0,
                                     nccl_ctx.comm_, nccl_ctx.stream());
      }
    }
    member_->nccl_ctxs_->WaitAll();
  }
#else
  PADDLE_THROW("Not compiled with CUDA");
#endif
}

void ParallelExecutor::Run(const std::vector<std::string> &fetch_tensors,
                           const std::string &fetched_var_name) {
  auto fetch_data = member_->executor_->Run(fetch_tensors);
  *member_->global_scope_->Var(fetched_var_name)->GetMutable<FeedFetchList>() =
      fetch_data;
}

}  // namespace framework
}  // namespace paddle
