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

#include <string>
#include <tuple>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#include "paddle/fluid/framework/details/scope_buffered_ssa_graph_executor.h"
#include "paddle/fluid/framework/details/ssa_graph_builder_factory.h"
#include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"
#include "paddle/fluid/platform/profiler.h"

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
  bool own_local_scope_;
  bool use_cuda_;
};

std::vector<Scope *> &ParallelExecutor::GetLocalScopes() {
  return member_->local_scopes_;
}

ParallelExecutor::ParallelExecutor(
    const std::vector<platform::Place> &places,
    const std::unordered_set<std::string> &params,
    const std::unordered_set<std::string> &bcast_vars,
    const ProgramDesc &main_program, const std::string &loss_var_name,
    Scope *scope, const std::vector<Scope *> &local_scopes,
    const ExecutionStrategy &exec_strategy, const BuildStrategy &build_strategy,
    size_t num_trainers, size_t trainer_id)
    : member_(new ParallelExecutorPrivate(places)) {
  member_->global_scope_ = scope;
  member_->use_cuda_ = exec_strategy.use_cuda_;

  // Step 1. Bcast the params to devs.
  // Create local scopes
  if (local_scopes.empty()) {
    member_->own_local_scope_ = true;
    member_->local_scopes_.emplace_back(member_->global_scope_);
    for (size_t i = 1; i < member_->places_.size(); ++i) {
      member_->local_scopes_.emplace_back(&scope->NewScope());
    }
  } else {
    member_->own_local_scope_ = false;
    PADDLE_ENFORCE_EQ(member_->places_.size(), local_scopes.size());
    for (size_t i = 0; i < member_->places_.size(); ++i) {
      member_->local_scopes_.emplace_back(&local_scopes[i]->NewScope());
    }
  }

  if (member_->use_cuda_) {
// Bcast Parameters to all GPUs
#ifdef PADDLE_WITH_CUDA
    auto *nccl_id_var = scope->FindVar(NCCL_ID_VARNAME);
    ncclUniqueId *nccl_id = nullptr;
    if (nccl_id_var != nullptr) {
      nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();
    }
    member_->nccl_ctxs_.reset(new platform::NCCLContextMap(
        member_->places_, nccl_id, num_trainers, trainer_id));
#else
    PADDLE_THROW("Not compiled with CUDA");
#endif
  }

  if (member_->local_scopes_.size() != 1 && local_scopes.empty()) {
    BCastParamsToGPUs(bcast_vars);
  }
  // Startup Program has been run. All local scopes has correct parameters.

  // Step 2. Create vars in each scope;
  std::vector<details::VariableInfo> var_infos;
  for (auto *var : main_program.Block(0).AllVars()) {
    var_infos.emplace_back();
    var_infos.back().name_ = var->Name();
    var_infos.back().type_ = var->GetType();
    var_infos.back().persistable_ = var->Persistable();
  }

  // Step 3. Convert main_program to SSA form and dependency graph. Also, insert
  // ncclOp
  details::SSAGraphBuilderFactory builder_factory(
      member_->places_, loss_var_name, params, member_->local_scopes_,
      build_strategy);
  if (member_->use_cuda_) {
#ifdef PADDLE_WITH_CUDA
    builder_factory.SetNCCLContextMap(member_->nccl_ctxs_.get());
#else
    PADDLE_THROW("Not compiled with CUDA");
#endif
  }

  builder_ = builder_factory.Create();
  member_->executor_.reset(new details::ThreadedSSAGraphExecutor(
      exec_strategy, member_->local_scopes_, places,
      builder_->Build(main_program)));

  member_->executor_.reset(new details::ScopeBufferedSSAGraphExecutor(
      exec_strategy, member_->local_scopes_, std::move(var_infos),
      member_->places_, std::move(member_->executor_)));
}

void ParallelExecutor::BCastParamsToGPUs(
    const std::unordered_set<std::string> &vars) const {
  // the the initializing bcast, all vars would be bcast from device(0),
  // otherwise
  // bcast from the specified device.
  bool initializing = builder_.get() == nullptr ? true : false;

  for (auto &var : vars) {
    int var_dev_id =
        builder_.get() == nullptr ? -1 : builder_->GetVarDeviceID(var);
    if (!initializing && var_dev_id == -1) continue;

    framework::Variable *main_var = nullptr;
    if (initializing) {
      main_var = member_->local_scopes_[0]->FindVar(var);
    } else {
      main_var = member_->local_scopes_[var_dev_id]->FindVar(var);
    }

    if (main_var == nullptr || !main_var->IsType<LoDTensor>()) {
      continue;
    }

    auto &main_tensor = main_var->Get<LoDTensor>();
    auto &dims = main_tensor.dims();
    if (paddle::platform::is_gpu_place(main_tensor.place())) {
#ifdef PADDLE_WITH_CUDA
      std::vector<void *> buffers;
      size_t numel = main_tensor.numel();
      ncclDataType_t data_type = platform::ToNCCLDataType(main_tensor.type());
      for (size_t i = 0; i < member_->places_.size(); ++i) {
        auto place = member_->places_[i];
        void *buffer;

        if ((initializing && i == 0) ||
            (!initializing && static_cast<int>(i) == var_dev_id)) {
          buffer = const_cast<void *>(main_tensor.data<void>());
        } else {
          auto local_scope = member_->local_scopes_[i];
          auto *t = local_scope->Var(var)->GetMutable<LoDTensor>();
          t->Resize(dims);
          buffer = t->mutable_data(place, main_tensor.type());
        }
        buffers.push_back(buffer);
      }

      PADDLE_ENFORCE_EQ(member_->places_.size(), buffers.size(),
                        "variables' buffer size to bcast NOT equal to places");
      {
        platform::NCCLGroupGuard guard;
        for (size_t i = 0; i < member_->places_.size(); ++i) {
          auto &nccl_ctx = member_->nccl_ctxs_->at(member_->places_[i]);
          if (initializing) {
            platform::dynload::ncclBcast(buffers[i], numel, data_type, 0,
                                         nccl_ctx.comm_, nccl_ctx.stream());
          } else {
            if (var_dev_id >= 0) {
              platform::dynload::ncclBcast(buffers[i], numel, data_type,
                                           var_dev_id, nccl_ctx.comm_,
                                           nccl_ctx.stream());
            }
          }
        }
        member_->nccl_ctxs_->WaitAll();
      }

#else
      PADDLE_THROW("Not compiled with CUDA");
#endif
    } else {
      platform::CPUPlace cpu;
      for (size_t i = 1; i < member_->places_.size(); ++i) {
        auto local_scope = member_->local_scopes_[i];
        auto *t = local_scope->Var(var)->GetMutable<LoDTensor>();
        t->Resize(dims);
        t->mutable_data(cpu, main_tensor.type());
        paddle::framework::TensorCopy(main_tensor, cpu, t);
      }
    }
  }
}

void ParallelExecutor::Run(const std::vector<std::string> &fetch_tensors,
                           const std::string &fetched_var_name) {
  platform::RecordBlock b(0);
  auto fetch_data = member_->executor_->Run(fetch_tensors);
  *member_->global_scope_->Var(fetched_var_name)->GetMutable<FeedFetchList>() =
      fetch_data;
}

void ParallelExecutor::FeedTensorsIntoLocalScopes(
    const std::vector<std::unordered_map<std::string, LoDTensor>> &tensors) {
  PADDLE_ENFORCE_EQ(member_->local_scopes_.size(), tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &map = tensors[i];
    auto *scope = member_->local_scopes_[i];
    for (auto &pair : map) {
      auto *trg = scope->Var(pair.first)->GetMutable<LoDTensor>();
      trg->ShareDataWith(pair.second);
      trg->set_lod(pair.second.lod());
    }
  }
}

void ParallelExecutor::FeedAndSplitTensorIntoLocalScopes(
    const std::unordered_map<std::string, LoDTensor> &tensors) {
  for (auto pair : tensors) {
    auto lod_tensors = pair.second.SplitLoDTensor(member_->places_);
    PADDLE_ENFORCE_EQ(
        member_->places_.size(), lod_tensors.size(),
        "The number of samples of current batch is less than the count of "
        "devices, currently, it is not allowed. (%d vs %d)",
        member_->places_.size(), lod_tensors.size());
    for (size_t j = 0; j < member_->places_.size(); ++j) {
      // TODO(panxy0718): Do I need to delete this var?
      auto t =
          member_->local_scopes_[j]->Var(pair.first)->GetMutable<LoDTensor>();
      t->ShareDataWith(lod_tensors[j]);
      t->set_lod(lod_tensors[j].lod());
    }
  }
  for (auto &p : member_->places_) {
    platform::DeviceContextPool::Instance().Get(p)->Wait();
  }
}

ParallelExecutor::~ParallelExecutor() {
  if (member_->own_local_scope_) {
    for (size_t i = 1; i < member_->local_scopes_.size(); ++i) {
      member_->global_scope_->DeleteScope(member_->local_scopes_[i]);
    }
  }
}

}  // namespace framework
}  // namespace paddle
