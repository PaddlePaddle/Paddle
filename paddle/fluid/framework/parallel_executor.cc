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
#include "lod_tensor.h"
#include "op_registry.h"
#include "paddle/fluid/framework/details/fetch_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/ssa_graph.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace framework {

class ParallelExecutorPrivate {
 public:
  explicit ParallelExecutorPrivate(size_t num_threads,
                                   const std::vector<platform::Place> &places)
      : places_(places),
        fetch_dev_ctxs_(places),
        pool_(num_threads <= 1 ? nullptr : new ThreadPool(num_threads)) {}

  std::vector<platform::Place> places_;
  platform::DeviceContextPool fetch_dev_ctxs_;
  std::vector<Scope *> local_scopes_;
  Scope *global_scope_;

  std::unique_ptr<platform::NCCLContextMap> nccl_ctxs_;

  details::SSAGraph graph_;

  // Use a simpler thread pool, might be faster.
  std::unique_ptr<ThreadPool> pool_;

  std::unique_ptr<platform::EnforceNotMet> exception_;

  void RunOp(bool use_event,
             std::unordered_map<details::VarHandleBase *, std::atomic<bool>>
                 &pending_vars,
             details::OpHandleBase *op) {
    std::vector<std::atomic<bool> *> *ready_buffer =
        new std::vector<std::atomic<bool> *>();
    for (auto *var : op->outputs_) {
      ready_buffer->emplace_back(&pending_vars[var]);
    }

    auto op_run = [ready_buffer, op, this, use_event] {
      try {
        VLOG(10) << op->DebugString();
        op->Run(use_event);
        for (auto *ready : *ready_buffer) {
          ready->store(true, std::memory_order_release);
        }
        delete ready_buffer;
      } catch (platform::EnforceNotMet ex) {
        exception_.reset(new platform::EnforceNotMet(ex));
      } catch (...) {
        LOG(FATAL) << "Unknown exception catched";
      }
    };
    if (pool_) {
      pool_->enqueue(op_run);
    } else {
      op_run();
    }
  }
};

ParallelExecutor::ParallelExecutor(
    size_t num_threads, const std::vector<platform::Place> &places,
    const std::unordered_set<std::string> &params,
    const ProgramDesc &startup_program, const ProgramDesc &main_program,
    const std::string &loss_var_name, Scope *scope)
    : member_(new ParallelExecutorPrivate(num_threads, places)) {
  member_->global_scope_ = scope;

  // Step 1. RunStartupProgram and Bcast the params to devs.
  Executor exe(places[0]);
  exe.Run(startup_program, scope, 0);
  // Create local scopes
  for (size_t i = 0; i < member_->places_.size(); ++i) {
    member_->local_scopes_.push_back(&scope->NewScope());
  }

  // Bcast Parameters to all GPUs
  BuildNCCLCommunicator();
  if (platform::is_gpu_place(places[0]) &&
      member_->local_scopes_.size() != 1) {  // Is CUDA
    BCastParamsToGPUs(startup_program);
  }
  // Startup Program has been run. All local scopes has correct parameters.

  // Step 2. Convert main_program to SSA form and dependency graph. Also, insert
  // ncclOp
  details::MultiDevSSAGraphBuilder builder(member_->places_, loss_var_name,
                                           params, member_->local_scopes_,
                                           member_->nccl_ctxs_.get());
  builder.Build(main_program, &member_->graph_);

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

void ParallelExecutor::BuildNCCLCommunicator() const {
#ifdef PADDLE_WITH_CUDA
  member_->nccl_ctxs_.reset(new platform::NCCLContextMap(member_->places_));
#endif
}

void ParallelExecutor::Run(const std::vector<std::string> &fetch_tensors,
                           const std::string &fetched_var_name) {
  bool use_event = true;
  FeedFetchList fetched_data(fetch_tensors.size());
  // Version --> VarHandle
  member_->exception_.reset();
  std::unordered_map<details::VarHandleBase *, std::atomic<bool>> pending_vars;
  std::unordered_map<details::OpHandleBase *, size_t> pending_ops;
  std::vector<details::DummyVarHandle> dummy_vars;

  for (auto &var_map : member_->graph_.vars_) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        pending_vars[&version_pair.second] =
            version_pair.second.generated_op_ == nullptr;
      }
    }
  }

  for (auto &var : member_->graph_.dep_vars_) {
    pending_vars[var.get()] = var->generated_op_ == nullptr;
  }

  std::vector<details::OpHandleBase *> to_run;

  for (auto &op : member_->graph_.ops_) {
    if (op->inputs_.empty()) {  // Special case, Op has no input.
      to_run.emplace_back(op.get());
    } else {
      pending_ops.insert({op.get(), op->inputs_.size()});
    }
  }

  std::unordered_map<std::string, std::vector<details::VarHandleBase *>>
      fetched_vars;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : member_->graph_.vars_) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].push_back(&it->second.rbegin()->second);
      }
    }
  }

  std::vector<details::FetchOpHandle> fetch_ops;

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto &vars = fetched_vars[var_name];
    fetch_ops.emplace_back(&fetched_data, i, &member_->local_scopes_);
    details::FetchOpHandle *op = &fetch_ops.back();

    // FIXME: Use new device context
    for (auto &p : member_->places_) {
      op->dev_ctx_[p] = member_->fetch_dev_ctxs_.Get(p);
    }

    for (auto *var : vars) {
      op->AddInput(var);
    }

    dummy_vars.emplace_back();
    auto *var = &dummy_vars.back();
    op->AddOutput(var);
    pending_vars[var] = false;

    pending_ops.insert({op, op->inputs_.size()});
  }

  for (auto *op : to_run) {
    member_->RunOp(use_event, pending_vars, op);
  }

  while (!pending_vars.empty()) {
    details::VarHandleBase *ready_var = nullptr;
    for (auto &pair : pending_vars) {
      if (pair.second.load(std::memory_order_acquire)) {
        ready_var = pair.first;
      }
    }
    if (ready_var == nullptr) {
      // FIXME use conditional var instead of busy wait.
      if (member_->exception_) {
        throw * member_->exception_;
      }
      continue;
    }
    pending_vars.erase(ready_var);
    to_run.clear();
    for (auto *op : ready_var->pending_ops_) {
      auto &deps = pending_ops[op];
      --deps;
      if (deps == 0) {
        to_run.emplace_back(op);
      }
    }
    for (auto *op : to_run) {
      pending_ops.erase(op);
      member_->RunOp(use_event, pending_vars, op);
    }
  }

  for (auto &fetch_op : fetch_ops) {
    fetch_op.WaitAndMergeCPUTensors();
  }

  *member_->global_scope_->Var(fetched_var_name)->GetMutable<FeedFetchList>() =
      fetched_data;
}

}  // namespace framework
}  // namespace paddle
