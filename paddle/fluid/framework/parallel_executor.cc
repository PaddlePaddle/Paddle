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

using details::DummyVarHandle;
using details::FetchOpHandle;
using details::OpHandleBase;
using details::SSAGraph;
using details::VarHandleBase;

class SSAGraphExecutor {
  DISABLE_COPY_AND_ASSIGN(SSAGraphExecutor);

 public:
  // Steal graph inside
  explicit SSAGraphExecutor(std::unique_ptr<SSAGraph> &&graph)
      : graph_(std::move(graph)) {}

  virtual ~SSAGraphExecutor() {}

  virtual FeedFetchList Run(const std::vector<std::string> &fetch_tensors) = 0;

 protected:
  std::unique_ptr<SSAGraph> graph_;
};

class ThreadedSSAGraphExecutor : public SSAGraphExecutor {
 public:
  ThreadedSSAGraphExecutor(size_t num_threads, bool use_event,
                           const std::vector<Scope *> &local_scopes,
                           const std::vector<platform::Place> &places,
                           std::unique_ptr<SSAGraph> &&graph)
      : SSAGraphExecutor(std::move(graph)),
        pool_(num_threads >= 2 ? new ::ThreadPool(num_threads) : nullptr),
        local_scopes_(local_scopes),
        places_(places),
        fetch_ctxs_(places),
        use_event_(use_event) {}

  // Run a SSAGraph by a thread pool
  // Use topological sort algorithm
  FeedFetchList Run(const std::vector<std::string> &fetch_tensors) override {
    std::unordered_map<OpHandleBase *, size_t> pending_ops;
    std::unordered_map<VarHandleBase *, std::atomic<bool>> pending_vars;
    std::unordered_set<OpHandleBase *> ready_ops;

    auto InsertPendingVar = [&pending_vars](VarHandleBase &var) {
      pending_vars[&var] = var.generated_op_ == nullptr;
    };

    auto InsertPendingOp = [&pending_ops](OpHandleBase &op_instance) {
      pending_ops.insert({&op_instance, op_instance.inputs_.size()});
    };

    // Transform SSAGraph to pending_ops & pending_vars
    for (auto &var_map : graph_->vars_) {
      for (auto &name_pair : var_map) {
        for (auto &version_pair : name_pair.second) {
          InsertPendingVar(version_pair.second);
        }
      }
    }
    for (auto &var : graph_->dep_vars_) {
      InsertPendingVar(*var);
    }

    for (auto &op : graph_->ops_) {
      if (op->inputs_.empty()) {  // Special case, Op has no input.
        ready_ops.insert(op.get());
      } else {
        InsertPendingOp(*op);
      }
    }

    // Step 2. Insert FetchOps
    std::vector<FetchOpHandle> fetch_ops;
    std::vector<DummyVarHandle> dummy_vars;
    FeedFetchList fetch_data(fetch_tensors.size());

    std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;

    for (auto &fetch_var_name : fetch_tensors) {
      for (auto &var_map : graph_->vars_) {
        auto it = var_map.find(fetch_var_name);
        if (it != var_map.end()) {
          fetched_vars[fetch_var_name].push_back(&it->second.rbegin()->second);
        }
      }
    }

    for (size_t i = 0; i < fetch_tensors.size(); ++i) {
      auto &var_name = fetch_tensors[i];
      auto &vars = fetched_vars[var_name];
      fetch_ops.emplace_back(&fetch_data, i, &local_scopes_);
      details::FetchOpHandle *op = &fetch_ops.back();

      // FIXME: Use new device context
      for (auto &p : places_) {
        op->dev_ctx_[p] = fetch_ctxs_.Get(p);
      }

      for (auto *var : vars) {
        op->AddInput(var);
      }

      dummy_vars.emplace_back();
      auto *var = &dummy_vars.back();
      var->generated_op_ = nullptr;
      op->AddOutput(var);
      InsertPendingVar(*var);
      InsertPendingOp(*op);
    }

    auto run_all_ready_ops = [&] {
      for (auto *op : ready_ops) {
        RunOp(pending_vars, op);
      }
      ready_ops.clear();
    };

    // Step 3. Execution
    while (!pending_vars.empty()) {
      // 1. Run All Ready ops
      run_all_ready_ops();

      // 2. Find ready variable
      VarHandleBase *ready_var = nullptr;
      for (auto &pair : pending_vars) {
        if (pair.second.load(std::memory_order_acquire)) {
          ready_var = pair.first;
          break;
        }
      }

      // if there is no variable ready
      if (ready_var == nullptr) {
        // FIXME use conditional var instead of busy wait.
        // if there is an exception, throw it
        if (exception_) {
          throw * exception_;
        }
        // keep waiting the ready variables
        continue;
      }

      // 3. Remove the dependency of ready_var.
      // Find the ready_ops after the ready_var.
      pending_vars.erase(ready_var);
      for (auto *op : ready_var->pending_ops_) {
        auto &deps = pending_ops[op];
        --deps;
        if (deps == 0) {
          ready_ops.insert(op);
        }
      }
      // Keep loop until all vars are ready.
    }

    // Wait FetchOps.
    for (auto &fetch_op : fetch_ops) {
      fetch_op.WaitAndMergeCPUTensors();
    }

    return fetch_data;
  }

  ~ThreadedSSAGraphExecutor() {}

 private:
  void RunOp(
      std::unordered_map<VarHandleBase *, std::atomic<bool>> &pending_vars,
      details::OpHandleBase *op) {
    std::vector<std::atomic<bool> *> *ready_buffer =
        new std::vector<std::atomic<bool> *>();
    for (auto *var : op->outputs_) {
      ready_buffer->emplace_back(&pending_vars[var]);
    }

    auto op_run = [ready_buffer, op, this] {
      try {
        VLOG(10) << op->DebugString();
        op->Run(use_event_);
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

 private:
  std::unique_ptr<::ThreadPool> pool_;
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
  platform::DeviceContextPool fetch_ctxs_;
  const bool use_event_;
  std::unique_ptr<platform::EnforceNotMet> exception_;
};

class ParallelExecutorPrivate {
 public:
  explicit ParallelExecutorPrivate(const std::vector<platform::Place> &places)
      : places_(places), fetch_dev_ctxs_(places) {}

  std::vector<platform::Place> places_;
  platform::DeviceContextPool fetch_dev_ctxs_;
  std::vector<Scope *> local_scopes_;
  Scope *global_scope_;

  std::unique_ptr<platform::NCCLContextMap> nccl_ctxs_;

  std::unique_ptr<SSAGraphExecutor> executor_;
};

ParallelExecutor::ParallelExecutor(
    size_t num_threads, const std::vector<platform::Place> &places,
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
  auto graph = builder.Build(main_program);

  member_->executor_.reset(new ThreadedSSAGraphExecutor(
      num_threads, true, member_->local_scopes_, places, std::move(graph)));

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
  auto fetch_data = member_->executor_->Run(fetch_tensors);
  *member_->global_scope_->Var(fetched_var_name)->GetMutable<FeedFetchList>() =
      fetch_data;
}

}  // namespace framework
}  // namespace paddle
