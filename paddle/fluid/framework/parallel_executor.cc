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
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/fetch_op_handle.h"
#include "paddle/fluid/framework/details/nccl_all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/details/ssa_graph.h"

namespace paddle {
namespace framework {

using details::ComputationOpHandle;
using details::DummyVarHandle;
using details::FetchOpHandle;
using details::NCCLAllReduceOpHandle;
using details::OpHandleBase;
using details::ScaleLossGradOpHandle;
using details::SSAGraph;
using details::VarHandle;
using details::VarHandleBase;

class SSAGraphBuilder {
 public:
  virtual ~SSAGraphBuilder() {}
  virtual void Build(const ProgramDesc &program, SSAGraph *graph) const = 0;

 protected:
  /**
   * We only handle write after read(WAR), since it should not have a write
   * after write in program. If there are write after write operators, we need
   * prune them.
   *
   * https://en.wikipedia.org/wiki/Hazard_(computer_architecture)#Write_after_read_(WAR)
   */
  static void PolishGraphToSupportDataHazards(SSAGraph *graph) {
    for (auto &var_map : graph->vars_) {
      for (auto &name_pair : var_map) {
        if (name_pair.second.size() <= 1) {
          return;
        }
        auto it_new = name_pair.second.rbegin();
        auto it_old = name_pair.second.rbegin();
        ++it_old;
        for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
          auto *write_op = it_new->second.generated_op_;
          auto &read_ops = it_old->second.pending_ops_;
          auto *ex_write_op = it_old->second.generated_op_;

          if (ex_write_op == nullptr) {  // Nobody write this var.
            continue;
          }

          for (auto *read_op : read_ops) {
            // Manually add a dependency var from read_op to write_op;
            if (read_op == write_op) {
              // Read Write is the same op.
              continue;
            }

            auto *dep_var = new DummyVarHandle();
            read_op->AddOutput(dep_var);
            write_op->AddInput(dep_var);
            graph->dep_vars_.emplace(dep_var);
          }
        }
      }
    }
  }

  static VarHandle *CreateOrGetLatestVarHandle(SSAGraph *graph,
                                               const std::string &each_var_name,
                                               const platform::Place &place,
                                               size_t place_offset) {
    auto &var_holders = graph->vars_[place_offset];
    auto &var_holder = var_holders[each_var_name];
    VarHandle *var = nullptr;
    if (var_holder.empty()) {
      auto &init_var = var_holder[0];
      init_var.place_ = place;
      init_var.name_ = each_var_name;
      init_var.generated_op_ = nullptr;
      init_var.version_ = 0;
      var = &init_var;
    } else {
      var = &var_holder.rbegin()->second;
    }
    return var;
  }

  static void CreateOpOutput(SSAGraph *graph, OpHandleBase *op_handle,
                             const std::string &each_var_name,
                             const platform::Place &place,
                             size_t place_offset) {
    auto &vars = graph->vars_[place_offset][each_var_name];
    size_t version = vars.size();
    auto &var = vars[version];
    var.version_ = version;
    var.name_ = each_var_name;
    var.place_ = place;
    op_handle->AddOutput(&var);
  }
};

class MultiDevSSAGraphBuilder : public SSAGraphBuilder {
 public:
  MultiDevSSAGraphBuilder(const std::vector<platform::Place> &places,
                          const std::string &loss_var_name,
                          const std::unordered_set<std::string> &params,
                          const std::vector<Scope *> &local_scopes,
                          platform::NCCLContextMap *nccl_ctxs)
      : loss_var_name_(loss_var_name),
        places_(places),
        local_scopes_(local_scopes),
        nccl_ctxs_(nccl_ctxs) {
    for (auto &p : params) {
      grad_names_.insert(GradVarName(p));
    }
  }

  void Build(const ProgramDesc &program, SSAGraph *graph) const override {
    SSAGraph &result = *graph;
    result.vars_.resize(places_.size());

    bool is_forwarding = true;
    for (auto *op : program.Block(0).AllOps()) {
      bool change_forward = false;
      if (!is_forwarding) {
        // FIXME(yy): Do not hard code like this
        if (op->OutputArgumentNames().size() == 1 &&
            op->OutputArgumentNames()[0] == GradVarName(loss_var_name_)) {
          continue;  // Drop fill 1. for backward coeff;
        }
      }

      for (size_t i = 0; i < places_.size(); ++i) {
        auto &p = places_[i];
        auto *s = local_scopes_[i];

        result.ops_.emplace_back(new ComputationOpHandle(*op, s, p));
        auto *op_handle = result.ops_.back().get();
        op_handle->dev_ctx_[p] = const_cast<platform::DeviceContext *>(
            platform::DeviceContextPool::Instance().Get(p));

        auto var_names = op->InputArgumentNames();

        for (auto &each_var_name : var_names) {
          VarHandle *var =
              CreateOrGetLatestVarHandle(&result, each_var_name, p, i);
          op_handle->AddInput(var);
        }
        var_names = op->OutputArgumentNames();

        for (auto &each_var_name : var_names) {
          CreateOpOutput(&result, op_handle, each_var_name, p, i);
        }

        if (is_forwarding) {
          if (var_names.size() == 1 && var_names[0] == loss_var_name_) {
            // Insert ScaleCost OpHandle
            op_handle = new ScaleLossGradOpHandle(local_scopes_.size(), s, p,
                                                  nccl_ctxs_->DevCtx(p));
            result.ops_.emplace_back(op_handle);

            // FIXME: Currently ScaleLossGradOp only use device_count as scale
            // factor. So it does not depend on any other operators.
            // VarHandle *loss = GetVarHandle(loss_var_name, place);
            // loss->pending_ops_.emplace_back(op_handle);
            // op_handle->inputs_.emplace_back(loss);

            CreateOpOutput(&result, op_handle, GradVarName(loss_var_name_), p,
                           i);
            change_forward = true;
          }
        }
      }

      if (change_forward) {
        is_forwarding = false;
      }

      if (!is_forwarding) {
        auto var_names = op->OutputArgumentNames();
        for (auto &og : var_names) {
          if (grad_names_.count(og) != 0) {  // is param grad
            // Insert NCCL AllReduce Op
            result.ops_.emplace_back(
                new NCCLAllReduceOpHandle(local_scopes_, places_, *nccl_ctxs_));
            auto *op_handle = result.ops_.back().get();

            for (size_t i = 0; i < places_.size(); ++i) {
              auto &p = places_[i];
              auto &vars = result.vars_[i][og];

              if (vars.empty()) {  // This device has no data. continue.
                continue;
              }
              auto *prev_grad = &vars[vars.size() - 1];
              op_handle->AddInput(prev_grad);

              auto &var = vars[vars.size()];
              var.place_ = p;
              var.name_ = og;
              var.version_ = vars.size() - 1;

              op_handle->AddOutput(&var);
            }
          }
        }
      }
    }

    /*
      Dependency graph has been constructed. However, there are still data
      harzaeds need to be handled.
     */
    PolishGraphToSupportDataHazards(&result);
  }

 private:
  std::string loss_var_name_;
  const std::vector<platform::Place> &places_;
  const std::vector<Scope *> &local_scopes_;
  platform::NCCLContextMap *nccl_ctxs_;

  std::unordered_set<std::string> grad_names_;
};

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

  SSAGraph graph_;

  // Use a simpler thread pool, might be faster.
  std::unique_ptr<ThreadPool> pool_;

  std::unique_ptr<platform::EnforceNotMet> exception_;

  void RunOp(
      bool use_event,
      std::unordered_map<VarHandleBase *, std::atomic<bool>> &pending_vars,
      OpHandleBase *op) {
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
  MultiDevSSAGraphBuilder builder(member_->places_, loss_var_name, params,
                                  member_->local_scopes_,
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
  std::unordered_map<VarHandleBase *, std::atomic<bool>> pending_vars;
  std::unordered_map<OpHandleBase *, size_t> pending_ops;
  std::vector<DummyVarHandle> dummy_vars;

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

  std::vector<OpHandleBase *> to_run;

  for (auto &op : member_->graph_.ops_) {
    if (op->inputs_.empty()) {  // Special case, Op has no input.
      to_run.emplace_back(op.get());
    } else {
      pending_ops.insert({op.get(), op->inputs_.size()});
    }
  }

  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : member_->graph_.vars_) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].push_back(&it->second.rbegin()->second);
      }
    }
  }

  std::vector<FetchOpHandle> fetch_ops;

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto &vars = fetched_vars[var_name];
    fetch_ops.emplace_back(&fetched_data, i, &member_->local_scopes_);
    FetchOpHandle *op = &fetch_ops.back();

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
    VarHandleBase *ready_var = nullptr;
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
