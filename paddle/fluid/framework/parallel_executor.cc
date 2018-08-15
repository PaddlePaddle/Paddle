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

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#include "paddle/fluid/framework/details/multi_devices_graph_check_pass.h"
#include "paddle/fluid/framework/details/multi_devices_graph_print_pass.h"
#include "paddle/fluid/framework/details/scope_buffered_ssa_graph_executor.h"
#include "paddle/fluid/framework/details/threaded_ssa_graph_executor.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {

std::unique_ptr<ir::Graph> ApplyParallelExecutorPass(
    const ProgramDesc &main_program, const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &param_names,
    const std::vector<Scope *> &local_scopes, const bool use_cuda,
#ifdef PADDLE_WITH_CUDA
    const BuildStrategy &strategy, platform::NCCLContextMap *nccl_ctxs) {
#else
    const BuildStrategy &strategy) {
#endif
  // Convert the program to graph.
  std::unique_ptr<ir::Graph> graph(new ir::Graph(main_program));

  // Apply a graph viz pass to record a graph.
  if (!strategy.debug_graphviz_path_.empty()) {
    auto viz_pass = ir::PassRegistry::Instance().Get("graph_viz_pass");
    const std::string graph_path = string::Sprintf(
        "%s%s", strategy.debug_graphviz_path_.c_str(), "_original_graph");
    viz_pass->Set<std::string>("graph_viz_path", new std::string(graph_path));
    graph = viz_pass->Apply(std::move(graph));
  }

  // Convert graph to run on multi-devices.
  auto multi_devices_pass =
      ir::PassRegistry::Instance().Get("multi_devices_pass");
  multi_devices_pass->SetNotOwned<const std::vector<platform::Place>>("places",
                                                                      &places);
  multi_devices_pass->SetNotOwned<const std::string>("loss_var_name",
                                                     &loss_var_name);
  multi_devices_pass->SetNotOwned<const std::unordered_set<std::string>>(
      "params", &param_names);
  multi_devices_pass->SetNotOwned<const std::vector<Scope *>>("local_scopes",
                                                              &local_scopes);
  multi_devices_pass->SetNotOwned<const BuildStrategy>("strategy", &strategy);

#ifdef PADDLE_WITH_CUDA
  platform::NCCLContextMap *nctx = use_cuda ? nccl_ctxs : nullptr;
  multi_devices_pass->SetNotOwned<platform::NCCLContextMap>("nccl_ctxs", nctx);
#endif
  graph = multi_devices_pass->Apply(std::move(graph));

  // Apply a graph print pass to record a graph with device info.
  if (!strategy.debug_graphviz_path_.empty()) {
    auto multi_devices_print_pass =
        ir::PassRegistry::Instance().Get("multi_devices_print_pass");
    multi_devices_print_pass->SetNotOwned<const std::string>(
        "debug_graphviz_path", &strategy.debug_graphviz_path_);
    multi_devices_print_pass->Set<details::GraphvizSSAGraphPrinter>(
        "graph_printer", new details::GraphvizSSAGraphPrinter);
    graph = multi_devices_print_pass->Apply(std::move(graph));
  }

  // Verify that the graph is correct for multi-device executor.
  auto multi_devices_check_pass =
      ir::PassRegistry::Instance().Get("multi_devices_check_pass");
  graph = multi_devices_check_pass->Apply(std::move(graph));
  return graph;
}

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
  bool use_all_reduce_;
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
  member_->use_all_reduce_ =
      build_strategy.reduce_ == BuildStrategy::ReduceStrategy::kAllReduce;

  if (!member_->use_all_reduce_) {
    PADDLE_ENFORCE(places.size() > 1,
                   "If you set build_strategy.reduce with 'Reduce',"
                   "the number of places must be greater than 1.");
  }

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
    BCastParamsToDevices(bcast_vars);
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
#ifdef PADDLE_WITH_CUDA
  std::unique_ptr<ir::Graph> graph = ApplyParallelExecutorPass(
      main_program, member_->places_, loss_var_name, params,
      member_->local_scopes_, member_->use_cuda_, build_strategy,
      member_->nccl_ctxs_.get());
#else
  std::unique_ptr<ir::Graph> graph = ApplyParallelExecutorPass(
      main_program, member_->places_, loss_var_name, params,
      member_->local_scopes_, member_->use_cuda_, build_strategy);
#endif

  member_->executor_.reset(new details::ThreadedSSAGraphExecutor(
      exec_strategy, member_->local_scopes_, places, std::move(graph)));
  member_->executor_.reset(new details::ScopeBufferedSSAGraphExecutor(
      exec_strategy, member_->local_scopes_, std::move(var_infos),
      member_->places_, std::move(member_->executor_)));
}

void ParallelExecutor::BCastParamsToDevices(
    const std::unordered_set<std::string> &vars) const {
  // the initializing bcast, all vars would be bcast from device(0),
  // otherwise
  // bcast from the specified device.
  bool initializing = member_->executor_ ? false : true;
  for (auto &var : vars) {
    int var_dev_id = -1;
    if (member_->executor_) {
      auto &sharded_var_device =
          member_->executor_->Graph().Get<details::ShardedVarDevice>(
              details::kShardedVarDevice);
      if (sharded_var_device.find(var) != sharded_var_device.end()) {
        var_dev_id = sharded_var_device.at(var);
      }
    }

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
      for (size_t i = 0; i < member_->places_.size(); ++i) {
        if ((initializing && i == 0) ||
            (!initializing && static_cast<int>(i) == var_dev_id))
          continue;

        auto local_scope = member_->local_scopes_[i];
        auto *t = local_scope->Var(var)->GetMutable<LoDTensor>();

        // FIXME(zcd): LR_DECAY_COUNTER should not be shared. This is a hot fix.
        if (member_->use_all_reduce_ || member_->use_cuda_ ||
            var == "@LR_DECAY_COUNTER@") {
          t->Resize(dims);
          t->mutable_data(cpu, main_tensor.type());
          paddle::framework::TensorCopy(main_tensor, cpu, t);
        } else {
          t->ShareDataWith(main_tensor);
        }
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

USE_PASS(graph_viz_pass);
USE_PASS(multi_devices_pass);
USE_PASS(multi_devices_check_pass);
USE_PASS(multi_devices_print_pass);
