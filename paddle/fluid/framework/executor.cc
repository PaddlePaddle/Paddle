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

#include "paddle/fluid/framework/executor.h"
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/trainer_factory.h"
#include "paddle/fluid/framework/transfer_scope_cache.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

#ifdef PADDLE_WITH_NGRAPH
#include "paddle/fluid/operators/ngraph/ngraph_engine.h"
DEFINE_bool(use_ngraph, false, "Use NGRAPH to run");
#endif

DECLARE_bool(benchmark);
DEFINE_bool(use_mkldnn, false, "Use MKLDNN to run");

namespace paddle {
namespace framework {

ExecutorPrepareContext::ExecutorPrepareContext(
    const framework::ProgramDesc& prog, size_t block_id)
    : prog_(prog), block_id_(block_id) {}

void ExecutorPrepareContext::PrepareUnusedVars(
    const std::vector<std::string>& keep_vars, bool force_disable_gc) {
  force_disable_gc_ = force_disable_gc;
  if (GetEagerDeletionThreshold() < 0 || force_disable_gc_) {
    return;
  }
  unused_vars_ = GetUnusedVars(prog_.Block(block_id_), ops_, keep_vars);
}

ExecutorPrepareContext::~ExecutorPrepareContext() {
  VLOG(5) << "destroy ExecutorPrepareContext";
}

Executor::Executor(const platform::Place& place) : place_(place) {
  cur_step_ = 0;
}

void Executor::Close() {
#ifdef PADDLE_WITH_DISTRIBUTE
  // TODO(typhoonzero): complete message will need to use real trainer_id,
  // except 0.
  auto client =
      paddle::operators::distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);
  client->SendComplete();
#endif
}

void Executor::CreateVariables(const ProgramDesc& pdesc, Scope* scope,
                               int block_id) {
  auto& global_block = pdesc.Block(block_id);

  const Scope* ancestor_scope = scope;
  while (ancestor_scope->parent()) {
    ancestor_scope = ancestor_scope->parent();
  }

  if (ancestor_scope != scope) {
    for (auto& var : global_block.AllVars()) {
      if (var->Name() == framework::kEmptyVarName) {
        continue;
      }

      if (var->Persistable()) {
        auto* ptr = const_cast<Scope*>(ancestor_scope)->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create Variable " << var->Name()
                << " global, which pointer is " << ptr;
      } else {
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create Variable " << var->Name()
                << " locally, which pointer is " << ptr;
      }
    }
  } else {
    for (auto& var : global_block.AllVars()) {
      auto* ptr = scope->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(3) << "Create variable " << var->Name() << ", which pointer is "
              << ptr;
    }
  }
}

void Executor::RunFromDataset(const ProgramDesc& main_program, Scope* scope,
                              Dataset* dataset,
                              const std::string& trainer_desc_str) {
  VLOG(3) << "Start to RunFromDataset in executor";
  TrainerDesc trainer_desc;
  google::protobuf::TextFormat::ParseFromString(trainer_desc_str,
                                                &trainer_desc);
  VLOG(3) << "Going to create trainer, trainer class is "
          << trainer_desc.class_name();
  std::shared_ptr<TrainerBase> trainer;
  trainer = TrainerFactory::CreateTrainer(trainer_desc.class_name());
  // initialize trainer
  VLOG(3) << "Going to initialize trainer";
  trainer->Initialize(trainer_desc, dataset);
  VLOG(3) << "Set root scope here";
  trainer->SetScope(scope);
  // prepare training environment and helper environment
  VLOG(3) << "Try to init train environment";
  trainer->InitTrainerEnv(main_program, place_);
  VLOG(3) << "Try to init other environment";
  trainer->InitOtherEnv(main_program);
  // training and finalize training
  VLOG(3) << "Trainer starts to run";
  trainer->Run();
  VLOG(3) << "Trainer going to finalize";
  trainer->Finalize();
  return;
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope, int block_id,
                   bool create_local_scope, bool create_vars,
                   const std::vector<std::string>& skip_ref_cnt_vars,
                   bool force_disable_gc) {
  platform::RecordBlock b(block_id);
  if (FLAGS_use_mkldnn) EnableMKLDNN(pdesc);
  if (open_ctx_cache_) {
    LOG(WARNING) << "open ctx cache is true";
  } else {
    LOG(WARNING) << "open ctx cache is false";
  }

  if (ctx_is_cached_) {
    LOG(WARNING) << "ctx is cached is true";
  } else {
    LOG(WARNING) << "ctx is cached is false";
  }

  if (open_ctx_cache_) {
    if (!ctx_is_cached_) {
      LOG(WARNING) << "going to run prepare ctx cache";
      PrepareCtxCache(pdesc, block_id, skip_ref_cnt_vars, force_disable_gc);
      ctx_is_cached_ = true;
    }
    RunPreparedContext(ctx_.get(), scope, create_local_scope, create_vars);
  } else {
    LOG(WARNING) << "going to run prepare ctx without cache";
    auto ctx = Prepare(pdesc, block_id, skip_ref_cnt_vars, force_disable_gc);
    RunPreparedContext(ctx.get(), scope, create_local_scope, create_vars);
  }
  // LOG(WARNING) << "run prepare context";
  // RunPreparedContext(ctx_ptr, scope, create_local_scope, create_vars);
  /*
  platform::RecordBlock b(block_id);
  if (FLAGS_use_mkldnn) EnableMKLDNN(pdesc);
  auto ctx = Prepare(pdesc, block_id, skip_ref_cnt_vars, force_disable_gc);
  RunPreparedContext(ctx.get(), scope, create_local_scope, create_vars);
  */
}

// Check whether the block already has feed operators and feed_holder.
// Return false if the block does not have any feed operators.
// If some feed operators have been prepended to the block, check that
// the info contained in these feed operators matches the feed_targets
// and feed_holder_name. Raise exception when any mismatch is found.
// Return true if the block has feed operators and holder of matching info.
static bool has_feed_operators(
    const BlockDesc& block,
    const std::map<std::string, const LoDTensor*>& feed_targets,
    const std::string& feed_holder_name) {
  size_t feed_count = 0;
  for (auto* op : block.AllOps()) {
    if (op->Type() == kFeedOpType) {
      feed_count++;
      // The input variable's name of feed_op should be feed_holder_name.
      PADDLE_ENFORCE_EQ(op->Input("X")[0], feed_holder_name,
                        "Input to feed op should be '%s'", feed_holder_name);
      std::string feed_target_name = op->Output("Out")[0];
      PADDLE_ENFORCE(
          feed_targets.find(feed_target_name) != feed_targets.end(),
          "Feed operator output name '%s' cannot be found in 'feed_targets'",
          feed_target_name);
    }
  }

  if (feed_count > 0) {
    PADDLE_ENFORCE_EQ(
        feed_count, feed_targets.size(),
        "The number of feed operators should match 'feed_targets'");

    if (!feed_holder_name.empty()) {
      // When feed operator are present, so should be feed_holder.
      auto var = block.FindVar(feed_holder_name);
      PADDLE_ENFORCE_NOT_NULL(var, "Block should already have a '%s' variable",
                              feed_holder_name);
      PADDLE_ENFORCE_EQ(var->GetType(), proto::VarType::FEED_MINIBATCH,
                        "'%s' variable should be 'FEED_MINIBATCH' type",
                        feed_holder_name);
    }
  }

  return feed_count > 0;
}

// Check whether the block already has fetch operators and fetch_holder.
// Return false if the block does not have any fetch operators.
// If some fetch operators have been appended to the block, check that
// the info contained in these fetch operators matches the fetch_targets
// and fetch_holder_name. Raise exception when any mismatch is found.
// Return true if the block has fetch operators and holder of matching info.
static bool has_fetch_operators(
    const BlockDesc& block,
    const std::map<std::string, LoDTensor*>& fetch_targets,
    const std::string& fetch_holder_name) {
  size_t fetch_count = 0;
  for (auto* op : block.AllOps()) {
    if (op->Type() == kFetchOpType) {
      fetch_count++;
      // The output variable's name of fetch_op should be fetch_holder_name.
      PADDLE_ENFORCE_EQ(op->Output("Out")[0], fetch_holder_name,
                        "Output of fetch op should be '%s'", fetch_holder_name);
      std::string fetch_target_name = op->Input("X")[0];
      PADDLE_ENFORCE(
          fetch_targets.find(fetch_target_name) != fetch_targets.end(),
          "Fetch operator input name '%s' cannot be found in 'fetch_targets'",
          fetch_target_name);
    }
  }

  if (fetch_count > 0) {
    PADDLE_ENFORCE_EQ(
        fetch_count, fetch_targets.size(),
        "The number of fetch operators should match 'fetch_targets'");

    if (!fetch_holder_name.empty()) {
      // When fetch operator are present, so should be fetch_holder.
      auto var = block.FindVar(fetch_holder_name);
      PADDLE_ENFORCE_NOT_NULL(var, "Block should already have a '%s' variable",
                              fetch_holder_name);
      PADDLE_ENFORCE_EQ(var->GetType(), proto::VarType::FETCH_LIST,
                        "'%s' variable should be 'FETCH_LIST' type",
                        fetch_holder_name);
    }
  }

  return fetch_count > 0;
}

void Executor::PrepareCtxCache(
    const ProgramDesc& program, int block_id,
    const std::vector<std::string>& skip_ref_cnt_vars, bool force_disable_gc) {
  ctx_.reset(new ExecutorPrepareContext(program, block_id));
  auto& block = program.Block(block_id);
  for (auto& op_desc : block.AllOps()) {
    ctx_->ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
#ifdef PADDLE_WITH_NGRAPH
  if (FLAGS_use_ngraph) {
    paddle::operators::NgraphEngine::FuseNgraphOps(
        ctx_->prog_.Block(ctx_->block_id_), &ctx_->ops_);
  }
#endif
  ctx_->PrepareUnusedVars(skip_ref_cnt_vars, force_disable_gc);
}

std::unique_ptr<ExecutorPrepareContext> Executor::Prepare(
    const ProgramDesc& program, int block_id,
    const std::vector<std::string>& skip_ref_cnt_vars, bool force_disable_gc) {
  std::unique_ptr<ExecutorPrepareContext> ctx(
      new ExecutorPrepareContext(program, block_id));
  PADDLE_ENFORCE_LT(static_cast<size_t>(block_id), program.Size());
  auto& block = program.Block(block_id);
  for (auto& op_desc : block.AllOps()) {
    ctx->ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
#ifdef PADDLE_WITH_NGRAPH
  if (FLAGS_use_ngraph) {
    paddle::operators::NgraphEngine::FuseNgraphOps(
        ctx->prog_.Block(ctx->block_id_), &ctx->ops_);
  }
#endif
  ctx->PrepareUnusedVars(skip_ref_cnt_vars, force_disable_gc);
  return ctx;
}

std::vector<std::shared_ptr<ExecutorPrepareContext>> Executor::Prepare(
    const ProgramDesc& program, const std::vector<int>& block_ids,
    const std::vector<std::vector<std::string>>& skip_ref_cnt_vars,
    bool force_disable_gc) {
  PADDLE_ENFORCE(
      skip_ref_cnt_vars.empty() || skip_ref_cnt_vars.size() == block_ids.size(),
      "skip_ref_cnt_vars should be either empty or equals to block number %d",
      block_ids.size());
  std::vector<std::shared_ptr<ExecutorPrepareContext>> result;
  size_t idx = 0;
  for (auto& bid : block_ids) {
    PADDLE_ENFORCE_LT(static_cast<size_t>(bid), program.Size());
    auto* ctx = new ExecutorPrepareContext(program, bid);
    auto& block = program.Block(bid);
    for (auto& op_desc : block.AllOps()) {
      ctx->ops_.push_back(OpRegistry::CreateOp(*op_desc));
    }
    if (skip_ref_cnt_vars.empty()) {
      ctx->PrepareUnusedVars(std::vector<std::string>(), force_disable_gc);
    } else {
      ctx->PrepareUnusedVars(skip_ref_cnt_vars[idx], force_disable_gc);
    }
    result.push_back(std::shared_ptr<ExecutorPrepareContext>(ctx));
    ++idx;
  }
  return result;
}

void Executor::RunPreparedContext(ExecutorPrepareContext* ctx, Scope* scope,
                                  bool create_local_scope, bool create_vars,
                                  bool keep_kids) {
  PADDLE_ENFORCE_NOT_NULL(scope);
  Scope* local_scope = scope;
  LOG(WARNING) << "going to create scope";
  if (create_vars) {
    if (create_local_scope) {
      local_scope = &scope->NewScope();
    }
    CreateVariables(ctx->prog_, local_scope, ctx->block_id_);
  }

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  LOG(WARNING) << "going to create gc";
  // FIXME(zjl): recurrent_op is rather complex, we would
  // disable gc forcely in recurrent_op
  if (!ctx->force_disable_gc_ && max_memory_size >= 0) {
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(place_)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            boost::get<platform::CUDAPlace>(place_), max_memory_size));
      } else {
        gc.reset(new DefaultStreamGarbageCollector(
            boost::get<platform::CUDAPlace>(place_), max_memory_size));
      }
    } else if (platform::is_cpu_place(place_)) {
#endif
      gc.reset(new CPUGarbageCollector(boost::get<platform::CPUPlace>(place_),
                                       max_memory_size));
#ifdef PADDLE_WITH_CUDA
    }
#endif
    LOG(WARNING) << "going to enable gc";
    // If gc is enabled and block size > 1
    if (gc && ctx->prog_.Size() > 1) {
      operators::PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(ctx->block_id_,
                                                                 ctx->ops_);
    }
  }

  LOG(WARNING) << "xx";
  for (auto& op : ctx->ops_) {
    op->Run(*local_scope, place_);
    if (gc) {
      DeleteUnusedTensors(*local_scope, op.get(), ctx->unused_vars_, gc.get());
    }
  }

  LOG(WARNING) << "going to wait instance";
  platform::DeviceContextPool::Instance().Get(place_)->Wait();

  if (local_scope != scope) {
    scope->DeleteScope(local_scope);
  } else {
    if (!keep_kids) {
      // By default, we should delete all kid scopes after run executor because
      // some operators may create local scope when running, such as while_op.
      // But when while_op also create a local executor to run it's sub block,
      // the sub scopes it created should not be dropped immediately, because
      // while_grad_op will use some variables created during while_op run, so
      // we need to keep the kids and wait for the outer executor to drop them.
      scope->DropKids();
    }
  }
}

void Executor::RunPreparedContext(
    ExecutorPrepareContext* ctx, Scope* scope,
    std::map<std::string, const LoDTensor*>* feed_targets,
    std::map<std::string, LoDTensor*>* fetch_targets, bool create_local_scope,
    bool create_vars, const std::string& feed_holder_name,
    const std::string& fetch_holder_name) {
  auto& global_block = ctx->prog_.Block(ctx->block_id_);

  PADDLE_ENFORCE(
      has_feed_operators(global_block, *feed_targets, feed_holder_name),
      "Program in ExecutorPrepareContext should has feed_ops.");
  PADDLE_ENFORCE(
      has_fetch_operators(global_block, *fetch_targets, fetch_holder_name),
      "Program in the prepared context should has fetch_ops.");

  // map the data of feed_targets to feed_holder
  for (auto* op : global_block.AllOps()) {
    if (op->Type() == kFeedOpType) {
      std::string feed_target_name = op->Output("Out")[0];
      int idx = boost::get<int>(op->GetAttr("col"));
      SetFeedVariable(scope, *(*feed_targets)[feed_target_name],
                      feed_holder_name, idx);
    }
  }

  RunPreparedContext(ctx, scope, create_local_scope, create_vars);

  // obtain the data of fetch_targets from fetch_holder
  for (auto* op : global_block.AllOps()) {
    if (op->Type() == kFetchOpType) {
      std::string fetch_target_name = op->Input("X")[0];
      int idx = boost::get<int>(op->GetAttr("col"));
      *(*fetch_targets)[fetch_target_name] =
          GetFetchVariable(*scope, fetch_holder_name, idx);
    }
  }
}

void Executor::EnableMKLDNN(const ProgramDesc& program) {
#ifdef PADDLE_WITH_MKLDNN
  VLOG(3) << "use_mkldnn=True";
  for (size_t bid = 0; bid < program.Size(); ++bid) {
    auto* block = const_cast<ProgramDesc&>(program).MutableBlock(bid);
    for (auto* op : block->AllOps()) {
      if (op->HasAttr("use_mkldnn")) {
        op->SetAttr("use_mkldnn", true);
      }
    }
  }
#else
  LOG(WARNING)
      << "'MKLDNN' is not supported, Please re-compile with WITH_MKLDNN option";
#endif
}
}  // namespace framework
}  // namespace paddle
