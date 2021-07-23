// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/executor_cache.h"

namespace paddle {
namespace framework {
class ProgramDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {

namespace details {

static ExecutionStrategy GetExecutionStrategy(
    const ExecutorInfoCache::CacheKey &cache_key) {
  framework::ExecutionStrategy execution_strategy;

  switch (cache_key.device_type_) {
    case platform::DeviceType::CPU: {
      execution_strategy.num_threads_ = 2;
      break;
    }
    case platform::DeviceType::CUDA: {
      // NOTE: According experiments, one thread is faster in
      // most model training.
      execution_strategy.num_threads_ = 1;
      break;
    }
    case platform::DeviceType::XPU: {
      execution_strategy.num_threads_ = 1;
      break;
    }
    default:
      PADDLE_THROW(platform::errors::Unavailable("Unsupported Device type %d.",
                                                 cache_key.device_type_));
  }
  execution_strategy.use_device_ = cache_key.device_type_;

  return execution_strategy;
}

void AppendSkipDeletionVars(const std::vector<std::string> &append_vars,
                            std::vector<std::string> *all_vars) {
  for (auto &var : append_vars) {
    all_vars->emplace_back(var);
  }
}

/*
 * NOTE(Aurelius84): In ParallelExecutor, memory optimized pass will be applied.
 * To avoid eagerly deleting last alive variables which are necessary in
 * backward program, we firstly parse these variable names as
 * skip_eager_vars. While executing pe.run skip_eager_vars are used to
 * skip memory optimization.
 *
 * Variables satisfying the following rules are considered as skip_eager_var:
 *
 *   1. it is an output var in run_program_op
 *   2. it is an input var used in backward_op
 */
void ParseSafeEagerDeletionSkipVars(
    const ProgramDesc &program, int64_t forward_op_nums,
    const std::vector<std::string> &output_var_names,
    std::vector<std::string> *skip_eager_delete_vars) {
  auto all_ops = program.Block(0).AllOps();
  auto &op_info_map = OpInfoMap::Instance();
  // NOTE: skip `shape` and `fill_constant` op created by
  // fluid.backward.gradients, one forward output will generate one `shape`
  // and `fill_constant`.
  size_t backward_op_start_index =
      forward_op_nums + (output_var_names.size() * 2);

  // step 2: parse the necessary variable of backward op
  std::unordered_set<std::string> op_outputs;
  std::unordered_set<std::string> op_inputs;
  std::unordered_set<std::string> no_need_buffer_ins;

  for (auto i = backward_op_start_index; i < all_ops.size(); ++i) {
    framework::OpDesc *op = all_ops[i];
    // NOTE: skip NoNeedBufferVars of grad_op and GC its memory in advance.
    auto &op_info = op_info_map.Get(op->Type());
    auto &inferer = op_info.NoNeedBufferVarsInferer();
    no_need_buffer_ins.clear();
    if (inferer != nullptr) {
      no_need_buffer_ins =
          inferer(op->Inputs(), op->Outputs(), op->GetAttrMap());
    }
    for (auto &in_names : op->Inputs()) {
      if (no_need_buffer_ins.count(in_names.first) == 0) {
        for (auto &in_name : in_names.second) {
          op_inputs.emplace(in_name);
        }
      } else {
        VLOG(2) << op->Type() << " has no_need_buffer_in: " << in_names.first
                << " , skip it.";
      }
    }

    for (const std::string &out_arg_name : op->OutputArgumentNames()) {
      op_outputs.emplace(out_arg_name);
    }
  }
  // For the grad op input variables, if it is not output of grad_op, it may
  // be output of forward op and we should set the variables as skip_var to
  // prevent it being deleted when grad op is called multiple times.
  for (const std::string &var_name : op_inputs) {
    if (op_outputs.find(var_name) == op_outputs.end()) {
      VLOG(2) << "skip eager var: " << var_name;
      skip_eager_delete_vars->emplace_back(var_name);
    }
  }
  VLOG(3) << "Found skip_eager_delete_vars: " << skip_eager_delete_vars->size();
}

}  // namespace details

// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
ExecutorInfoCache &ExecutorInfoCache::Instance() {
  static ExecutorInfoCache g_exe_cache_info_map;
  return g_exe_cache_info_map;
}

void ExecutorInfoCache::Finalize() {
  // NOTE(Aurelius84): DO NOT perform finalize in destructor
  // to avoid problems caused by destructor order of static
  // object.
  info_map_.clear();
}

CacheInfo GetExecutorInfoFromCache(const ExecutorInfoCache::CacheKey &cache_key,
                                   framework::Scope *scope) {
  auto &cached_exe_info = framework::ExecutorInfoCache::Instance();

  if (!cached_exe_info.Has(cache_key)) {
    VLOG(1) << "create exe_info for " << cache_key.DebugString();

    // TODO(Aurelius84): Consider to use LRU algorithm to replace this.
    if (cached_exe_info.Size() > 4u /* max_cached_size*/) {
      VLOG(2) << "The cached info size has exceeded max_cached_size: 4, clear "
                 "all cache!";
      cached_exe_info.Finalize();
    }

    framework::BuildStrategy build_strategy;
    auto execution_strategy = details::GetExecutionStrategy(cache_key);

    auto graph = std::make_shared<framework::ir::Graph>(
        *cache_key.program_desc_, cache_key.start_op_index_,
        cache_key.end_op_index_);
    auto parallel_executor = std::make_shared<framework::ParallelExecutor>(
        cache_key.place_, scope, execution_strategy, build_strategy,
        graph.get());
    parallel_executor->PrepareVariables(scope);

    framework::ExecutorInfoCache::ValueType cache_val = {parallel_executor,
                                                         graph};
    cached_exe_info.Insert(cache_key, cache_val);

    bool is_new_created = true;
    return std::make_pair(parallel_executor, is_new_created);
  } else {
    VLOG(1) << "get exe_info from cache by: " << cache_key.DebugString();
    bool is_new_created = false;
    auto cache_val = cached_exe_info.GetMutable(cache_key);
    auto parallel_executor = cache_val.first;

    // update op_handle scope_map in pe->executor_->Graph
    std::unordered_map<Scope *, Scope *> scope_map = {
        {parallel_executor->GetLocalScopes().front(), scope}};
    parallel_executor->ResetOpHandleScopeMapOfGraphs(scope_map);
    // need to recreate tmp variables in new scope
    parallel_executor->PrepareVariables(scope);

    return std::make_pair(parallel_executor, is_new_created);
  }
}

}  // namespace framework
}  // namespace paddle
