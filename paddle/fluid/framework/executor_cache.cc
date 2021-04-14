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

static platform::DeviceType Place2DeviceType(const platform::Place &place) {
  if (platform::is_cpu_place(place)) {
    return platform::DeviceType::CPU;
  } else if (platform::is_gpu_place(place)) {
    return platform::DeviceType::CUDA;
  } else {
    return platform::DeviceType::XPU;
  }
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
std::vector<std::string> ParseSafeEagerDeletionSkipVars(
    const ProgramDesc &program, int64_t forward_op_nums,
    const std::vector<std::string> &output_var_names) {
  // step 1: all out_vars are skip_eager_var
  std::vector<std::string> skip_eager_vars(output_var_names);
  std::unordered_set<std::string> visited_vars;

  auto all_ops = program.Block(0).AllOps();
  // Between forward and backward op, there are many fill_constant_ops
  size_t backward_op_start_index =
      forward_op_nums + (output_var_names.size() * 2);

  // step 2: parse the necegssary variable of backward op
  // std::unordered_set<std::string> op_outputs;
  // std::unordered_set<std::string> op_inputs;
  for (auto i = backward_op_start_index; i < all_ops.size(); ++i) {
    framework::OpDesc *op = all_ops[i];
    for (const std::string &in_arg_name : op->InputArgumentNames()) {
      if (!visited_vars.count(in_arg_name)) {
        skip_eager_vars.emplace_back(in_arg_name);
        VLOG(3) << "skip var: " << in_arg_name;
        visited_vars.insert(in_arg_name);
      }
    }

    // for (const std::string &in_arg_name : op->InputArgumentNames()) {
    //   op_inputs.emplace(in_arg_name);
    // }
    // for (const std::string &out_arg_name : op->OutputArgumentNames()) {
    //   op_outputs.emplace(out_arg_name);
    // }
  }

  // For the grad op input variables, if it is not output of grad_op, it may
  // be output of forward op and we should set the variables as skip_var to
  // prevent it being deleted when grad op is called multiple times.
  // for (const std::string &var_name : op_inputs) {
  //   if (op_outputs.find(var_name) == op_outputs.end()) {
  //     skip_eager_vars.emplace_back(var_name);
  //   }
  // }

  VLOG(3) << "Found skip_eager_vars: " << skip_eager_vars.size();

  return skip_eager_vars;
}

}  // namespace details

// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
ExecutorInfoCache &ExecutorInfoCache::Instance() {
  static ExecutorInfoCache g_exe_cache_info_map;
  return g_exe_cache_info_map;
}

std::shared_ptr<framework::ParallelExecutor> GetExecutorInfoFromCache(
    const ProgramDesc *program, int64_t start_op_index, int64_t end_op_index,
    const platform::Place &place, framework::Scope *scope,
    const std::vector<std::string> &output_var_names, bool is_grad) {
  auto &cached_exe_info = framework::ExecutorInfoCache::Instance();
  auto device_type = details::Place2DeviceType(place);
  auto cache_key = framework::ExecutorInfoCache::KeyInfo(
      program, static_cast<int>(device_type), start_op_index, end_op_index,
      is_grad);

  if (!cached_exe_info.Has(cache_key)) {
    VLOG(1) << "create exe_info for program: " << program
            << " is_grad: " << is_grad;

    framework::BuildStrategy build_strategy;
    framework::ExecutionStrategy execution_strategy;
    if (platform::is_cpu_place(place)) {
      execution_strategy.use_device_ = platform::DeviceType::CPU;
      execution_strategy.num_threads_ = 2;
    } else if (platform::is_gpu_place(place)) {
      execution_strategy.use_device_ = platform::DeviceType::CUDA;
      execution_strategy.num_threads_ = 4;
    } else {
      execution_strategy.use_device_ = platform::DeviceType::XPU;
      execution_strategy.num_threads_ = 1;
    }

    auto graph = std::make_shared<framework::ir::Graph>(
        *program, start_op_index, end_op_index);
    auto parallel_executor = std::make_shared<framework::ParallelExecutor>(
        place, scope, execution_strategy, build_strategy, graph.get());
    parallel_executor->PrepareLocalExeScopes(scope);

    framework::ExecutorInfoCache::ValueType cache_val = {parallel_executor,
                                                         graph};
    cached_exe_info.Insert(cache_key, cache_val);

    return parallel_executor;
  } else {
    VLOG(1) << "get exe_info from cache by program: " << program
            << " is_grad: " << is_grad;
    auto cache_val = cached_exe_info.GetMutable(cache_key);
    auto parallel_executor = cache_val.first;
    // update op_handle scope_map in pe->executor_->Graph
    std::unordered_map<Scope *, Scope *> scope_map = {
        {parallel_executor->GetLocalScopes().front(), scope}};
    parallel_executor->ReSetOpScopeMapOfGraphs(scope_map);
    // need to re-create tmp variable in new scope
    parallel_executor->PrepareLocalExeScopes(scope);

    return parallel_executor;
  }
}

}  // namespace framework
}  // namespace paddle
