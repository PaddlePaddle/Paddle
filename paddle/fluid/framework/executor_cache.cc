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

#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {

namespace details {

static void AppendSkipDeletionVars(const std::vector<std::string> &append_vars,
                                   std::vector<std::string> *all_vars) {
  for (auto &var : append_vars) {
    all_vars->emplace_back(var);
  }
}

static void AppendSafeEagerDeletionSkipVars(
    const framework::ProgramDesc &program,
    std::vector<std::string> *skip_vars) {
  const framework::BlockDesc &block = program.Block(0);
  const std::vector<framework::OpDesc *> &all_ops = block.AllOps();

  std::unordered_set<std::string> grad_op_output;
  std::unordered_set<std::string> grad_op_input;
  for (const framework::OpDesc *op : all_ops) {
    int op_role = BOOST_GET_CONST(
        int, op->GetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName()));
    if ((op_role & static_cast<int>(framework::OpRole::kBackward)) == 0) {
      continue;
    }

    for (const std::string &in_arg_name : op->InputArgumentNames()) {
      grad_op_input.emplace(in_arg_name);
    }
    for (const std::string &out_arg_name : op->OutputArgumentNames()) {
      grad_op_output.emplace(out_arg_name);
    }
  }

  // For the grad op input variables, if it is not output of grad_op, it may
  // be output of forward op and we should set the variables as skip_var to
  // prevent it being deleted when grad op is called multiple times.
  for (const std::string &var_name : grad_op_input) {
    if (grad_op_output.find(var_name) == grad_op_output.end()) {
      skip_vars->emplace_back(var_name);
    }
  }
}
}  // namespace details

// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
ExecutorInfoCache &ExecutorInfoCache::Instance() {
  static ExecutorInfoCache g_exe_cache_info_map;
  return g_exe_cache_info_map;
}

std::shared_ptr<framework::ExecutorPrepareContext> GetExecutorInfoFromCache(
    const framework::Executor &exe, const framework::ExecutionContext &ctx,
    const std::vector<std::vector<std::string>> &ctx_output_names,
    bool is_grad) {
  auto *program = ctx.Attr<BlockDesc *>("global_block")->Program();

  auto &cached_exe_info = framework::ExecutorInfoCache::Instance();
  auto cache_key = framework::ExecutorInfoCache::KeyInfo(program, is_grad);

  if (!cached_exe_info.Has(cache_key)) {
    VLOG(1) << "create exe_info for program: " << program
            << " is_grad: " << is_grad;
    // skip delete vars
    std::vector<std::string> skip_vars;
    for (auto &output_names : ctx_output_names) {
      details::AppendSkipDeletionVars(output_names, &skip_vars);
    }
    if (is_grad) {
      details::AppendSafeEagerDeletionSkipVars(*program, &skip_vars);
    }

    VLOG(2) << "Prepare to skip " << skip_vars.size()
            << " var(s): " << string::join_strings(skip_vars, ' ');
    std::shared_ptr<framework::ExecutorPrepareContext> exe_ctx =
        std::move(exe.Prepare(*program, /*block_id=*/0, skip_vars));

    cached_exe_info.Insert(cache_key, exe_ctx);
    return exe_ctx;
  } else {
    VLOG(1) << "get exe_info from cache by program: " << program
            << " is_grad: " << is_grad;
    return cached_exe_info.Get(cache_key);
  }
}

}  // namespace framework
}  // namespace paddle
