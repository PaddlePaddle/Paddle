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

#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/ir/transforms/inplace_pass.h"
#include "paddle/fluid/ir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/value.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"

namespace paddle {
namespace framework {
class ProgramDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {

namespace details {

static ExecutionStrategy GetExecutionStrategy(const platform::Place &place) {
  framework::ExecutionStrategy execution_strategy;

  auto device_type = platform::Place2DeviceType(place);
  switch (device_type) {
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
    case platform::DeviceType::IPU: {
      execution_strategy.num_threads_ = 1;
      break;
    }
    case platform::DeviceType::CUSTOM_DEVICE: {
      execution_strategy.num_threads_ = 1;
      break;
    }
    default:
      PADDLE_THROW(platform::errors::Unavailable("Unsupported Device type %d.",
                                                 device_type));
  }
  execution_strategy.use_device_ = device_type;

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
    const ProgramDesc &program,
    int64_t forward_op_nums,
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

void AppendSkipDeletionVars(const std::vector<std::string> &append_vars,
                            std::set<std::string> *all_vars) {
  for (auto &var : append_vars) {
    all_vars->insert(var);
  }
}

std::set<std::string> ParseSafeEagerDeletionSkipVarsSet(
    const ProgramDesc &backward_program, bool skip_no_need_buffer) {
  std::set<std::string> skip_eager_delete_vars;
  auto backward_ops = backward_program.Block(0).AllOps();
  auto &op_info_map = OpInfoMap::Instance();
  std::unordered_set<std::string> op_outputs;
  std::unordered_set<std::string> op_inputs;
  std::unordered_set<std::string> no_need_buffer_ins;
  for (auto op : backward_ops) {
    VLOG(4) << "parse op type: " << op->Type();
    if (op->Type() == "share_buffer") {
      VLOG(1) << "skip share_buffer op";
      continue;
    }
    // NOTE: skip NoNeedBufferVars of grad_op and GC its memory in advance.
    auto &op_info = op_info_map.Get(op->Type());
    auto &inferer = op_info.NoNeedBufferVarsInferer();
    no_need_buffer_ins.clear();
    // TODO(Aurelius84): Need remove skip_no_need_buffer after cinn fix this
    // problem.
    if (inferer != nullptr && !skip_no_need_buffer) {
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
  for (const std::string &var_name : op_inputs) {
    VLOG(4) << "parse op.input: " << var_name;
    if (op_outputs.find(var_name) == op_outputs.end()) {
      VLOG(1) << "skip eager var: " << var_name;
      skip_eager_delete_vars.insert(var_name);
    }
  }
  VLOG(1) << "Found skip_eager_delete_vars: " << skip_eager_delete_vars.size();
  return skip_eager_delete_vars;
}
}  // namespace details

// C++11 removes the need for manual locking. Concurrent execution shall wait if
// a static local variable is already being initialized.
// https://stackoverflow.com/questions/11711920/how-to-implement-multithread-safe-singleton-in-c11-without-using-mutex
ExecutorInfoCache &ExecutorInfoCache::Instance() {
  static ExecutorInfoCache g_exe_cache_info_map;
  return g_exe_cache_info_map;
}

static PEAndGraphPair CreateExecutorInfo(
    const ProgramDesc &program_desc,
    const platform::Place &place,
    int64_t start_op_index,
    int64_t end_op_index,
    framework::Scope *scope,
    const details::BuildStrategy &build_strategy) {
  auto execution_strategy = details::GetExecutionStrategy(place);
  auto graph = std::make_shared<framework::ir::Graph>(
      program_desc, start_op_index, end_op_index);
  auto parallel_executor = std::make_shared<framework::ParallelExecutor>(
      place, scope, execution_strategy, build_strategy, graph.get());
  parallel_executor->PrepareVariables(scope);
  return std::make_pair(parallel_executor, graph);
}

PEAndGraphPair CreateFixOrderExecutorInfo(const ProgramDesc &program_desc,
                                          const platform::Place &place,
                                          int64_t start_op_index,
                                          int64_t end_op_index,
                                          framework::Scope *scope) {
  details::BuildStrategy build_strategy;
  build_strategy.fix_op_run_order_ = true;
  auto pe_and_graph = CreateExecutorInfo(
      program_desc, place, start_op_index, end_op_index, scope, build_strategy);
  return pe_and_graph;
}

CacheInfo GetExecutorInfoFromCache(const ProgramDesc &program_desc,
                                   const platform::Place &place,
                                   int64_t start_op_index,
                                   int64_t end_op_index,
                                   bool is_grad,
                                   int64_t program_id,
                                   framework::Scope *scope) {
  auto &cached_exe_info = framework::ExecutorInfoCache::Instance();

  if (!cached_exe_info.Has(program_id, is_grad)) {
    // TODO(Aurelius84): Consider to use LRU algorithm to replace this.
    if (cached_exe_info.Size() > 4u /* max_cached_size*/) {
      VLOG(2) << "The cached info size has exceeded max_cached_size: 4, clear "
                 "all cache!";
      cached_exe_info.Finalize();
    }

    VLOG(1) << "create exe_info for " << program_id << " is_grad: " << is_grad;
    auto &build_strategy = cached_exe_info.GetBuildStrategy(program_id);

    // 2. Construct Graph and ParallelExecutor.
    auto pe_and_graph = CreateExecutorInfo(program_desc,
                                           place,
                                           start_op_index,
                                           end_op_index,
                                           scope,
                                           build_strategy);

    // 3. Insert value into cached map.
    auto &cached_value = cached_exe_info.GetMutable(program_id, is_grad);
    cached_value.executor_ = pe_and_graph.first;
    cached_value.graph_ = pe_and_graph.second;
    return std::make_pair(pe_and_graph.first, true);
  } else {
    VLOG(1) << "get exe_info from cache by: " << program_id
            << " is_grad: " << is_grad;
    auto &cached_value = cached_exe_info.GetMutable(program_id, is_grad);

    auto &parallel_executor = cached_value.executor_;
    // update op_handle scope_map in pe->executor_->Graph
    std::unordered_map<Scope *, Scope *> scope_map = {
        {parallel_executor->GetLocalScopes().front(), scope}};
    parallel_executor->ResetOpHandleScopeMapOfGraphs(scope_map);
    // need to recreate tmp variables in new scope
    parallel_executor->PrepareVariables(scope);

    return std::make_pair(parallel_executor, false);
  }
}

InterpreterCoreInfoCache &InterpreterCoreInfoCache::Instance() {
  static InterpreterCoreInfoCache g_info_cache;
  return g_info_cache;
}

std::shared_ptr<InterpreterCore> CreateProgramInterpreterCoreInfoToCache(
    const ProgramDesc &program_desc,
    const platform::Place &place,
    bool is_grad,
    int64_t program_id,
    framework::Scope *scope) {
  auto &interpretercore_info_cache =
      framework::InterpreterCoreInfoCache::Instance();
  if (interpretercore_info_cache.Size() > 10u /* max_cached_size*/) {
    VLOG(2) << "The cached info size has exceeded max_cached_size: 4, clear "
               "all cache!";
    interpretercore_info_cache.Finalize();
  }
  interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_jit = true;

  std::shared_ptr<InterpreterCore> core = nullptr;

  core.reset(new InterpreterCore(
      place, program_desc.Block(0), scope, execution_config));

  auto &cached_value =
      interpretercore_info_cache.GetMutable(program_id, scope, is_grad);
  cached_value.core_ = core;
  return core;
}

std::shared_ptr<InterpreterCore> CreateNewIRInterpreterCoreInfoToCache(
    std::unique_ptr<::ir::Program> ir_program,
    const platform::Place &place,
    bool is_grad,
    int64_t program_id,
    framework::Scope *scope) {
  auto &interpretercore_info_cache =
      framework::InterpreterCoreInfoCache::Instance();
  if (interpretercore_info_cache.Size() > 10u /* max_cached_size*/) {
    VLOG(2) << "The cached info size has exceeded max_cached_size: 4, clear "
               "all cache!";
    interpretercore_info_cache.Finalize();
  }
  interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_jit = true;

  std::shared_ptr<InterpreterCore> core = nullptr;

  core.reset(new InterpreterCore(
      place, {}, std::move(ir_program), scope, execution_config));

  auto &cached_value =
      interpretercore_info_cache.GetMutable(program_id, scope, is_grad);
  cached_value.core_ = core;
  return core;
}

std::unique_ptr<::ir::Program> ConstructFowardIrProgram(
    const paddle::framework::BlockDesc *forward_global_block,
    const paddle::framework::BlockDesc *backward_global_block,
    const std::vector<std::string> output_names,
    const std::vector<paddle::Tensor> &x,
    const std::vector<paddle::Tensor> &params) {
  auto ir_ctx = ::ir::IrContext::Instance();
  auto program = std::make_unique<::ir::Program>(ir_ctx);

  std::set<std::string> set_output_names;
  auto local_program =
      paddle::framework::ProgramDesc(*(forward_global_block->Program()));

  for (auto op_desc : local_program.Block(0).AllOps()) {
    for (const auto &n : op_desc->Outputs()) {
      const auto &input_var_names = n.second;
      for (const auto &var_name : input_var_names) {
        set_output_names.insert(var_name);
      }
    }
  }

  // add data op to program
  auto *block = local_program.MutableBlock(0);
  for (auto &in_t : x) {
    auto name = in_t.name();
    if (block->FindVarRecursive(name) == nullptr) {
      continue;
    }
    auto place = in_t.place().GetType();

    auto op_desc = block->PrependOp();
    op_desc->SetType("data");
    op_desc->SetAttr("shape", std::vector<int64_t>());
    // TODO(phlrain) : using tensor dtype
    op_desc->SetAttr("dtype", 0);
    op_desc->SetAttr("place", static_cast<int>(place));
    op_desc->SetAttr("name", name);
    op_desc->SetOutput("out", {name});
  }

  std::set<std::string> input_param_names;
  for (auto &param : params) {
    auto &name = param.name();
    auto place = param.place().GetType();

    auto op_desc = local_program.MutableBlock(0)->PrependOp();
    op_desc->SetType("data");
    op_desc->SetAttr("shape", std::vector<int64_t>());
    // TODO(phlrain) : using tensor dtype
    op_desc->SetAttr("dtype", 0);
    op_desc->SetAttr("place", static_cast<int>(place));
    op_desc->SetAttr("name", name);
    op_desc->SetOutput("out", {name});

    input_param_names.insert(name);
  }

  std::set<std::string> set_parameter_names;
  for (auto op_desc : backward_global_block->Program()->Block(0).AllOps()) {
    for (const auto &n : op_desc->Inputs()) {
      const auto &input_var_names = n.second;
      for (const auto &var_name : input_var_names) {
        set_parameter_names.insert(var_name);
      }
    }
  }

  for (auto &t : output_names) {
    set_parameter_names.insert(t);
  }

  for (auto &name : set_parameter_names) {
    if (!set_output_names.count(name)) {
      continue;
    }

    if (input_param_names.count(name)) {
      continue;
    }

    auto op_desc = local_program.MutableBlock(0)->AppendOp();
    op_desc->SetType("shadow_output");
    op_desc->SetAttr("name", name);
    op_desc->SetInput("x", {name});
    op_desc->SetOutput("out", {"@EMPTY@"});
  }

  paddle::translator::ProgramTranslator program_translator(&local_program,
                                                           program.get());

  program_translator.Translate();

  auto ir_res = paddle::dialect::PdOpLowerToKernelPass(program.get());

  ::ir::PassManager pm(::ir::IrContext::Instance(), 3);
  pm.AddPass(::ir::CreateInplacePass());
  pm.Run(ir_res.get());

  return ir_res;
}

std::unique_ptr<::ir::Program> ConstructBackwardIrProgram(
    const paddle::framework::BlockDesc *backward_global_block,
    const std::vector<paddle::Tensor> &out_grad,
    const std::vector<paddle::Tensor *> &x_grad,
    const std::vector<paddle::Tensor *> &params_grad,
    const paddle::framework::Scope *scope) {
  auto ir_ctx = ::ir::IrContext::Instance();
  auto program = std::make_unique<::ir::Program>(ir_ctx);

  auto local_program =
      paddle::framework::ProgramDesc(*(backward_global_block->Program()));

  // get feed with data
  std::set<std::string> set_parameter_names;
  for (auto op_desc : backward_global_block->Program()->Block(0).AllOps()) {
    for (const auto &n : op_desc->Inputs()) {
      const auto &input_var_names = n.second;
      for (const auto &var_name : input_var_names) {
        set_parameter_names.insert(var_name);
      }
    }
  }

  for (auto &var_name : set_parameter_names) {
    if (scope->FindVar(var_name)) {
      auto tensor = scope->FindVar(var_name)->Get<phi::DenseTensor>();
      phi::AllocationType place(phi::AllocationType::UNDEFINED);
      if (tensor.initialized()) {
        place = tensor.place().GetType();
      }

      if (var_name == "@EMPTY@") {
        continue;
      }
      auto op_desc = local_program.MutableBlock(0)->PrependOp();
      op_desc->SetType("data");
      op_desc->SetAttr("shape", std::vector<int64_t>());
      // TODO(phlrain) : using tensor dtype
      op_desc->SetAttr("dtype", 0);
      op_desc->SetAttr("place", static_cast<int>(place));
      op_desc->SetAttr("name", var_name);
      op_desc->SetOutput("out", {var_name});
    }
  }

  std::vector<std::string> param_grad_names;
  for (auto &p_g : params_grad) {
    param_grad_names.push_back(p_g->name());
  }

  for (auto &t : x_grad) {
    param_grad_names.push_back(t->name());
  }
  for (auto &name : param_grad_names) {
    if (name == "@EMPTY@") {
      continue;
    }
    auto op_desc = local_program.MutableBlock(0)->AppendOp();
    op_desc->SetType("shadow_output");
    op_desc->SetAttr("name", name);
    op_desc->SetInput("x", {name});
    op_desc->SetOutput("out", {"@EMPTY@"});
  }

  paddle::translator::ProgramTranslator program_translator(&local_program,
                                                           program.get());
  program_translator.Translate();

  auto res = paddle::dialect::PdOpLowerToKernelPass(program.get());

  ::ir::PassManager pm(::ir::IrContext::Instance(), 3);
  pm.AddPass(::ir::CreateInplacePass());
  pm.Run(res.get());

  return res;
}

}  // namespace framework
}  // namespace paddle
