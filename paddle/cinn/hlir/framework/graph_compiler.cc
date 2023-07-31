// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/graph_compiler.h"

#include <absl/container/flat_hash_map.h>

#include <memory>
#include <unordered_set>

#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace hlir {
namespace framework {

using cinn::common::bfloat16;
using cinn::common::float16;

std::unique_ptr<Program> GraphCompiler::Build(const std::string& code) {
  utils::RecordEvent("GraphCompiler::Build", utils::EventType::kGraph);
  GraphCompiler::CompileOptions options;
  options.attached_code = code;
  options.with_instantiate_variables = true;

  auto&& result = Build(options);
  return std::move(result.runtime_program);
}

void GraphCompiler::CompileOptions::Apply(
    const auto_schedule::TuningResult& tuning_result) {
  // assign options with TuningResult directly
  groups.assign(tuning_result.subgraphs.begin(), tuning_result.subgraphs.end());
  lowered_funcs.assign(tuning_result.function_groups.begin(),
                       tuning_result.function_groups.end());
}

GraphCompiler::CompilationResult GraphCompiler::Build(
    const GraphCompiler::CompileOptions& options,
    std::unordered_set<std::string>&& fetch_var_ids,
    void* stream) {
  Context::Global().ResetNameId();

  // write group's information into FLAGS_cinn_fusion_groups_graphviz_dir
  graph_->VisualizeGroupedGraph(fetch_var_ids.empty() ? fetch_var_ids_
                                                      : fetch_var_ids);

  if (options.with_instantiate_variables) {
    InstantiateVariables();
  }

  VLOG(2) << "Compile With Parallel Compiler!";
  utils::RecordEvent("GraphCompiler CompileResult",
                     utils::EventType::kOrdinary);
  ParallelCompiler::CompileOptions option;
  option.lowered_funcs = options.lowered_funcs;

  parallel_compiler_ =
      std::make_shared<ParallelCompiler>(scope_, graph_, option, target_);
  auto instructions = (*parallel_compiler_.get())();

  if (options.remove_unused_variables) {
    RemoveInvalidVariables(instructions);
  }

  if (options.with_buffer_handle_instruction_inserted) {
    VLOG(3) << "option.with_buffer_handle_instruction_inserted enable";
    InsertBufferHandlers(&instructions);
  }
  VLOG(2) << "Compile With Parallel Compiler Done!";

  GraphCompiler::CompilationResult compilation_result;
  compilation_result.runtime_program.reset(
      new Program(scope_, std::move(instructions)));
  return compilation_result;
}

void GraphCompiler::InstantiateVariables() {
  VLOG(3) << "Instantiate all variables on compile-time";
  utils::RecordEvent("GraphCompiler MutableData", utils::EventType::kOrdinary);
  // All variables reside in scope_, so traverse it to instantiate each one
  for (auto& name : scope_->var_names()) {
    auto* var = scope_->Var<Tensor>(std::string({name.data(), name.size()}));
    auto& tensor = absl::get<Tensor>(*var);
    if (reuse_vars_map_.count(name)) {
      auto src_var_name = reuse_vars_map_.at(name);
      auto* src_var = scope_->Var<Tensor>(src_var_name);
      auto& src_tensor = absl::get<Tensor>(*src_var);
      tensor->set_buffer(src_tensor->get_buffer());
    } else {
      tensor->mutable_data(target_, tensor->type());
    }
  }
}

void GraphCompiler::RemoveInvalidVariables(
    const std::vector<std::unique_ptr<Instruction>>& instructions) {
  // mark all variables are invalid initially
  utils::RecordEvent("GraphCompiler RemoveInvalidVariables",
                     utils::EventType::kOrdinary);
  std::unordered_set<std::string> invalid_variables;
  auto var_names = scope_->var_names();
  invalid_variables.reserve(var_names.size());
  std::transform(
      var_names.begin(),
      var_names.end(),
      std::inserter(invalid_variables, invalid_variables.end()),
      [](const auto& name_view) { return std::string(name_view.data()); });

  // erase used variable names
  auto exclude_arguments_fn =
      [&invalid_variables](const std::vector<std::string>& args) {
        std::for_each(args.begin(),
                      args.end(),
                      [&invalid_variables](const std::string& var_name) {
                        invalid_variables.erase(var_name);
                      });
      };

  // iterate the arguments of each instruction, eliminate the
  // used variables, and remain variables are invalid finally
  auto unused_var_num = invalid_variables.size();
  VLOG(3) << "Before removing invalid variables: " << instructions.size()
          << " instructions, " << invalid_variables.size() << " variables";
  for (auto i = 0; i < instructions.size(); ++i) {
    const auto& instr = instructions.at(i);
    const auto& in_args = instr->GetInArgs();
    const auto& out_args = instr->GetOutArgs();
    std::for_each(in_args.begin(), in_args.end(), exclude_arguments_fn);
    std::for_each(out_args.begin(), out_args.end(), exclude_arguments_fn);

    VLOG(3) << "Instruction-" << i << " filter "
            << unused_var_num - invalid_variables.size() << " used variables";
    unused_var_num = invalid_variables.size();
  }

  VLOG(3) << "There are " << unused_var_num
          << " invalid variables to be removed from scope";
  std::for_each(invalid_variables.begin(),
                invalid_variables.end(),
                [this](const std::string& var_name) {
                  scope_->EraseVar(var_name);
                  VLOG(3) << "Variable(" << var_name << ") is erased";
                });
}

static void BufferMallocWithCallback(void* args, int num_args) {
  cinn_pod_value_t* pod_args = static_cast<cinn_pod_value_t*>(args);
  for (int i = 0; i < num_args; ++i) {
    cinn_buffer_t* buffer = static_cast<cinn_buffer_t*>(pod_args[i]);
    CHECK(buffer->external_malloc)
        << "external_malloc is nullptr at " << i << "-th argumemnts";
    buffer->external_malloc->operator()(nullptr, buffer);
  }
}

static void BufferFreeWithCallback(void* args, int num_args) {
  cinn_pod_value_t* pod_args = static_cast<cinn_pod_value_t*>(args);
  for (int i = 0; i < num_args; ++i) {
    cinn_buffer_t* buffer = static_cast<cinn_buffer_t*>(pod_args[i]);
    CHECK(buffer->external_free) << "external_free is nullptr";
    buffer->external_free->operator()(nullptr, buffer);
  }
}

void GraphCompiler::AnalyzeVariableLifeTime(
    const std::vector<std::unique_ptr<Instruction>>& instructions,
    std::unordered_map<int, std::vector<std::string>>* step2malloc,
    std::unordered_map<int, std::vector<std::string>>* step2free) {
  utils::RecordEvent("GraphCompiler AnalyzeVariableLifeTime",
                     utils::EventType::kOrdinary);
  absl::flat_hash_map<std::string, int> variable_last_used, variable_first_used;
  for (auto step = 0; step < instructions.size(); ++step) {
    const auto& instr = instructions.at(step);

    for (const auto& args : instr->GetInArgs()) {
      for (const auto& var_name : args) {
        // use try_emplace to record the first time a variable appearance
        variable_first_used.try_emplace(var_name, step);
        // will update until last time a variable used
        variable_last_used[var_name] = step;
      }
    }
    for (const auto& args : instr->GetOutArgs()) {
      for (const auto& var_name : args) {
        variable_first_used.try_emplace(var_name, step);
        variable_last_used[var_name] = step;
      }
    }
  }

  for (const auto& var2first : variable_first_used) {
    (*step2malloc)[var2first.second].emplace_back(var2first.first);
  }
  for (const auto& var2last : variable_last_used) {
    (*step2free)[var2last.second].emplace_back(var2last.first);
  }
}

void GraphCompiler::InsertBufferHandlers(
    std::vector<std::unique_ptr<Instruction>>* instructions) {
  utils::RecordEvent("GraphCompiler InsertBufferHandlers",
                     utils::EventType::kOrdinary);
  std::unordered_map<int, std::vector<std::string>> step2malloc, step2free;
  AnalyzeVariableLifeTime(*instructions, &step2malloc, &step2free);

  std::vector<std::unique_ptr<Instruction>> results;
  for (auto step = 0; step < instructions->size(); ++step) {
    auto& instr = instructions->at(step);

    // insert a buffer malloc instruction applying on variables
    // before they are firstly used in the next instruction
    auto m_it = step2malloc.find(step);
    if (m_it != step2malloc.end()) {
      const auto& malloc_var_names = m_it->second;
      auto function_name = "malloc_buffer_instruction_" + std::to_string(step);
      auto malloc_instr =
          std::make_unique<Instruction>(common::DefaultHostTarget(),
                                        scope_.get(),
                                        malloc_var_names,
                                        std::vector<std::string>({}),
                                        function_name);
      VLOG(4) << "seting malloc function " << function_name << " for var "
              << cinn::utils::Join(malloc_var_names, ", ");
      malloc_instr->SetLoweredFunc(
          reinterpret_cast<void*>(BufferMallocWithCallback), function_name);
      malloc_instr->Finalize();
      results.emplace_back(std::move(malloc_instr));
    }

    // join the real computation instruction
    results.emplace_back(std::move(instr));

    // insert a buffer free instruction applying on variables
    // after no instruction will use them anymore
    auto f_it = step2free.find(step);
    if (f_it != step2free.end()) {
      const auto& free_var_names = f_it->second;
      auto function_name = "free_buffer_instruction_" + std::to_string(step);
      auto free_instr =
          std::make_unique<Instruction>(common::DefaultHostTarget(),
                                        scope_.get(),
                                        std::vector<std::string>({}),
                                        free_var_names,
                                        function_name);
      VLOG(4) << "setting free function " << function_name << " for var "
              << cinn::utils::Join(free_var_names, ", ");
      free_instr->SetLoweredFunc(
          reinterpret_cast<void*>(BufferFreeWithCallback), function_name);
      free_instr->Finalize();
      results.emplace_back(std::move(free_instr));
    }
  }

  // replace original instructions
  instructions->swap(results);
}

std::shared_ptr<Scope> BuildScope(Target target,
                                  const std::shared_ptr<Graph>& graph,
                                  std::shared_ptr<Scope> scope) {
  utils::RecordEvent("GraphCompiler BuildScope", utils::EventType::kOrdinary);
  auto& shape_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");
  auto& dtype_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  if (!scope) scope = std::make_shared<Scope>();
  for (auto& iter : shape_dict) {
    auto* var = scope->Var<Tensor>(iter.first);
    auto& tensor = absl::get<Tensor>(*var);
    std::vector<Shape::dim_t> shape;
    for (auto& shape_dim : iter.second) {
      shape.push_back(Shape::dim_t(shape_dim));
    }
    VLOG(3) << "Tensor [" << iter.first << "] resize to "
            << utils::Join(shape, ",");
    tensor->Resize(Shape{shape});
    CHECK(dtype_dict.count(iter.first));
    CHECK(dtype_dict.at(iter.first).is_supported())
        << "The dtype of node " << iter.first
        << " is not float or bool or int! Its type "
        << dtype_dict.at(iter.first).type() << ", "
        << dtype_dict.at(iter.first).bits() << " is not implemented yet.";
    tensor->set_type(dtype_dict.at(iter.first));
  }
  return scope;
}

std::vector<ir::LoweredFunc> GetFuncFromImpl(
    const std::shared_ptr<OpImpl>& impl,
    const common::CINNValuePack& cinn_inputs,
    std::vector<ir::Tensor>& all_arg_tensors,  // NOLINT
    const std::vector<std::string>& input_output_nodes,
    const std::string& node_id,
    const Target& target) {
  utils::RecordEvent("GraphCompiler GetFuncFromImpl",
                     utils::EventType::kOrdinary);
  // 1.Call Op's Compute function, using the default stages and LowerVec to get
  // IR tree.
  common::CINNValuePack C = impl->fcompute(cinn_inputs);

  // 2. Collect tensors and arguments
  // Add output tensors to all_arg_tensors
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    // checkout whether the tensor is with buffer.
    if (!temp.as_tensor_ref()->buffer.defined() ||
        target != common::DefaultNVGPUTarget()) {
      all_arg_tensors.push_back(temp.as_tensor_ref());
    }
  }

  poly::StageMap stages = C.back();
  std::string func_name_prefix = "fn_";
  auto funcs = lang::LowerVec(func_name_prefix + node_id,
                              stages,
                              all_arg_tensors,
                              {},
                              {},
                              nullptr,
                              target,
                              true);

  std::vector<common::CINNValue> schedule_inputs;
  for (int i = 0; i < C.size() - 1; ++i) {
    CHECK(C[i].is_tensor());
    schedule_inputs.push_back(common::CINNValue(C[i]));
  }
  for (auto& f : funcs) {
    schedule_inputs.push_back(common::CINNValue(f->body));
  }

  // 3. Call Op's Schedule function, optimizing the IR tree by new IR schedule
  common::CINNValuePack expr_pack =
      impl->fschedule(common::CINNValuePack{schedule_inputs});

  // 4. Optimize the LoweredFunc
  VLOG(3) << "expr_pack.size() is : " << expr_pack.size()
          << ", funcs.size() is " << funcs.size();
  VLOG(3) << "input_output_nodes.size() is: " << input_output_nodes.size()
          << ", all_arg_tensors.size() is: " << all_arg_tensors.size();
  std::vector<ir::LoweredFunc> funcs_after_schedule;
  CHECK_GE(funcs.size(), expr_pack.size());
  if (funcs.size() > expr_pack.size() ||
      all_arg_tensors.size() > input_output_nodes.size()) {
    for (int i = 0; i < funcs.size(); i++) {
      for (int j = 0; j < expr_pack.size(); j++) {
        Expr temp = expr_pack[j];
        if (temp == funcs[i]->body) {
          auto new_args = lang::GetArgs(funcs[i]->body, input_output_nodes);
          funcs[i]->args = new_args;
          funcs_after_schedule.push_back(funcs[i]);
          break;
        }
      }
    }
  } else if (funcs.size() == expr_pack.size()) {
    funcs_after_schedule = funcs;
  } else {
    LOG(FATAL) << "The number of funcs should not less than expr_pack's";
  }
  CHECK_EQ(funcs_after_schedule.size(), expr_pack.size());
  std::vector<ir::LoweredFunc> res;
  for (int i = 0; i < funcs_after_schedule.size(); i++) {
#ifdef CINN_WITH_CUDA
    optim::OptimizeExprGPU(&(funcs_after_schedule[i]->body));
#endif
    auto temp_buffers = lang::GetTempBuffers(
        all_arg_tensors, stages, funcs_after_schedule[i]->body);
    funcs_after_schedule[i]->temp_bufs = temp_buffers;
    funcs_after_schedule[i] =
        ir::_LoweredFunc_::Make(funcs_after_schedule[i]->name,
                                funcs_after_schedule[i]->args,
                                funcs_after_schedule[i]->body,
                                funcs_after_schedule[i]->temp_bufs);
    res.emplace_back(
        optim::Optimize(Expr(funcs_after_schedule[i]), target, false)
            .as_lowered_func_ref());
  }
  // 5. Return the result.
  return res;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
