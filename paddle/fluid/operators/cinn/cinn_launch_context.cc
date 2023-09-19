// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/cinn/cinn_launch_context.h"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/paddle2cinn/transform_type.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/cinn/cinn_op_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/value.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace operators::details {

using framework::ParallelExecutor;
using framework::Scope;
using CinnInstruction = ::cinn::hlir::framework::Instruction;
using CinnRuntimeProgram = ::cinn::hlir::framework::Program;
using ::cinn::frontend::paddle::InplaceOutSuffix;
using framework::paddle2cinn::kInplaceVarNames;
using framework::paddle2cinn::kMemOptVarInfoFromMainGraph;
using framework::paddle2cinn::kSkipGcVarNames;
using framework::paddle2cinn::Name2VarInfoMap;

CinnLaunchContext::CinnLaunchContext(const framework::ir::Graph& graph,
                                     const CinnCompiledObject& compiled_obj)
    : cinn_scope_(compiled_obj.scope) {
  // collect all names of the CINN execution arguments
  auto var_names = cinn_scope_->var_names();
  cinn_argument_names_.reserve(var_names.size());
  std::transform(
      var_names.begin(),
      var_names.end(),
      std::inserter(cinn_argument_names_, cinn_argument_names_.end()),
      [](const auto& name_view) { return std::string(name_view.data()); });
  // build name map between the original variables and compiled ones
  BuildVarNameMap(compiled_obj.paddle2cinn_varmap, cinn_argument_names_);

  const auto& input_var_names =
      graph.Get<std::vector<std::string>>(framework::paddle2cinn::kInputVars);
  const auto& output_var_names =
      graph.Get<std::vector<std::string>>(framework::paddle2cinn::kOutputVars);
  inplace_var_names_ =
      graph.Get<std::unordered_set<std::string>>(kInplaceVarNames);
  internal_var_names_ =
      ExtractInternalVarNames(input_var_names, output_var_names);
  // initialize all execution arguments
  InitializeArguments();
  // DEPRECATED(CtfGo): following callback assignment will be deprecated soon
  for (auto&& var_name : input_var_names) {
    if (IsVariableUsed(var_name)) {
      AssignExternalVariable(var_name);
    }
  }
  for (auto&& var_name : output_var_names) {
    if (inplace_var_names_.count(var_name)) {
      VLOG(4) << "Inplaced variable:" << var_name << " -> "
              << var_name + InplaceOutSuffix << " as paddle2cinn varmap key";
      AssignExternalVariable(var_name + InplaceOutSuffix);
    } else {
      AssignExternalVariable(var_name);
    }
  }
  for (auto&& var_name : internal_var_names_) {
    AssignInternalVariable(var_name);
  }

  // Convert the CINN runtime program to a Paddle graph
  runtime_program_desc_ = BuildCompiledProgram(graph, compiled_obj);
  runtime_graph_ =
      std::make_unique<framework::ir::Graph>(*runtime_program_desc_.get());
  auto& outer_varinfo = graph.Get<Name2VarInfoMap>(kMemOptVarInfoFromMainGraph);
  runtime_graph_->SetNotOwned<Name2VarInfoMap>(kMemOptVarInfoFromMainGraph,
                                               &outer_varinfo);
  // use kSkipGcVarNames attr of graph to initialize skip_gc_vars_
  if (graph.Has(kSkipGcVarNames)) {
    const auto& skip_gc_vars =
        graph.Get<std::unordered_set<std::string>>(kSkipGcVarNames);
    skip_gc_vars_.insert(skip_gc_vars.begin(), skip_gc_vars.end());
    VLOG(4) << "Append skip_gc_vars:["
            << string::join_strings(skip_gc_vars, ',') << "]";
  }

  // collect variables name list to be skipped in GC
  skip_eager_vars_.reserve(input_var_names.size() + output_var_names.size());
  auto add_skip_var_fn = [&outer_varinfo, this](const std::string& var_name) {
    // Always consider Input/Output of Graph as skip_gc_vars, because
    // InterpreterCore has no eager_deletion_op to deal with it.

    VLOG(4) << "Append a skip_gc_var for InterpreterCore:" << var_name;
    skip_gc_vars_.insert(var_name);
    // if a var exists at the outer_varinfo map, that means it will be
    // erased by the following eager_deletion_op of current cinn_launch op
    if (!outer_varinfo.count(var_name)) {
      skip_eager_vars_.emplace_back(var_name);
      VLOG(4) << "Append a skip_gc_var for PE:" << var_name;
    }
  };
  std::for_each(
      input_var_names.begin(), input_var_names.end(), add_skip_var_fn);
  std::for_each(
      output_var_names.begin(), output_var_names.end(), add_skip_var_fn);
  VLOG(4) << string::Sprintf(
      "Distribution of variables in the graph compiled:"
      "input[%lu],internal[%lu],output[%lu],"
      "outer_eager_deletion[%lu],skip_eager_deletion[%lu],"
      "skip_gc_vars_[%lu]",
      input_var_names.size(),
      internal_var_names_.size(),
      output_var_names.size(),
      outer_varinfo.size(),
      skip_eager_vars_.size(),
      skip_gc_vars_.size());
}

void CinnLaunchContext::BuildVarNameMap(
    const std::unordered_map<std::string, std::string>& compiled_varmap,
    const std::unordered_set<std::string>& argument_names) {
  for (const auto& x : compiled_varmap) {
    if (!argument_names.count(x.second)) {
      // exclude variables not used
      continue;
    }
    // copy to local paddle2cinn map
    paddle2cinn_varmap_.emplace(x.first, x.second);
    // add an entry to local cinn2paddle map reversely
    auto res = cinn2paddle_varmap_.emplace(x.second, x.first);
    PADDLE_ENFORCE_EQ(
        res.second,
        true,
        platform::errors::InvalidArgument(
            "Cinn variable(%s) maps to more than one paddle variable(%s,%s)",
            x.second,
            res.first->second,
            x.first));
  }
  // supplement the relations of the remain variables
  // not appearing in above map, which are internal variables
  // and here we use the names from cinn compiled.
  for (const auto& var_name : argument_names) {
    if (!cinn2paddle_varmap_.count(var_name)) {
      cinn2paddle_varmap_.emplace(var_name, var_name);
      paddle2cinn_varmap_.emplace(var_name, var_name);
    }
  }

  PADDLE_ENFORCE_EQ(
      paddle2cinn_varmap_.size(),
      cinn2paddle_varmap_.size(),
      platform::errors::PreconditionNotMet(
          "Size of variables is not euqal, paddle[%ld] vs cinn[%ld]",
          paddle2cinn_varmap_.size(),
          cinn2paddle_varmap_.size()));
}

std::unordered_set<std::string> CinnLaunchContext::GetVisibleVarNames() const {
  std::unordered_set<std::string> remain_var_names;
  for (const auto& pair : paddle2cinn_varmap_) {
    remain_var_names.insert(this->RedirectVarName(pair.first));
  }
  return remain_var_names;
}

void CinnLaunchContext::UpdateCapturedEnv(const framework::Scope& scope,
                                          const platform::Place& place) {
  if (std::addressof(scope) == cached_scope_ &&
      std::addressof(place) == cached_place_) {
    VLOG(4) << "Captured scope:" << cached_scope_ << ", place:" << cached_place_
            << " are not changed";
    return;
  }
  cached_scope_ = std::addressof(scope);
  cached_place_ = std::addressof(place);
  cached_temp_scope_ = scope.NewTmpScope();
  VLOG(4) << "Captured env is update, scope:" << cached_scope_ << "->"
          << std::addressof(scope) << ", place:" << cached_place_ << "->"
          << std::addressof(place);
}

bool CinnLaunchContext::IsVariableUsed(const std::string& var_name) const {
  return paddle2cinn_varmap_.count(var_name) > 0;
}

CinnTensor CinnLaunchContext::GetCinnTensorOfVar(const std::string& var_name) {
  PADDLE_ENFORCE_EQ(
      IsVariableUsed(var_name),
      true,
      platform::errors::NotFound("Variable(%s) not applied in CINN", var_name));
  const auto& arg_name = paddle2cinn_varmap_.at(var_name);
  return cinn_scope_->GetTensor(arg_name);
}

std::unordered_set<std::string> CinnLaunchContext::ExtractInternalVarNames(
    const std::vector<std::string>& input_var_names,
    const std::vector<std::string>& output_var_names) {
  std::unordered_set<std::string> remain_var_names;
  remain_var_names.reserve(paddle2cinn_varmap_.size());
  std::transform(paddle2cinn_varmap_.begin(),
                 paddle2cinn_varmap_.end(),
                 std::inserter(remain_var_names, remain_var_names.end()),
                 [](const auto& name_pair) { return name_pair.first; });

  // exclude the input variables and output variables
  auto exclude_names_fn = [this,
                           &remain_var_names](const std::string& var_name) {
    remain_var_names.erase(var_name);
    if (inplace_var_names_.count(var_name)) {
      remain_var_names.erase(var_name + InplaceOutSuffix);
    }
  };

  VLOG(1) << "Input var list: " << string::join_strings(input_var_names, ", ");
  VLOG(1) << "Output var list: "
          << string::join_strings(output_var_names, ", ");
  std::for_each(
      input_var_names.begin(), input_var_names.end(), exclude_names_fn);
  std::for_each(
      output_var_names.begin(), output_var_names.end(), exclude_names_fn);
  VLOG(1) << "Internal var list: "
          << string::join_strings(remain_var_names, ", ");
  return remain_var_names;
}

void CinnLaunchContext::CheckTensorEquivalent(
    const std::string& var_name, const phi::DenseTensor& paddle_tensor) {
  PADDLE_ENFORCE_EQ(IsVariableUsed(var_name),
                    true,
                    platform::errors::InvalidArgument(
                        "Variable(%s) not applied in cinn", var_name));
  // check dimension
  auto cinn_tensor = GetCinnTensorOfVar(var_name);
  auto cinn_dims = phi::make_ddim(cinn_tensor->shape().data());
  if (paddle_tensor.dims().size() == 0) {
    // VLOG when paddle inputs 0D-Tensor
    VLOG(4) << "Paddle inputs 0D-Tensor, CINN changes 0D-Tensor " << var_name
            << " to 1D-Tensor";
    PADDLE_ENFORCE_EQ(phi::make_ddim({1}),
                      cinn_dims,
                      phi::errors::PreconditionNotMet(
                          "Tensor's shape of variable(%s) are not consistent, "
                          "paddle inputs 0D-Tensor, cinn should get 1D-Tensor "
                          "instead of [%s].",
                          var_name,
                          paddle_tensor.dims(),
                          cinn_dims));
  } else {
    PADDLE_ENFORCE_EQ(paddle_tensor.dims(),
                      cinn_dims,
                      phi::errors::PreconditionNotMet(
                          "Tensor's shape of variable(%s) are not equivalent, "
                          "paddle is = [%s], but cinn is = [%s].",
                          var_name,
                          paddle_tensor.dims(),
                          cinn_dims));
  }

  auto cinn_dtype =
      framework::paddle2cinn::TransToPaddleDataType(cinn_tensor->type());
  PADDLE_ENFORCE_EQ(paddle_tensor.dtype(),
                    cinn_dtype,
                    platform::errors::PreconditionNotMet(
                        "Tensors' dtype in variable(%s) are not equivalent, "
                        "paddle is = [%s], but cinn is = [%s].",
                        var_name,
                        paddle_tensor.dtype(),
                        cinn_dtype));
}

void CinnLaunchContext::InitializeArguments() {
  for (auto&& arg : cinn_argument_names_) {
    auto cinn_buffer = std::make_unique<cinn_buffer_t>();
    auto paddle_varname = cinn2paddle_varmap_.at(arg);
    auto cinn_tensor = GetCinnTensorOfVar(paddle_varname);
    // assign dimensions with corresponding compiled tensor
    cinn_buffer->resize(cinn_tensor->shape().data().data(),
                        cinn_tensor->shape().data().size());
    cinn_buffer->type = cinn::runtime::ToRuntimeType(cinn_tensor->type());
    VLOG(4) << string::Sprintf(
        "Append an argument:name(%s),paddle_name(%s), "
        "dims(%s),type(%s),tensor(%p)",
        arg,
        paddle_varname,
        framework::DDim(cinn_buffer->dims, cinn_buffer->dimensions).to_str(),
        cinn_tensor->type(),
        cinn_tensor.get());
    name2argument_.emplace(arg, cinn_buffer.get());

    paddle2argument_.emplace(paddle_varname, cinn_buffer.get());
    hold_buffers_.emplace_back(std::move(cinn_buffer));
  }
  VLOG(4) << "Total argument size:" << name2argument_.size();
}

void CinnLaunchContext::AssignExternalVariable(const std::string& var_name) {
  PADDLE_ENFORCE_EQ(IsVariableUsed(var_name),
                    true,
                    platform::errors::InvalidArgument(
                        "Variable(%s) not applied in cinn", var_name));
  auto* cinn_buffer = GetCinnBufferOfVar(var_name);
  std::string revise_var_name = RedirectVarName(var_name);
  // assign external malloc/free callbacks of cinn_buffer_t
  cinn_buffer->external_malloc = new std::function<int(void*, cinn_buffer_t*)>(
      [this, revise_var_name](void* ctx, cinn_buffer_t* buffer) {
        auto* tensor = cached_scope_->GetVar(revise_var_name)
                           ->GetMutable<phi::DenseTensor>();
        tensor->Resize(framework::DDim(buffer->dims, buffer->dimensions));
        buffer->memory = reinterpret_cast<uint8_t*>(tensor->mutable_data(
            *cached_place_,
            framework::paddle2cinn::TransToPaddleDataType(buffer->type)));
        return 0;
      });

  // external variables will be recycled by global gc, so do nothing here
  cinn_buffer->external_free = new std::function<int(void*, cinn_buffer_t*)>(
      [](void* ctx, cinn_buffer_t* buffer) {
        // Do nothing
        return 0;
      });
}

void CinnLaunchContext::AssignInternalVariable(const std::string& var_name) {
  PADDLE_ENFORCE_EQ(IsVariableUsed(var_name),
                    true,
                    platform::errors::InvalidArgument(
                        "Variable(%s) not applied in cinn", var_name));
  auto* cinn_buffer = GetCinnBufferOfVar(var_name);
  std::string revise_var_name = RedirectVarName(var_name);
  // assign external malloc/free callbacks of cinn_buffer_t
  cinn_buffer->external_malloc = new std::function<int(void*, cinn_buffer_t*)>(
      [this, revise_var_name](void* ctx, cinn_buffer_t* buffer) {
        auto* tensor = cached_temp_scope_->Var(revise_var_name)
                           ->GetMutable<phi::DenseTensor>();
        tensor->Resize(framework::DDim(buffer->dims, buffer->dimensions));
        buffer->memory = reinterpret_cast<uint8_t*>(tensor->mutable_data(
            *cached_place_,
            framework::paddle2cinn::TransToPaddleDataType(buffer->type)));
        return 0;
      });

  // internal variables should release its buffer immediately
  // if no instruction use it
  cinn_buffer->external_free = new std::function<int(void*, cinn_buffer_t*)>(
      [this, revise_var_name](void* ctx, cinn_buffer_t* buffer) {
        auto* tensor = cached_temp_scope_->GetVar(revise_var_name)
                           ->GetMutable<phi::DenseTensor>();
        tensor->clear();
        return 0;
      });
}

std::unique_ptr<framework::ProgramDesc> CinnLaunchContext::BuildCompiledProgram(
    const framework::ir::Graph& graph, const CinnCompiledObject& compiled_obj) {
  CinnRuntimeProgram* runtime_program = compiled_obj.runtime_program.get();
  // Step 0: Create an empty program_desc, there will be only one block
  // framework::ProgramDesc program_desc;
  std::unique_ptr<framework::ProgramDesc> program_desc(
      new framework::ProgramDesc());
  auto* block = program_desc->MutableBlock(0);
  const std::vector<std::unique_ptr<CinnInstruction>>& instructions =
      runtime_program->GetRunInstructions();

  // build a map that links the name of a Paddle variable to its VarDesc
  const std::unordered_set<framework::ir::Node*>& nodes = graph.Nodes();
  std::unordered_map<std::string, framework::VarDesc*> original_vardescs;
  for (auto* node : nodes) {
    if (node->IsVar() && node->Var()) {
      original_vardescs.emplace(node->Name(), node->Var());
    }
  }

  // Step 1: Create a VarDesc for each execution argument:
  //   (1) For those variables that are input or output variables of the
  //   original subgraph, there must exist an original VarDesc, so
  //   we copy some useful info(such as IsParameter,Persistable)
  //   to the new VarDesc.
  //   (2) For all variables, the shape, data type of their VarDescs
  //   are set by values of the corresponding compiled tensors,
  //   including the in/out variables where the equiality between their tensors
  //   and the CINN compiled ones is verified in corresponding cinn_launch_op.
  for (auto&& arg : cinn_argument_names_) {
    const std::string& var_name = cinn2paddle_varmap_.at(arg);
    framework::VarDesc* var_desc = block->Var(var_name);
    var_desc->SetType(framework::proto::VarType::LOD_TENSOR);

    auto res = original_vardescs.find(var_name);
    if (res != original_vardescs.end()) {
      auto* ori_desc = res->second;
      var_desc->SetPersistable(ori_desc->Persistable());
      var_desc->SetIsParameter(ori_desc->IsParameter());
    }

    auto cinn_tensor = GetCinnTensorOfVar(var_name);
    var_desc->SetDataType(framework::TransToProtoVarType(
        framework::paddle2cinn::TransToPaddleDataType(cinn_tensor->type())));
    var_desc->SetShape(std::vector<int64_t>(cinn_tensor->shape().data().begin(),
                                            cinn_tensor->shape().data().end()));
  }

  // transform names of the input or output arguments of a CINN instruction
  // to the corresponding Paddle variable names, and repack them as one vector
  auto trans_and_pack_args_fn =
      [this](const std::vector<std::vector<std::string>>& cinn_args_array) {
        std::vector<std::string> var_names;
        for (auto&& cinn_args : cinn_args_array) {
          for (auto&& arg : cinn_args) {
            auto res = cinn2paddle_varmap_.find(arg);
            PADDLE_ENFORCE_NE(
                res,
                cinn2paddle_varmap_.end(),
                platform::errors::NotFound("Argument(%s) not found", arg));
            var_names.emplace_back(res->second);
          }
        }
        return var_names;
      };

  // Step 2: create a VarDesc of cinn_instruction_run op for
  //         each CINN instruction and append it to the main block
  for (auto ins_idx = 0; ins_idx < instructions.size(); ++ins_idx) {
    auto* ins = instructions.at(ins_idx).get();
    auto in_args = trans_and_pack_args_fn(ins->GetInArgs());
    auto out_args = trans_and_pack_args_fn(ins->GetOutArgs());
    auto* op_desc = block->AppendOp();
    op_desc->SetType("cinn_instruction_run");
    op_desc->SetInput(kX, in_args);
    op_desc->SetOutput(kOutputs, out_args);
    op_desc->SetAttr(kCachedIndex,
                     {static_cast<int64_t>(compiled_obj.cached_index)});
    op_desc->SetAttr(kInstructionIndex, {static_cast<int64_t>(ins_idx)});
  }

  return program_desc;
}

ParallelExecutor* CinnLaunchContext::InitializePE(const platform::Place& place,
                                                  framework::Scope* scope) {
  if (!parallel_executor_) {
    framework::details::ExecutionStrategy exec_strategy;
    exec_strategy.num_threads_ = 1;
    exec_strategy.use_device_ = platform::Place2DeviceType(place);
    framework::details::BuildStrategy build_strategy;
    parallel_executor_ = std::make_unique<ParallelExecutor>(
        place, scope, exec_strategy, build_strategy, runtime_graph_.get());
  }

  // update the scope bound to an OpHandle and rebuild temporary variables
  VLOG(4) << "Reset scope and initialize temporary variables";
  std::unordered_map<Scope*, Scope*> scope_map = {
      {parallel_executor_->GetLocalScopes().front(), scope}};
  parallel_executor_->ResetOpHandleScopeMapOfGraphs(scope_map);
  // instead of using the PrepareVariables function of ParallelExecutor to
  // initialize all variables, here we only initialize internal variables
  // because external variables are already included in parent scope.
  for (auto&& var_name : internal_var_names_) {
    auto* var = scope->FindVar(var_name);
    if (var != nullptr) {
      VLOG(5) << "internal variable:" << var_name
              << " has been initialized beforehand in global scope, skipped.";
      continue;
    }
    framework::InitializeVariable(scope->Var(var_name),
                                  framework::proto::VarType::LOD_TENSOR);
  }

  return parallel_executor_.get();
}

framework::InterpreterCore* CinnLaunchContext::InitializeInterpreterCore(
    const platform::Place& place, framework::Scope* scope) {
  if (!interpreter_core_ || scope != cached_scope_) {
    VLOG(1) << "interpreter_core_ is null or scope != cached_scope_: "
               "interpreter_core_: "
            << interpreter_core_.get() << "; scope: " << scope
            << "; cached_scope_: " << cached_scope_;
    VLOG(1) << "Internal var list: "
            << string::join_strings(internal_var_names_, ", ");

    for (auto&& var_name : internal_var_names_) {
      auto* var = scope->FindVar(var_name);
      if (var != nullptr) {
        continue;
      }
      VLOG(4) << "Create Variable " << var_name << " locally";
      framework::InitializeVariable(scope->Var(var_name),
                                    framework::proto::VarType::LOD_TENSOR);
    }

    // Actually, cinn_instruction will not use the var with name
    // var_name+InplaceOutSuffix in paddle scope, but use the var with name.
    // That means, var 'a' and 'a@InplaceOut' in cinn scope both links to var
    // 'a' in paddle scope.

    // So, why create 'a@InplaceOut' here?
    // In order to make some paddle functions can visit all inputs and outputs
    // of cinn_instruction_run op, for example, infer_shape function.

    // It should be refined.
    for (auto&& var_name : inplace_var_names_) {
      auto name = var_name + InplaceOutSuffix;
      auto* var = scope->FindVar(name);
      if (var != nullptr) {
        continue;
      }
      VLOG(4) << "Create Variable " << name << " locally";
      framework::InitializeVariable(scope->Var(name),
                                    framework::proto::VarType::LOD_TENSOR);
    }
    if (!interpreter_core_) {
      framework::interpreter::ExecutionConfig execution_config;
      execution_config.create_local_scope = false;
      execution_config.used_for_cinn = true;
      execution_config.skip_gc_vars = skip_gc_vars_;
      interpreter_core_ = std::make_unique<framework::InterpreterCore>(
          place, runtime_program_desc_->Block(0), scope, execution_config);
    } else {
      interpreter_core_->reset_scope(scope);
    }
    UpdateCapturedEnv(*scope, place);
  }
  return interpreter_core_.get();
}

std::string CinnLaunchContext::RedirectVarName(
    const std::string& var_name) const {
  auto pos = var_name.find(InplaceOutSuffix);
  if (pos == std::string::npos) {
    return var_name;
  }
  std::string remove_suffix_name = var_name.substr(0, pos);
  if (!inplace_var_names_.count(remove_suffix_name)) {
    return var_name;
  }
  VLOG(4) << "Inplaced variable:" << var_name << " redirect to "
          << remove_suffix_name;
  return remove_suffix_name;
}

cinn_buffer_t* CinnLaunchContext::GetCinnBufferOfVar(
    const std::string& var_name) {
  auto res = paddle2argument_.find(var_name);
  PADDLE_ENFORCE_NE(
      res,
      paddle2argument_.end(),
      platform::errors::NotFound("Variable(%s) not found in compilation result",
                                 var_name));
  return static_cast<cinn_buffer_t*>(res->second);
}

}  // namespace operators::details
}  // namespace paddle
