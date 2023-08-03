// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/new_ir_compiler.h"

#include <absl/types/variant.h>
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/utils/attribute_util.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/ir/core/builtin_type.h"

namespace cinn {
namespace hlir {
namespace framework {

const std::unordered_map<std::string, std::string> CompatibleInfo::OP_NAMES = {
    {"pd.full", "fill_constant"}, {"pd.matmul", "matmul"}};

// TODO(Aurelius84): Need abstract this logic to implement Proxy for
// the co-existance with GraphCompiler.
std::unique_ptr<Program> NewIRCompiler::Build() {
  m_builder_.Clear();
  // NOTE(Aurelius84): Currently only support each op for one group
  std::vector<std::vector<::ir::Operation*>> groups;
  for (auto it = program_.block()->begin(); it != program_.block()->end();
       ++it) {
    groups.push_back({*it});
  }
  VLOG(4) << "Groups size: " << groups.size();

  std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  for (int i = 0; i < groups.size(); ++i) {
    lowered_funcs.emplace_back(GetOpFunc(*groups[i][0], i));
  }

  for (auto&& lowered_func : lowered_funcs) {
    ProcessFunction(lowered_func);
  }

  compiler_ = backends::Compiler::Create(target_);
  auto build_module = m_builder_.Build();
  compiler_->Build(build_module, "");

  auto instructions = BuildInstructions(groups);

  // TODO(Aurelius84): Instantiate all tensors on compile-time, which is
  // controlled by 'options.with_instantiate_variables' in GraphCompiler.
  // Moreover, it's better to implement InsertBufferHandlers() logic
  // to automatically insert Malloc and Free instructions.
  for (auto& name : scope_->var_names()) {
    std::string var_name({name.data(), name.size()});
    VLOG(4) << "Instantiate " << var_name << " on compile-time";
    auto* var = scope_->Var<Tensor>(var_name);
    auto& tensor = absl::get<Tensor>(*var);
    tensor->mutable_data(target_, tensor->type());
  }
  return std::make_unique<Program>(scope_, std::move(instructions));
}

std::vector<ir::LoweredFunc> NewIRCompiler::GetOpFunc(const ::ir::Operation& op,
                                                      int idx) {
  std::vector<ir::Tensor> inputs;
  std::vector<common::CINNValue> cinn_inputs;
  auto op_name = op.name();
  VLOG(4) << "GetOpFunc for op: " << op_name;
  // step 1: Deal with Oprands
  for (int i = 0; i < op.num_operands(); ++i) {
    auto in_value = op.operand_source(i);
    // TODO(Aurelius84): For now, use addr as name but it's not wise.
    std::string input_id = CompatibleInfo::kInputPrefix +
                           std::to_string(std::hash<::ir::Value>()(in_value));
    auto type_info =
        in_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

    auto in_shape = phi::vectorize<int>(type_info.dims());
    auto dtype = type_info.dtype();
    ir::Tensor temp = lang::CreatePlaceHolder(
        in_shape, utils::ConvertIRType(dtype), input_id);
    inputs.push_back(temp);
    cinn_inputs.push_back(common::CINNValue(temp));
  }
  for (auto out_name : OpGetOutputNames(op)) {
    cinn_inputs.push_back(common::CINNValue(out_name));
  }

  VLOG(4) << "inputs.size(): " << inputs.size();

  // step 2: Deal with OpResult
  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;
  for (int i = 0; i < op.num_results(); ++i) {
    auto out_value = op.result(i);
    auto type_info =
        out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();
    out_types.push_back(utils::ConvertIRType(type_info.dtype()));
    auto out_shape = phi::vectorize<int>(type_info.dims());
    out_shapes.push_back(std::move(out_shape));
  }
  VLOG(4) << "out_types.size(): " << out_types.size();

  NodeAttr node_attrs;
  {
    VLOG(4) << "op.attributes():" << op.attributes().size();
    auto attrs = utils::ConvertAttributes(op.attributes());
    node_attrs.node_name = CompatibleInfo::OP_NAMES.at(op_name);
    node_attrs.attr_store = std::move(attrs);
  }
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  // NOTE(Aurelius84): Do we need replace all hlir::framework Operator with
  // ::ir::Program ？
  const hlir::framework::Operator* cinn_op =
      Operator::Get(CompatibleInfo::OP_NAMES.at(op_name));
  auto impl = OpStrategy::SelectImpl(
      strategy[cinn_op](node_attrs, inputs, out_types, out_shapes, target_));
  common::CINNValuePack C = impl->fcompute(common::CINNValuePack{cinn_inputs});
  poly::StageMap stages = C.back();
  // make sure all the tensors in the stages before schedule launch.
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    stages->InsertLazily(temp.as_tensor_ref());
  }
  C = impl->fschedule(C);
  for (int i = 0; i < C->size() - 1; i++) {
    ir::Expr temp = C[i];
    // checkout whether the tensor is with buffer.
    if ((!temp.as_tensor_ref()->buffer.defined() ||
         this->target_ != common::DefaultNVGPUTarget()) &&
        !stages[temp.as_tensor_ref()]->inlined()) {
      inputs.push_back(temp.as_tensor_ref());
    }
  }
  auto func = lang::LowerVec(
      GenOpFuncName(op, idx), stages, inputs, {}, {}, nullptr, target_);
  return func;
}

void NewIRCompiler::ProcessFunction(
    const std::vector<ir::LoweredFunc>& lowered_funcs) {
  for (auto&& func : lowered_funcs) {
    for (auto&& arg : func->args) {
      std::string arg_name = arg.name();
      if (arg_name[0] == '_') arg_name = arg_name.substr(1);

      auto* var = scope_->FindVar(arg_name);
      // For argument buffer not in scope, create it.
      if (!var && arg.is_buffer()) {
        auto* new_var = scope_->Var<Tensor>(arg_name);
        auto& tensor = absl::get<Tensor>(*new_var);
        std::vector<Shape::dim_t> shape;
        for (auto& shape_dim : arg.buffer_arg()->shape) {
          CHECK(shape_dim.is_constant());
          shape.push_back(static_cast<int>(shape_dim.get_constant()));
        }
        tensor->Resize(Shape{shape});
        tensor->set_type(arg.buffer_arg()->dtype);
      }
    }
    m_builder_.AddFunction(func);
  }
}

std::vector<std::unique_ptr<Instruction>> NewIRCompiler::BuildInstructions(
    const std::vector<std::vector<::ir::Operation*>>& groups) {
  std::vector<std::unique_ptr<Instruction>> instructions;
  for (int idx = 0; idx < groups.size(); ++idx) {
    // TODO(Aurelius84): only support single op in groups
    auto& op = *groups[idx][0];
    auto instr_name = op.name();
    auto instr =
        std::unique_ptr<Instruction>(new Instruction(target_,
                                                     scope_.get(),
                                                     OpGetInputNames(op),
                                                     OpGetOutputNames(op),
                                                     instr_name));
    auto& op_func_name = GenOpFuncName(op, idx);
    auto* fn_ptr = compiler_->Lookup(op_func_name);
    CHECK(fn_ptr);
    instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), op_func_name);
    // As some instruction like reduce, will generate more than one kernel.
    // So try to find the rest kernel, if it exists.
    // SetSubKernels(instr.get(), op_func_name);

    instr->Finalize();
    instructions.push_back(std::move(instr));
  }
  return instructions;
}

const std::string& NewIRCompiler::GenOpFuncName(const ::ir::Operation& op,
                                                int idx) {
  // TODO(Aurelius84): . will raise compiler error in pd.xxx, need more
  // elegant way to generate function name.
  std::string op_name = op.name().substr(3) + "_" + std::to_string(idx);
  std::string func_name = Context::Global().NewName("fn_" + op_name);
  func_names_.try_emplace(op_name, func_name);
  return func_names_.at(op_name);
}

std::vector<std::string> NewIRCompiler::OpGetInputNames(
    const ::ir::Operation& op) {
  std::vector<std::string> names;
  std::unordered_set<std::string> repeat;
  for (int i = 0; i < op.num_operands(); ++i) {
    auto value = op.operand_source(i);
    std::string name = CompatibleInfo::kInputPrefix +
                       std::to_string(std::hash<::ir::Value>()(value));
    if (repeat.count(name)) {
      continue;
    }
    repeat.insert(name);
    names.push_back(name);
  }
  return names;
}

std::vector<std::string> NewIRCompiler::OpGetOutputNames(
    const ::ir::Operation& op) {
  std::vector<std::string> names;
  for (int i = 0; i < op.num_results(); ++i) {
    auto value = op.result(i);
    std::string name = CompatibleInfo::kOutputPrefix +
                       std::to_string(std::hash<::ir::Value>()(value));
    names.push_back(std::move(name));
  }
  return names;
}

std::shared_ptr<Scope> BuildScope(const Target& target,
                                  const ::ir::Program& program) {
  std::unordered_set<::ir::Value> visited;
  auto scope = std::make_shared<Scope>();

  auto create_var = [&](const std::string& name_prefix, ::ir::Value value) {
    if (visited.count(value) > 0) return;
    visited.emplace(value);

    std::string name =
        name_prefix + std::to_string(std::hash<::ir::Value>()(value));
    auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto* var = scope->Var<Tensor>(name);
    auto& tensor = absl::get<Tensor>(*var);
    // NOTE: can be replaced with phi::vectorized ?
    std::vector<Shape::dim_t> shape;
    for (auto i = 0; i < type_info.dims().size(); ++i) {
      shape.push_back(Shape::dim_t(type_info.dims()[i]));
    }
    tensor->Resize(Shape{shape});
    tensor->set_type(utils::ConvertIRType(type_info.dtype()));
  };

  for (auto it = program.block()->begin(); it != program.block()->end(); ++it) {
    for (auto i = 0; i < (*it)->num_operands(); ++i) {
      auto in_value = (*it)->operand_source(i);
      create_var(CompatibleInfo::kInputPrefix, in_value);
    }

    for (auto i = 0; i < (*it)->num_results(); ++i) {
      auto out_value = (*it)->result(i);
      create_var(CompatibleInfo::kOutputPrefix, out_value);
    }
  }
  return scope;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
