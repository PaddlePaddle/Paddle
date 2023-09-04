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
#include "paddle/cinn/hlir/framework/new_ir/utils.h"
#include "paddle/cinn/utils/attribute_util.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/ir/core/builtin_type.h"

namespace cinn {
namespace hlir {
namespace framework {
using newir::CompatibleInfo;

// TODO(Aurelius84): Need abstract this logic to implement Proxy for
// the co-existance with GraphCompiler.
std::unique_ptr<Program> NewIRCompiler::Build() {
  m_builder_.Clear();
  // NOTE(Aurelius84): Currently only support each op for one group
  std::vector<newir::GroupPtr> groups;
  for (auto it = program_.block()->begin(); it != program_.block()->end();
       ++it) {
    std::vector<::ir::Operation*> ops = {*it};
    groups.push_back(std::make_shared<newir::Group>(ops));
    groups.back()->fn_name = CompatibleInfo::GroupOpsName(groups.back()->ops);
  }
  VLOG(4) << "Groups size: " << groups.size();
  return std::move(Build(groups));
}

std::unique_ptr<Program> NewIRCompiler::Build(
    const std::vector<newir::GroupPtr>& groups) {
  auto op_lowerer = CreateOpLowerer<newir::GroupPtr>(target_);

  std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  for (int i = 0; i < groups.size(); ++i) {
    lowered_funcs.emplace_back(op_lowerer.Lower(groups[i]));
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
    const std::vector<newir::GroupPtr>& groups) {
  std::vector<std::unique_ptr<Instruction>> instructions;
  for (int idx = 0; idx < groups.size(); ++idx) {
    // TODO(Aurelius84): only support single op in groups
    auto& op = *(groups[idx]->ops[0]);

    auto& fn_name = groups[idx]->fn_name;
    auto instr = std::unique_ptr<Instruction>(
        new Instruction(target_,
                        scope_.get(),
                        CompatibleInfo::InputNames(op),
                        CompatibleInfo::OutputNames(op),
                        fn_name));
    VLOG(1) << "Lookup kernel name: " << fn_name;
    auto* fn_ptr = compiler_->Lookup(fn_name);
    CHECK(fn_ptr);
    instr->SetLoweredFunc(reinterpret_cast<void*>(fn_ptr), fn_name);
    // As some instruction like reduce, will generate more than one kernel.
    // So try to find the rest kernel, if it exists.
    // SetSubKernels(instr.get(), op_func_name);
    instr->Finalize();
    instructions.push_back(std::move(instr));
  }
  return instructions;
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
