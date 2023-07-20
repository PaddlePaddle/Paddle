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

#pragma once
#include <memory>
#include <unordered_map>
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/utils/attribute_util.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/program.h"

#include "paddle/cinn/hlir/framework/graph_compiler.h"

namespace cinn {
namespace hlir {
namespace framework {

// TODO(Aurelius): Need add name mapping logic in REGISTER_CINN_OP
// macros or attempt to unify Op name with Paddle and CINN.
static const std::unordered_map<std::string, std::string> OP_NAMES = {
    {"pd.full", "fill_constant"}, {"pd.matmul", "matmul"}};

// TODO(Aurelius84): Need abstract this logic to implement Proxy for
// the co-existance with GraphCompiler.
class NewIRCompiler final {
 public:
  NewIRCompiler(const ::ir::Program& prog,
                const Target& target,
                const std::shared_ptr<Scope>& scope)
      : program_(prog),
        m_builder_("NewIR", target),  // TODO(dev): need unique name
        target_(target),
        scope_(scope) {}
  std::unique_ptr<Program> Build() {
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
    return std::make_unique<Program>(scope_, std::move(instructions));
  }

  std::vector<ir::LoweredFunc> GetOpFunc(const ::ir::Operation& op, int idx) {
    std::vector<ir::Tensor> inputs;
    std::vector<common::CINNValue> cinn_inputs;
    VLOG(4) << "GetOpFunc for op: " << op.name();
    // step 1: Deal with Oprands
    for (int i = 0; i < op.num_operands(); ++i) {
      auto in_value = op.operand(i);
      // TODO(Aurelius84): For now, use addr as name but it's not wise.
      std::string input_id = std::to_string(std::hash<::ir::Value>()(in_value));
      // NOTE(Aurelius84): whether need to support other Type?
      auto type_info =
          in_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

      auto in_shape = phi::vectorize<int>(type_info.dims());
      ir::Tensor temp;
      auto dtype = type_info.dtype();
      // TODO(Aurelius84): support more type
      if (dtype.isa<::ir::Float32Type>()) {
        temp = lang::Placeholder<float>(input_id, in_shape);
      } else if (dtype.isa<::ir::Int32Type>()) {
        temp = lang::Placeholder<int>(input_id, in_shape);
      }

      inputs.push_back(temp);
      cinn_inputs.push_back(common::CINNValue(temp));
    }
    for (auto out_name : OpGetOutputNames(op)) {
      cinn_inputs.push_back(
          common::CINNValue(op.name().substr(3) + "_" + out_name));
    }

    VLOG(4) << "inputs.size(): " << inputs.size();

    // step 2: Deal with OpResult
    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;
    for (int i = 0; i < op.num_results(); ++i) {
      auto out_value = op.result(i);
      auto type_info =
          out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();
      // TODO(Aurelius84): need to support ::ir::Type -> common::Type
      out_types.push_back(common::Float(32));
      auto out_shape = phi::vectorize<int>(type_info.dims());
      out_shapes.push_back(std::move(out_shape));
    }
    VLOG(4) << "out_types.size(): " << out_types.size();

    NodeAttr node_attrs;
    {
      VLOG(4) << "op.attributes():" << op.attributes().size();
      auto attrs = utils::ConvertAttributes(op.attributes());
      node_attrs.node_name = OP_NAMES.at(op.name());
      node_attrs.attr_store = std::move(attrs);
    }
    auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
    // NOTE(Aurelius84): Do we need replace all hlir::framework Operator with
    // ::ir::Program ？
    const hlir::framework::Operator* cinn_op =
        Operator::Get(OP_NAMES.at(op.name()));
    auto impl = OpStrategy::SelectImpl(
        strategy[cinn_op](node_attrs, inputs, out_types, out_shapes, target_));
    common::CINNValuePack C =
        impl->fcompute(common::CINNValuePack{cinn_inputs});
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

  void ProcessFunction(const std::vector<ir::LoweredFunc>& lowered_funcs) {
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

  std::vector<std::unique_ptr<Instruction>> BuildInstructions(
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

 protected:
  const std::string& GenOpFuncName(const ::ir::Operation& op, int idx) {
    // TODO(Aurelius84): . will raise compiler error in pd.xxx, need more
    // elegant way to generate function name.
    std::string op_name = op.name().substr(3) + "_" + std::to_string(idx);
    std::string func_name = Context::Global().NewName("fn_" + op_name);
    func_names_.try_emplace(op_name, func_name);
    return func_names_.at(op_name);
  }

  std::vector<std::string> OpGetInputNames(const ::ir::Operation& op) {
    std::vector<std::string> names;
    std::unordered_set<std::string> repeat;
    for (int i = 0; i < op.num_operands(); ++i) {
      auto value = op.operand(i);
      std::string name = std::to_string(std::hash<::ir::Value>()(value));
      if (repeat.count(name)) {
        continue;
      }
      repeat.insert(name);
      names.push_back(name);
    }
    return names;
  }

  std::vector<std::string> OpGetOutputNames(const ::ir::Operation& op) {
    std::vector<std::string> names;
    for (int i = 0; i < op.num_results(); ++i) {
      auto value = op.result(i);
      std::string name = std::to_string(std::hash<::ir::Value>()(value));
      names.push_back(std::move(name));
    }
    return names;
  }

 private:
  const ::ir::Program& program_;
  ir::Module::Builder m_builder_;
  std::unique_ptr<backends::Compiler> compiler_;
  Target target_;
  std::shared_ptr<Scope> scope_;
  std::unordered_map<std::string, std::string> func_names_;
};

std::shared_ptr<Scope> BuildScope(const Target& target,
                                  const ::ir::Program& program) {
  std::unordered_set<::ir::Value> visited;
  auto scope = std::make_shared<Scope>();

  auto create_var = [&](::ir::Value value) {
    if (visited.count(value) > 0) return;
    visited.emplace(value);

    std::string name = std::to_string(std::hash<::ir::Value>()(value));
    auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto* var = scope->Var<Tensor>(name);
    auto& tensor = absl::get<Tensor>(*var);
    // NOTE: can be replaced with phi::vectorized ?
    std::vector<Shape::dim_t> shape;
    for (auto i = 0; i < type_info.dims().size(); ++i) {
      shape.push_back(Shape::dim_t(type_info.dims()[i]));
    }
    tensor->Resize(Shape{shape});
    // TODO(Aurelius84): need convert this.
    tensor->set_type(common::Float(32));
  };

  for (auto it = program.block()->begin(); it != program.block()->end(); ++it) {
    // visit OpOprands
    for (auto i = 0; i < (*it)->num_operands(); ++i) {
      auto in_value = (*it)->operand(i);
      create_var(in_value);
    }

    for (auto i = 0; i < (*it)->num_results(); ++i) {
      auto out_value = (*it)->result(i);
      create_var(out_value);
    }
  }
  return scope;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
