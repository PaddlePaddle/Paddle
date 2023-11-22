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

#include "paddle/cinn/hlir/framework/pir_compiler.h"

#include <absl/types/variant.h>
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/core/builtin_type.h"

namespace cinn {
namespace hlir {
namespace framework {

std::vector<pir::CUDAJITInfo> PirCompiler::Build(
    const std::vector<pir::GroupPtr>& groups) {
  std::vector<pir::CUDAJITInfo> jit_infos;
  m_builder_.Clear();

  std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  for (int i = 0; i < groups.size(); ++i) {
    lowered_funcs.emplace_back(op_lowerer_.Lower(groups[i]));
  }

  for (auto&& lowered_func : lowered_funcs) {
    ProcessFunction(lowered_func);
  }

  compiler_ = backends::Compiler::Create(target_);
  auto build_module = m_builder_.Build();
  compiler_->Build(build_module, "");

  auto fn_ptrs = compiler_->GetFnPtr();

  for (int idx = 0; idx < groups.size(); ++idx) {
    pir::CUDAJITInfo jit_info;
    jit_info.fn_ptr = fn_ptrs[idx];
    jit_info.compiler = reinterpret_cast<void*>(compiler_.get());

    lowered_funcs[idx][0]->cuda_axis_info.CopyBlockDimsTo(
        &(jit_info.block_dims));

    lowered_funcs[idx][0]->cuda_axis_info.CopyGridDimsTo(&(jit_info.grid_dims));

    jit_infos.push_back(jit_info);
  }

  return jit_infos;
}

void PirCompiler::ProcessFunction(
    const std::vector<ir::LoweredFunc>& lowered_funcs) {
  for (auto&& func : lowered_funcs) {
    m_builder_.AddFunction(func);
  }
}

std::shared_ptr<Scope> BuildScope(const Target& target,
                                  const ::pir::Program& program) {
  std::unordered_set<::pir::Value> visited;
  auto scope = std::make_shared<Scope>();

  auto create_var = [&](::pir::Value value) {
    if (!(value) || !(value.type())) {
      return;
    }
    if (visited.count(value) > 0) return;
    visited.emplace(value);

    std::string name = pir::CompatibleInfo::ValueName(value);
    auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto* var = scope->Var<Tensor>(name);
    auto& tensor = absl::get<Tensor>(*var);

    std::vector<Shape::dim_t> shape;
    for (auto i = 0; i < type_info.dims().size(); ++i) {
      shape.push_back(Shape::dim_t(type_info.dims()[i]));
    }
    tensor->Resize(Shape{shape});
    tensor->set_type(pir::CompatibleInfo::ConvertIRType(type_info.dtype()));
  };

  for (auto& op : *program.block()) {
    for (auto oprand : op.operands()) {
      create_var(oprand.source());
    }

    for (auto result : op.results()) {
      create_var(result);
    }
  }
  return scope;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
