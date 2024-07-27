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

#include "paddle/cinn/ir/module.h"

#include <memory>

#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/optimize.h"

namespace cinn {
namespace ir {

void Module::Builder::AddFunction(ir::LoweredFunc func) {
  optim::Simplify(&(func->body));
  optim::SimplifyForLoops(&(func->body));
  optim::SimplifyBlocks(&(func->body));
  func->body = optim::Optimize(func->body, module_->target);
  module_->functions.push_back(func);
}

void Module::Builder::AddFunctionWithoutOptim(const ir::LoweredFunc &func) {
  module_->functions.push_back(func);
}

std::optional<int> GetDataAlignmentImpl(common::UnknownArch arch) {
  return std::nullopt;
}

std::optional<int> GetDataAlignmentImpl(common::X86Arch arch) { return 32; }

std::optional<int> GetDataAlignmentImpl(common::ARMArch arch) {
  return std::nullopt;
}

std::optional<int> GetDataAlignmentImpl(common::NVGPUArch) {
  return std::nullopt;
}

std::optional<int> GetDataAlignmentImpl(common::HygonDCUArchHIP arch) {
  return std::nullopt;
}

std::optional<int> GetDataAlignment(common::Arch arch) {
  return std::visit([](const auto &impl) { return GetDataAlignmentImpl(impl); },
                    arch.variant());
}

void Module::Builder::AddBuffer(ir::Buffer buffer) {
  CHECK(buffer->target.defined())
      << "buffer [" << buffer->name << "]'s target is undefined";
  if (std::find_if(
          module_->buffers.begin(), module_->buffers.end(), [&](const Expr &x) {
            return x.as_buffer()->name == buffer->name;
          }) == std::end(module_->buffers)) {
    module_->buffers.push_back(buffer);
    if (auto alignment = GetDataAlignment(module_->target.arch)) {
      module_->buffers.back().as_buffer()->data_alignment = alignment.value();
    }
  }
}

void Module::Builder::AddPredicate(ir::Expr predicate) {
  module_->predicates.push_back(predicate);
}

void Module::Builder::AddPriority(int priority) {
  module_->priorities.push_back(priority);
}

void Module::Builder::SetInferShapeFunc(ir::Expr infer_shape_func) {
  module_->infer_shape_func = infer_shape_func;
}

void Module::Builder::Clear() {
  module_->buffers.clear();
  module_->functions.clear();
  module_->submodules.clear();
  module_->predicates.clear();
}

common::Arch Module::Builder::GetTargetArch() { return module_->target.arch; }

Module Module::Builder::Build() {
  if (module_->functions.empty()) {
    VLOG(1) << "Module has no functions";
  }

  auto res = ir::Module(module_.get());

  res = optim::Optimize(res, module_->target);
  return res;
}

ir::_Module_ *Module::self() { return p_->as<ir::_Module_>(); }
const ir::_Module_ *Module::self() const { return p_->as<ir::_Module_>(); }

const Target &Module::target() const { return self()->target; }

const std::string &Module::name() const { return self()->name; }

std::vector<ir::Buffer> Module::buffers() const {
  std::vector<ir::Buffer> buffers;
  for (auto &buffer : self()->buffers) {
    buffers.emplace_back(buffer.as_buffer_ref());
  }
  return buffers;
}

std::vector<ir::LoweredFunc> Module::functions() const {
  std::vector<ir::LoweredFunc> functions;
  for (auto &x : self()->functions) {
    functions.emplace_back(x.as_lowered_func_ref());
  }
  return functions;
}

std::vector<Module> Module::submodules() const {
  std::vector<ir::Module> modules;
  for (auto &x : self()->submodules) {
    modules.push_back(x.as_module_ref());
  }
  return modules;
}

void Module::Compile(const backends::Outputs &outputs) const {}

Module::operator Expr() const { return Expr(ptr()); }

}  // namespace ir
}  // namespace cinn
