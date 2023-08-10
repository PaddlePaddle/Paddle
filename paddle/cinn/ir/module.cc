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

#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/optimize.h"

namespace cinn {
namespace ir {

void Module::Builder::AddFunction(const ir::LoweredFunc &func, bool use_optim) {
  if (use_optim) {
    auto tmp_func = optim::IRCopy(func);
    optim::Simplify(&(tmp_func->body));
    optim::SimplifyForLoops(&(tmp_func->body));
    optim::SimplifyBlocks(&(tmp_func->body));
    tmp_func->body = optim::Optimize(tmp_func->body, module_->target);
    module_->functions.emplace_back(std::move(func));
  } else {
    module_->functions.push_back(func);
  }
}

void Module::Builder::AddBuffer(ir::Buffer buffer) {
  CHECK(buffer->target.defined())
      << "buffer [" << buffer->name << "]'s target is undefined";
  if (std::find_if(
          module_->buffers.begin(), module_->buffers.end(), [&](const Expr &x) {
            return x.as_buffer()->name == buffer->name;
          }) == std::end(module_->buffers)) {
    module_->buffers.push_back(buffer);
    if (module_->target.arch == Target::Arch::X86) {
      module_->buffers.back().as_buffer()->data_alignment = 32;
    }
  }
}

void Module::Builder::Clear() {
  module_->buffers.clear();
  module_->functions.clear();
  module_->submodules.clear();
}

Module Module::Builder::Build() {
  if (module_->functions.empty()) {
    VLOG(1) << "Module has no functions";
  }

  auto res = ir::Module(module_.get());

  return optim::Optimize(res, module_->target);
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
