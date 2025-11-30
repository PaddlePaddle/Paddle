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

#pragma once
#include <string>
#include <vector>

#include "paddle/cinn/backends/outputs.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/lang/buffer.h"

namespace cinn {

namespace backends {
class CodeGenC;
}  // namespace backends

namespace ir {

/**
 * Content of a module.
 */
struct _Module_ : public IrNode {
  std::string name;
  Target target;
  std::vector<Expr> buffers;
  std::vector<LoweredFunc> functions;
  std::vector<Module> submodules;
  std::vector<Expr> predicates;
  std::vector<int> priorities;
  LoweredFunc infer_shape_func;

  static ir::Module Make(const std::string& name, Target target);

  void Verify() const override {}

  IrNodeTy node_type() const override { return _node_type_; }

  static const IrNodeTy _node_type_ = IrNodeTy::Module;
};

/**
 * Module represents IR containing lowered function definitions and buffers.
 */
class Module : public ir::IrNodeRef {
 public:
  struct Builder {
    Builder(const std::string& name, const Target& target)
        : module_(cinn::common::make_shared<ir::_Module_>()) {
      module_->name = name;
      module_->target = target;
    }

    void AddFunction(ir::LoweredFunc func);
    void AddFunctionWithoutOptim(const ir::LoweredFunc& func);
    void AddBuffer(ir::Buffer buffer);
    void AddPredicate(ir::Expr predicate);
    void AddPriority(int priority);
    void SetInferShapeFunc(ir::LoweredFunc infer_shape_func);
    void Clear();
    common::Arch GetTargetArch();

    Module Build();

   private:
    Shared<ir::_Module_> module_;
  };

  //! Get the target of this module.
  const Target& target() const;

  //! Get the name of the module.
  const std::string& name() const;

  //! The members in the module.
  // @{
  std::vector<ir::Buffer> buffers() const;
  const std::vector<ir::LoweredFunc>& functions() const;
  const std::vector<Module>& submodules() const;
  // @}

  //! Compile a module to some outputs.
  void Compile(const backends::Outputs& outputs) const;

  ir::_Module_* self();
  const ir::_Module_* self() const;

  ir::_Module_* operator->() { return self(); }
  const ir::_Module_* operator->() const { return self(); }

 protected:
  Module(const std::string& name, const Target& target);

  explicit Module(ir::IrNode* n) : ir::IrNodeRef(n) {}

  friend class Module::Builder;
  friend class backends::CodeGenC;
  friend class ::cinn::ir::Expr;
  friend class ::cinn::ir::_Module_;
};

}  // namespace ir
}  // namespace cinn
