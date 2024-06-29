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

#include <list>
#include <ostream>
#include <unordered_map>

#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/parameter.h"

namespace pir {

class IrContext;
///
/// \brief Program is an abstraction of model structure, divided into
/// computational graphs and weights. At the current stage, a computational
/// graph is represented in the form of a list<Operation *>. Todo: In the
/// future, detailed design of control flow operators will be carried out, and
/// concepts such as basic blocks, closures, and functions will be introduced to
/// continuously improve Program's ability to represent computational graphs.
///
class IR_API Program {
 public:
  using ParameterMap =
      std::unordered_map<std::string, std::shared_ptr<Parameter>>;
  explicit Program(IrContext* context);
  Program(Program&&) = delete;
  Program(const Program& program) = delete;
  Program& operator=(const Program&) = delete;
  Program& operator=(Program&&);
  ~Program();
  size_t parameters_num() const { return parameters_.size(); }

  ModuleOp module_op() const { return module_; }

  void Print(std::ostream& os) const;

  static std::unique_ptr<Program> Parse(std::istream& is, IrContext* ctx);

  std::shared_ptr<Program> Clone(IrMapping& ir_mapping) const;  // NOLINT

  void CopyToBlock(IrMapping& ir_mapping, Block* insert_block) const;  // NOLINT
  Block* block() { return &module_.block(); }
  const Block* block() const { return &module_op().block(); }

  uint32_t num_ops() { return block()->num_ops(); }
  Parameter* GetParameter(const std::string& name) const;
  void SetParameter(const std::string& name,
                    std::shared_ptr<Parameter> parameter);

  ParameterMap& parameters() { return parameters_; }
  void set_parameters(const ParameterMap& parameters) {
    parameters_ = parameters;
  }

  uint64_t id() const { return id_; }

 private:
  // computation graph
  ModuleOp module_;
  // unique in current process, "almost" unique between processes.
  uint64_t id_;
  // weight
  ParameterMap parameters_;
};

IR_API std::ostream& operator<<(std::ostream& os, const Program& prog);

}  // namespace pir
