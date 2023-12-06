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

#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/parameter.h"

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
      std::unordered_map<std::string, std::unique_ptr<Parameter>>;
  TEST_API explicit Program(IrContext* context);
  Program(Program&&) = delete;
  Program(const Program& program) = delete;
  Program& operator=(const Program&) = delete;
  Program& operator=(Program&&);
  TEST_API ~Program();
  size_t parameters_num() const { return parameters_.size(); }

  ModuleOp module_op() const { return module_; }

  void Print(std::ostream& os) const;

  static std::unique_ptr<Program> Parse(std::istream& is, IrContext* ctx);

  Block* block() { return &module_.block(); }
  const Block* block() const { return &module_op().block(); }

  Parameter* GetParameter(const std::string& name) const;
  void SetParameter(const std::string& name,
                    std::unique_ptr<Parameter>&& parameter);

  ParameterMap& parameters() { return parameters_; }
  void set_parameters(ParameterMap&& parameters) {
    parameters_ = std::move(parameters);
  }

 private:
  // computation graph
  ModuleOp module_;
  // weight
  ParameterMap parameters_;
};

std::ostream& operator<<(std::ostream& os, const Program& prog);

}  // namespace pir
