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
#include <unordered_map>

#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/operation.h"
#include "paddle/ir/parameter.h"

namespace ir {
///
/// \brief Program is an abstraction of model structure, divided into
/// computational graphs and weights. At the current stage, a computational
/// graph is represented in the form of a list<Operation *>. Todo: In the
/// future, detailed design of control flow operators will be carried out, and
/// concepts such as basic blocks, closures, and functions will be introduced to
/// continuously improve Program's ability to represent computational graphs.
///
class Program {
 public:
  ~Program();

  std::list<Operation*> ops() const { return ops_; }

  size_t parameters_num() const { return parameters_.size(); }

  ///
  /// \brief Insert the Operation* constructed by Operation::create(...) into
  /// this Program. NOTE: At this time, the memory management permission of
  /// Operation* will be owned by this Program. The user does not need to call
  /// Operation::destroy() manually
  ///
  void InsertOp(Operation* op);

  Parameter* GetParameter(std::string name) const;

  void SetParameter(std::string name, std::unique_ptr<Parameter>&& parameter);

 private:
  std::list<Operation*> ops_;  // owned

  std::unordered_map<std::string, std::unique_ptr<Parameter>> parameters_;
};

std::ostream& operator<<(std::ostream& os, Program& program);

}  // namespace ir
