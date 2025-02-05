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

#include <pybind11/pybind11.h>
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"

namespace paddle {
namespace pybind {
class PyIfOp : public dialect::IfOp {
 public:
  explicit PyIfOp(dialect::IfOp if_op);
  void UpdateOutput();
};

class PyWhileOp : public dialect::WhileOp {
 public:
  explicit PyWhileOp(dialect::WhileOp while_op);

  ///
  /// \brief Construct a new while_op to replace the original while_op. The
  /// input, output, and parameters of the new while_op no longer contain the
  /// variables that have not been modified in the loop. The size of the return
  /// value is equal to the output size of the original while_op, where the
  /// value of the read-only loop variable is the corresponding operand of the
  /// original while_op, and the value of the non-read-only loop variable is the
  /// corresponding output of the new while_op,
  ///
  std::vector<pir::Value> OptimizeUpdate();
  void AddExtraInput(const pir::Value &value);

 private:
  std::vector<pir::Value> extra_inputs_;
};

void BindControlFlowApi(pybind11::module *m);
}  // namespace pybind
}  // namespace paddle
