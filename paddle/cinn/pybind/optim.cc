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

#include "cinn/common/ir_util.h"
#include "cinn/common/object.h"
#include "cinn/common/shared.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/pybind/bind.h"
#include "cinn/pybind/bind_utils.h"
#include "cinn/utils/string.h"

namespace py = pybind11;

namespace cinn::pybind {

using optim::Simplify;

namespace {

void BindSimplify(py::module* m) {
  m->def(
      "simplify",
      [](const Expr& expr) -> Expr {
        auto copied = optim::IRCopy(expr);
        Simplify(&copied);
        return copied;
      },
      py::arg("expr"));

  m->def("ir_copy", py::overload_cast<Expr>(&optim::IRCopy));
}

}  // namespace

void BindOptim(py::module* m) { BindSimplify(m); }

}  // namespace cinn::pybind
