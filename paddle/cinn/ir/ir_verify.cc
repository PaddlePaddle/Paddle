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

#include "cinn/ir/ir_verify.h"

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

namespace cinn::ir {

struct IrVerifyVisitor : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

#define __(op__)                                    \
  void Visit(const op__ *op, Expr *expr) override { \
    op->Verify();                                   \
    IRMutator::Visit(op, expr);                     \
  }
  NODETY_FORALL(__)
#undef __
};

void IrVerify(Expr e) {
  IrVerifyVisitor visitor;
  visitor.Visit(&e, &e);
}

}  // namespace cinn::ir
