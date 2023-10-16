// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <ostream>
#include <string>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/utils/flags.h"

namespace cinn {

namespace ir {
class IrPythonPrinter : public IRVisitorRequireReImpl<void> {
 public:
  explicit IrPythonPrinter(std::ostream &os) : os_(os), str_("") {}
  //! Emit an expression on the output stream.
  void Print(const Expr &e);
  //! Emit a expression list with , splitted.
  void Print(const std::vector<Expr> &exprs,
             const std::string &splitter = ", ");

  //! Prefix the current line with `indent_` spaces.
  void DoIndent();
  //! Increase the indent size.
  void IncIndent();
  //! Decrease the indent size.
  void DecIndent();
  template <typename IRN>
  void PrintBinaryOp(const std::string &op, const BinaryOpNode<IRN> *x);
  void Visit(const Expr &x) { IRVisitorRequireReImpl::Visit(&x); }
  void Visit(const std::vector<Expr> &exprs,
             const std::string &splitter = ", ") {
    for (std::size_t i = 0; !exprs.empty() && i + 1 < exprs.size(); i++) {
      Visit(exprs[i]);
      str_ += splitter;
    }
    if (!exprs.empty()) Visit(exprs.back());
  }

#define __DEFINE_VISIT(op__) void Visit(const ir::op__ *op) override;
  NODETY_FORALL(__DEFINE_VISIT)
#undef __DEFINE_VISIT

#define __DEFINE_VISIT(op__) void Visit(const ir::intrinsics::op__ *op);
  INTRINSIC_KIND_FOR_EACH(__DEFINE_VISIT)
#undef __DEFINE_VISIT

  void Print(const common::Type &);
  void PrintShape(const std::vector<Expr> &shape);

 private:
  std::string str_;
  std::ostream &os_;
  uint16_t indent_{};
  const int indent_unit{4};
};
template <typename IRN>
void IrPythonPrinter::PrintBinaryOp(const std::string &op,
                                    const BinaryOpNode<IRN> *x) {
  str_ += "(";
  Visit(x->a());
  str_ += " ";
  str_ += op;
  str_ += " ";
  Visit(x->b());
  str_ += ")";
}

}  // namespace ir
}  // namespace cinn
