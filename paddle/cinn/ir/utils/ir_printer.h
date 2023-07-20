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

#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/utils/ir_visitor.h"

namespace cinn {

namespace lang {
class LoweredFunc;
}  // namespace lang

namespace ir {
class Module;

struct IrPrinter : public IRVisitorRequireReImpl<void> {
  explicit IrPrinter(std::ostream &os) : os_(os) {}

  //! Emit an expression on the output stream.
  void Print(Expr e);
  //! Emit a expression list with , splitted.
  void Print(const std::vector<Expr> &exprs,
             const std::string &splitter = ", ");
  //! Emit a binary operator
  template <typename IRN>
  void PrintBinaryOp(const std::string &op, const BinaryOpNode<IRN> *x);

  //! Prefix the current line with `indent_` spaces.
  void DoIndent();
  //! Increase the indent size.
  void IncIndent();
  //! Decrease the indent size.
  void DecIndent();

  std::ostream &os() { return os_; }

#define __(op__) void Visit(const op__ *x) override;
  NODETY_FORALL(__)
#undef __

#define __(op__) virtual void Visit(const intrinsics::op__ *x);
  INTRINSIC_KIND_FOR_EACH(__)
#undef __

 private:
  std::ostream &os_;
  uint16_t indent_{};
  const int indent_unit{2};
};

std::ostream &operator<<(std::ostream &os, Expr a);
std::ostream &operator<<(std::ostream &os, const std::vector<Expr> &a);
std::ostream &operator<<(std::ostream &os, const Module &m);

template <typename IRN>
void IrPrinter::PrintBinaryOp(const std::string &op,
                              const BinaryOpNode<IRN> *x) {
  os_ << "(";
  Print(x->a());
  os_ << " " + op + " ";
  Print(x->b());
  os_ << ")";
}

}  // namespace ir
}  // namespace cinn
