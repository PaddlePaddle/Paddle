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

#include "paddle/cinn/backends/codegen_c_x86.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace backends {

void CodeGenCX86::Visit(const ir::Add *op) {
  VisitBinaryOp(op, op->a(), op->b(), "add");
}
void CodeGenCX86::Visit(const ir::Sub *op) {
  VisitBinaryOp(op, op->a(), op->b(), "sub");
}
void CodeGenCX86::Visit(const ir::Mul *op) {
  VisitBinaryOp(op, op->a(), op->b(), "mul");
}
void CodeGenCX86::Visit(const ir::Div *op) {
  VisitBinaryOp(op, op->a(), op->b(), "div");
}

void CodeGenCX86::Visit(const ir::Load *op) {
  Expr dense_strided_ramp = detail::StridedRampBase(op->index(), 1);
  if (dense_strided_ramp.defined()) {  // Loading a continuous Ramp address.
    PADDLE_ENFORCE_EQ(
        op->type().is_vector(),
        true,
        ::common::errors::InvalidArgument(
            "The operation type is expected to be a vector, but it is not. "
            "Please check the operation type and ensure it is correctly set to "
            "a vector."));

    int bits = op->type().bits() * op->type().lanes();
    if (SupportsAVX512() && bits == 512) {
      str_ += "cinn_avx512_load(";
      PrintAbsAddr(op);
      str_ += ")";
    } else if (SupportsAVX256() && bits == 256) {
      str_ += "cinn_avx256_load(";
      PrintAbsAddr(op);
      str_ += ")";
    } else {
      CodeGenC::Visit(op);
    }
  } else {
    CodeGenC::Visit(op);
  }
}

void CodeGenCX86::Visit(const ir::Broadcast *op) {
  PADDLE_ENFORCE_GT(
      op->type().lanes(),
      1,
      ::common::errors::InvalidArgument(
          "The lanes of the broadcast op should be greater than 1."));
  int bits = op->type().bits() * op->type().lanes();

  if (SupportsAVX512() && bits == 512) {
    str_ += "cinn_avx512_set1(";
    PrintCastExpr(op->value.type().ElementOf(), op->value);
    str_ += ")";
  } else if (SupportsAVX256() && bits == 256) {
    str_ += "cinn_avx256_set1(";
    PrintCastExpr(op->value.type().ElementOf(), op->value);
    str_ += ")";
  } else {
    CodeGenC::Visit(op);
  }
}

void CodeGenCX86::Visit(const ir::Store *op) {
  if (op->type().lanes() == 1) {
    CodeGenC::Visit(op);
    return;
  }

  int bits = op->type().bits() * op->type().lanes();
  if (SupportsAVX512() && bits == 512) {
    str_ += "cinn_avx512_store(";
    PrintAbsAddr(op);
    str_ += ", ";
    IrPrinter::Visit(op->value);
    str_ += ")";
  } else if (SupportsAVX256() && bits == 256) {
    str_ += "cinn_avx256_store(";
    PrintAbsAddr(op);
    str_ += ", ";
    IrPrinter::Visit(op->value);
    str_ += ")";
  } else {
    CodeGenC::Visit(op);
  }
}

void CodeGenCX86::PrintVecInputArgument(const Expr *op) {
  int bits = op->type().bits() * op->type().lanes();
  auto *broadcast_n = op->As<ir::Broadcast>();

  if (op->type().lanes() == 1 || broadcast_n) {
    Expr value = op->type().lanes() == 1 ? *op : broadcast_n->value;

    if (SupportsAVX512()) {
      str_ += "cinn_avx512_set1(";
      IrPrinter::Visit(value);
      str_ += ")";
    } else if (SupportsAVX256()) {
      str_ += "cinn_avx256_set1(";
      IrPrinter::Visit(value);
      str_ += ")";
    } else {
      CINN_NOT_IMPLEMENTED
    }
  } else {
    IrPrinter::Visit(*op);
  }
}

void CodeGenCX86::Visit(const ir::intrinsics::BuiltinIntrin *op) {
  if (op->type().lanes() == 1) {
    CodeGenC::Visit(op);
    return;
  }
  int bits = op->type().bits() * op->type().lanes();
  if (SupportsAVX512() && bits == 512) {
    str_ += "cinn_avx512_";
    str_ += op->name;
    str_ += "(";
    if (!op->args.empty()) {
      for (int i = 0; i < op->args.size() - 1; i++) {
        PrintVecInputArgument(&op->args[i]);
        str_ += ", ";
      }
      IrPrinter::Visit(op->args.back());
    }
    str_ += ")";
  } else if (SupportsAVX256() && bits == 256) {
    str_ += "cinn_avx256_";
    str_ += op->name;
    str_ += "(";
    if (!op->args.empty()) {
      for (int i = 0; i < op->args.size() - 1; i++) {
        PrintVecInputArgument(&op->args[i]);
        str_ += ", ";
      }
      PrintVecInputArgument(&op->args.back());
    }
    str_ += ")";
  } else if (bits == 128) {
    str_ += "cinn_avx128_";
    str_ += op->name;
    str_ += "(";
    if (!op->args.empty()) {
      for (int i = 0; i < op->args.size() - 1; i++) {
        PrintVecInputArgument(&op->args[i]);
        str_ += ", ";
      }
      PrintVecInputArgument(&op->args.back());
    }
    str_ += ")";
  } else {
    CodeGenC::Visit(op);
  }
}

}  // namespace backends
}  // namespace cinn
