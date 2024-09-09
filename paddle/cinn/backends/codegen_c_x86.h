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

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace backends {

/**
 * C code generation with X86 instruction or math library support.
 */
class CodeGenCX86 : public CodeGenC {
 public:
  //! The X86 CPU supports some following features. We use SSE or AVX to
  //! accelerate the basic operations if forloop is vectorized.
  enum class Feature : int {
    None = 0,
    SSE = 1,          //! support SSE instruction set.
    AVX256 = 1 << 1,  // ! support AVX256 instruction set.
    AVX512 = 1 << 2,  // ! support AVX512 instruction set.
    BLAS = 1 << 3,    // ! support BLAS library.
  };

  Feature feature{Feature::None};

  /**
   * constructor.
   * @param target The device.
   * @param features Features it supported.
   */
  CodeGenCX86(Target target, Feature feature)
      : CodeGenC(target), feature(feature) {}

 protected:
  void Visit(const ir::Add *op) override;
  void Visit(const ir::Sub *op) override;
  void Visit(const ir::Mul *op) override;
  void Visit(const ir::Div *op) override;
  void Visit(const ir::Mod *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::EQ *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::NE *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::LT *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::LE *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::GT *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::GE *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::And *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::Or *op) override { CodeGenC::Visit(op); }
  void Visit(const ir::Load *op) override;
  void Visit(const ir::Store *op) override;
  void Visit(const ir::Broadcast *op) override;
  void Visit(const ir::intrinsics::BuiltinIntrin *op);

  //! Check the features.
  // @{
  bool SupportsSSE() {
    return static_cast<int>(feature) & static_cast<int>(Feature::SSE);
  }
  bool SupportsAVX256() {
    return static_cast<int>(feature) & static_cast<int>(Feature::AVX256);
  }
  bool SupportsAVX512() {
    return static_cast<int>(feature) & static_cast<int>(Feature::AVX512);
  }
  bool SupportsBLAS() {
    return static_cast<int>(feature) & static_cast<int>(Feature::BLAS);
  }
  // @}

  //! Print (and prepare) a argument in vectorize type, for example:
  // 3. -> set1(3.)
  // a[i:j] -> load_ps(a+i)
  void PrintVecInputArgument(const Expr *op);
  //! The output argument, such as the destination for Load.
  void PrintVecOutputArgument(const Expr *op);

  template <typename Op>
  void PrintAbsAddr(const Op *op) {
    str_ += op->tensor.template As<ir::_Tensor_>()->name;
    str_ += " + ";

    auto index = op->index();
    auto *ramp_n = index.template As<ir::Ramp>();
    if (ramp_n) {
      PADDLE_ENFORCE_EQ(
          !ramp_n->base.template As<ir::Ramp>(),
          true,
          ::common::errors::InvalidArgument(
              "The base of a Ramp node should not be of Ramp type. "
              "Please ensure that the base is correctly set to a non-Ramp "
              "type."));
      IrPrinter::Visit(ramp_n->base);
    } else {
      IrPrinter::Visit(op->index());
    }
  }

  template <typename Op>
  void VisitBinaryOp(const Op *op, Expr a, Expr b, const std::string &op_repr);
};

template <typename Op>
void CodeGenCX86::VisitBinaryOp(const Op *op,
                                Expr a,
                                Expr b,
                                const std::string &op_repr) {
  PADDLE_ENFORCE_EQ(a.type(),
                    b.type(),
                    ::common::errors::InvalidArgument(
                        "The type of a and b should be the same."));

  // scalar.
  if (a.type().lanes() == 1) {
    CodeGenC::Visit(op);
    return;
  }

  // TODO(Superjomn) Consider support BLAS.
  int bits = a.type().bits() * a.type().lanes();
  if (SupportsAVX512() && bits == 512) {
    str_ += "cinn_avx512_";
    str_ += op_repr;
    str_ += "(";
    PrintVecInputArgument(&a);
    str_ += ", ";
    PrintVecInputArgument(&b);
    str_ += ")";
  } else if (SupportsAVX256() && bits == 256) {
    str_ += "cinn_avx256_";
    str_ += op_repr;
    str_ += "(";
    PrintVecInputArgument(&a);
    str_ += ", ";
    PrintVecInputArgument(&b);
    str_ += ")";
  } else {
    CodeGenC::Visit(op);
  }
}

}  // namespace backends
}  // namespace cinn
