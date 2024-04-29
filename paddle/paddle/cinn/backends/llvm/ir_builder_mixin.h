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

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#include <utility>

namespace cinn {
namespace backends {
template <typename Derived>
class IrBuilderMixin {
 protected:
  template <typename... Args>
  decltype(auto) BinOp(Args &&...args) {
    return mixin_builder()->CreateBinOp(std::forward<Args>(args)...);
  }

  /// \brief +
  template <typename... Args>
  decltype(auto) Add(Args &&...args) {
    return mixin_builder()->CreateAdd(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FAdd(Args &&...args) {
    return mixin_builder()->CreateFAdd(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) NSWAdd(Args &&...args) {
    return mixin_builder()->CreateNSWAdd(std::forward<Args>(args)...);
  }

  /// \brief -
  template <typename... Args>
  decltype(auto) Sub(Args &&...args) {
    return mixin_builder()->CreateSub(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FSub(Args &&...args) {
    return mixin_builder()->CreateFSub(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) NSWSub(Args &&...args) {
    return mixin_builder()->CreateNSWSub(std::forward<Args>(args)...);
  }

  /// \brief *
  template <typename... Args>
  decltype(auto) Mul(Args &&...args) {
    return mixin_builder()->CreateMul(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FMul(Args &&...args) {
    return mixin_builder()->CreateFMul(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) NSWMul(Args &&...args) {
    return mixin_builder()->CreateNSWMul(std::forward<Args>(args)...);
  }

  /// \brief /
  template <typename... Args>
  decltype(auto) SDiv(Args &&...args) {
    return mixin_builder()->CreateSDiv(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) UDiv(Args &&...args) {
    return mixin_builder()->CreateUDiv(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FDiv(Args &&...args) {
    return mixin_builder()->CreateFDiv(std::forward<Args>(args)...);
  }

  /// \brief %
  template <typename... Args>
  decltype(auto) SRem(Args &&...args) {
    return mixin_builder()->CreateSRem(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) URem(Args &&...args) {
    return mixin_builder()->CreateURem(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FRem(Args &&...args) {
    return mixin_builder()->CreateFRem(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) And(Args &&...args) {
    return mixin_builder()->CreateAnd(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) Or(Args &&...args) {
    return mixin_builder()->CreateOr(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) Not(Args &&...args) {
    return mixin_builder()->CreateNot(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) Neg(Args &&...args) {
    return mixin_builder()->CreateNeg(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FNeg(Args &&...args) {
    return mixin_builder()->CreateFNeg(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) ICmpEQ(Args &&...args) {
    return mixin_builder()->CreateICmpEQ(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FCmpOEQ(Args &&...args) {
    return mixin_builder()->CreateFCmpOEQ(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FCmpUEQ(Args &&...args) {
    return mixin_builder()->CreateFCmpUEQ(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) ICmpNE(Args &&...args) {
    return mixin_builder()->CreateICmpNE(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FCmpONE(Args &&...args) {
    return mixin_builder()->CreateFCmpONE(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FCmpUNE(Args &&...args) {
    return mixin_builder()->CreateFCmpUNE(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) ICmpULE(Args &&...args) {
    return mixin_builder()->CreateICmpULE(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FCmpOLE(Args &&...args) {
    return mixin_builder()->CreateFCmpOLE(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) ICmpULT(Args &&...args) {
    return mixin_builder()->CreateICmpULT(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) ICmpSLT(Args &&...args) {
    return mixin_builder()->CreateICmpSLT(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FCmpOLT(Args &&...args) {
    return mixin_builder()->CreateFCmpOLT(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) ICmpUGE(Args &&...args) {
    return mixin_builder()->CreateICmpUGE(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) ICmpSGE(Args &&...args) {
    return mixin_builder()->CreateICmpSGE(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FCmpOGE(Args &&...args) {
    return mixin_builder()->CreateFCmpOGE(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) ICmpUGT(Args &&...args) {
    return mixin_builder()->CreateICmpUGT(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) ICmpSGT(Args &&...args) {
    return mixin_builder()->CreateICmpSGT(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FCmpOGT(Args &&...args) {
    return mixin_builder()->CreateFCmpOGT(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) BitCast(Args &&...args) {
    return mixin_builder()->CreateBitCast(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) IntCast(Args &&...args) {
    return mixin_builder()->CreateIntCast(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FPCast(Args &&...args) {
    return mixin_builder()->CreateFPCast(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) PointerCast(Args &&...args) {
    return mixin_builder()->CreatePointerCast(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) FPToSI(Args &&...args) {
    return mixin_builder()->CreateFPToSI(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) FPToUI(Args &&...args) {
    return mixin_builder()->CreateFPToUI(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) SIToFP(Args &&...args) {
    return mixin_builder()->CreateSIToFP(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) UIToFP(Args &&...args) {
    return mixin_builder()->CreateUIToFP(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) Select(Args &&...args) {
    return mixin_builder()->CreateSelect(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) Br(Args &&...args) {
    return mixin_builder()->CreateBr(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) CondBr(Args &&...args) {
    return mixin_builder()->CreateCondBr(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) Alloca(Args &&...args) {
    return mixin_builder()->CreateAlloca(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) Load(Args &&...args) {
    return mixin_builder()->CreateLoad(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) AlignedLoad(Args &&...args) {
    return mixin_builder()->CreateAlignedLoad(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) Store(Args &&...args) {
    return mixin_builder()->CreateStore(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) AlignedStore(Args &&...args) {
    return mixin_builder()->CreateAlignedStore(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) Call(Args &&...args) {
    return mixin_builder()->CreateCall(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) RetVoid(Args &&...args) {
    return mixin_builder()->CreateRetVoid(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) GEP(Args &&...args) {
    return mixin_builder()->CreateGEP(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) InBoundsGEP(Args &&...args) {
    return mixin_builder()->CreateInBoundsGEP(std::forward<Args>(args)...);
  }
  template <typename... Args>
  decltype(auto) PHI(Args &&...args) {
    return mixin_builder()->CreatePHI(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) InsertValue(Args &&...args) {
    return mixin_builder()->CreateInsertValue(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) ExtractValue(Args &&...args) {
    return mixin_builder()->CreateExtractValue(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) InsertElement(Args &&...args) {
    return mixin_builder()->CreateInsertElement(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) ShuffleVector(Args &&...args) {
    return mixin_builder()->CreateShuffleVector(std::forward<Args>(args)...);
  }

 private:
  llvm::IRBuilder<> *mixin_builder() {
    return static_cast<Derived *>(this)->b();
  }
};
}  // namespace backends
}  // namespace cinn
