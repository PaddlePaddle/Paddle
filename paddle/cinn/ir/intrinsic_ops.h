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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/Casting.h>

#include <string>

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/ir.h"

//! This file defines some intrinsic IR nodes, this is similar to the MLIR
//! operations, we try to expose some underlying opaque operations to IR system
//! to help more intuitive codegen.

namespace cinn::ir {

// clang-format off
#define INTRINSIC_KIND_FOR_EACH(macro__)                 \
  macro__(BufferGetDataHandle)                           \
  macro__(BufferGetDataConstHandle)                      \
  macro__(PodValueToX)                                   \
  macro__(BufferCreate)                                  \
  macro__(GetAddr)                                       \
  macro__(ArgsConstruct)                                 \
  macro__(BuiltinIntrin)
// clang-format on

enum class IntrinsicKind {
// All the intrinsics should registered here.
#define __(x__) k##x__,
  INTRINSIC_KIND_FOR_EACH(__)
#undef __
};

class IntrinsicOp : public IrNode {
 public:
  IntrinsicOp(IntrinsicKind kind,
              llvm::ArrayRef<Type> input_types,
              llvm::ArrayRef<Type> output_types)
      : kind_(kind),
        input_types_(input_types.begin(), input_types.end()),
        output_types_(output_types.begin(), output_types.end()) {}

  const Type& GetInputType(int offset) const;
  const Type& GetOutputType(int offset) const;

  void AddInputType(const Type& type) { input_types_.push_back(type); }
  void AddOutputType(const Type& type) { output_types_.push_back(type); }

  const llvm::SmallVectorImpl<Type>& input_types() const {
    return input_types_;
  }
  const llvm::SmallVectorImpl<Type>& output_types() const {
    return input_types_;
  }

  //! Verify the \p input_types and \p output_types matches the signature of
  //! this operation.
  void Verify(llvm::ArrayRef<Type> input_types,
              llvm::ArrayRef<Type> output_types) const;
  void Verify(llvm::ArrayRef<Expr> inputs, llvm::ArrayRef<Expr> outputs) const;
  void Verify(llvm::ArrayRef<Expr> inputs) const;

  void Verify() const override {}

  const char* type_info() const override;

  IntrinsicKind getKind() const { return kind_; }

  IrNodeTy node_type() const override { return _node_type_; }

  static constexpr IrNodeTy _node_type_{IrNodeTy::IntrinsicOp};

 protected:
  llvm::SmallVector<Type, 4> input_types_;
  llvm::SmallVector<Type, 4> output_types_;
  const IntrinsicKind kind_;
};

namespace intrinsics {

/**
 * The operation to get the memory address from cinn_buffer_t.
 */
struct BufferGetDataHandle : public IntrinsicOp {
  // signature: (cinn_buffer_t*) -> (void*)
  BufferGetDataHandle()
      : IntrinsicOp(IntrinsicKind::kBufferGetDataHandle,
                    {type_of<cinn_buffer_t*>()},
                    {type_of<void*>()}) {}

  static Expr Make(Expr buffer);

  static bool classof(const IntrinsicOp* s) {
    return s->getKind() == IntrinsicKind::kBufferGetDataHandle;
  }

  Expr buffer;
};

/**
 * The operation to get the memory address from cinn_buffer_t.
 */
struct BufferGetDataConstHandle : public IntrinsicOp {
  // signature: (cinn_buffer_t*) -> (const void*)
  BufferGetDataConstHandle()
      : IntrinsicOp(IntrinsicKind::kBufferGetDataConstHandle,
                    {type_of<const cinn_buffer_t*>()},
                    {type_of<void*>()}) {}

  static Expr Make(Expr buffer);

  static bool classof(const IntrinsicOp* s) {
    return s->getKind() == IntrinsicKind::kBufferGetDataConstHandle;
  }

  Expr buffer;
};

/**
 * The operation to represent the helper methods:
 * - cinn_pod_value_to_float
 * - cinn_pod_value_to_duoble
 * - cinn_pod_value_to_int64
 * - cinn_pod_value_to_int32
 * - cinn_pod_value_to_void_p
 * - cinn_pod_value_to_buffer_p
 */
struct PodValueToX : public IntrinsicOp {
  // signature: (cinn_pod_value_t*) -> (X), X is some pod type.
  PodValueToX()
      : IntrinsicOp(
            IntrinsicKind::kPodValueToX, {type_of<cinn_pod_value_t*>()}, {}) {}

  static Expr Make(Expr pod_value_ptr, const Type& type);

  static bool classof(const IntrinsicOp* s) {
    return s->getKind() == IntrinsicKind::kPodValueToX;
  }

  Expr pod_value_ptr;
};

/**
 * The operation to create a buffer.
 */
struct BufferCreate : public IntrinsicOp {
  // signature: (cinn_buffer_t*) -> void
  BufferCreate()
      : IntrinsicOp(
            IntrinsicKind::kBufferCreate, {type_of<cinn_buffer_t*>()}, {}) {}

  static Expr Make(Expr buffer);

  static bool classof(const IntrinsicOp* s) {
    return s->getKind() == IntrinsicKind::kBufferCreate;
  }

  Expr buffer;
};

/**
 * The operation to get the address of a data.
 */
struct GetAddr : public IntrinsicOp {
  // signature: (X) -> (X*)
  GetAddr() : IntrinsicOp(IntrinsicKind::kGetAddr, {}, {}) {}

  static Expr Make(Expr data);

  static bool classof(const IntrinsicOp* s) {
    return s->getKind() == IntrinsicKind::kGetAddr;
  }

  Expr data;
};

/**
 * The operation to construct a cinn_pod_value_t*
 */
struct ArgsConstruct : public IntrinsicOp {
  ArgsConstruct() : IntrinsicOp(IntrinsicKind::kArgsConstruct, {}, {}) {}

  static Expr Make(Var var, llvm::ArrayRef<Expr> args);

  static bool classof(const IntrinsicOp* s) {
    return s->getKind() == IntrinsicKind::kArgsConstruct;
  }

  Var var;
  llvm::SmallVector<Expr, 4> args;
};

/**
 * The llvm intrinsic op
 */
struct BuiltinIntrin : public IntrinsicOp {
  BuiltinIntrin() : IntrinsicOp(IntrinsicKind::kBuiltinIntrin, {}, {}) {}

  static Expr Make(const std::string& name,
                   llvm::ArrayRef<Expr> args,
                   llvm::Intrinsic::ID id,
                   int64_t arg_nums,
                   const Type& type);

  static bool classof(const IntrinsicOp* s) {
    return s->getKind() == IntrinsicKind::kBuiltinIntrin;
  }

  std::string name;
  llvm::SmallVector<Expr, 4> args;
  llvm::Intrinsic::ID id;
  int64_t arg_nums;
};

}  // namespace intrinsics

}  // namespace cinn::ir
