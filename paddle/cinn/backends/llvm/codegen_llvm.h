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

#include <absl/container/flat_hash_map.h>
#include <absl/strings/string_view.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/cinn/backends/llvm/ir_builder_mixin.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/module.h"

namespace cinn {
namespace backends {

class LLVMIRVisitor : public ir::IRVisitorRequireReImpl<llvm::Value *> {
 public:
  LLVMIRVisitor() = default;

  using ir::IRVisitorRequireReImpl<llvm::Value *>::Visit;
#define __m(t__) virtual llvm::Value *Visit(const ir::t__ *x) = 0;
  NODETY_FORALL(__m)
#undef __m
};

class SymbolTable {
 public:
  SymbolTable() = default;

  void PushScope() { scopes_.emplace_back(); }

  llvm::Value *Lookup(const std::string &id) {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); it++) {
      auto vt = (*it).find(id);
      if (vt != (*it).end()) return vt->second;
    }
    return nullptr;
  }

  void Insert(const std::string &id, llvm::Value *value) {
    PADDLE_ENFORCE_EQ(
        !scopes_.empty(),
        true,
        ::common::errors::InvalidArgument("sorry, scopes_ can't be empty"));
    scopes_.back().emplace(id, value);
  }

  void Erase(const std::string &id) {
    PADDLE_ENFORCE_EQ(
        !scopes_.empty(),
        true,
        ::common::errors::InvalidArgument("sorry, scopes_ can't be empty"));
    scopes_.back().erase(id);
  }

  void PopScope() {
    PADDLE_ENFORCE_EQ(
        !scopes_.empty(),
        true,
        ::common::errors::InvalidArgument("sorry, scopes_ can't be empty"));
    scopes_.pop_back();
  }

  //! Get the number of the variables contained in the current scope.
  size_t size() const { return scopes_.empty() ? 0 : scopes_.back().size(); }

  size_t num_scopes() const { return scopes_.size(); }

 private:
  std::vector<absl::flat_hash_map<std::string, llvm::Value *>> scopes_;

  SymbolTable(const SymbolTable &) = delete;
};

struct SymbolTableGuard {
  explicit SymbolTableGuard(SymbolTable &symbol_table)  // NOLINT
      : symbol_table_(symbol_table) {
    symbol_table.PushScope();
  }

  ~SymbolTableGuard() { symbol_table_.PopScope(); }

 private:
  SymbolTable &symbol_table_;
};

/**
 * Base class of all the LLVM-based codegen.
 */
class CodeGenLLVM : public LLVMIRVisitor, public IrBuilderMixin<CodeGenLLVM> {
 public:
  explicit CodeGenLLVM(
      llvm::Module *m,
      llvm::IRBuilder<> *b,
      const std::shared_ptr<SymbolTable> &symbol_table = nullptr,
      const Target &target = cinn::common::DefaultHostTarget());

  // Common llvm types
  // @{
  inline llvm::Type *ll_void_p_ty() const { return llvm_type_of<void *>(m_); }
  inline llvm::Type *ll_void_pp_ty() const { return llvm_type_of<void **>(m_); }

  inline llvm::Type *ll_int8_ty() const { return llvm_type_of<int8_t>(m_); }
  inline llvm::Type *ll_int16_ty() const { return llvm_type_of<int16_t>(m_); }
  inline llvm::Type *ll_int32_ty() const { return llvm_type_of<int32_t>(m_); }
  inline llvm::Type *ll_int64_ty() const { return llvm_type_of<int64_t>(m_); }

  inline llvm::Type *ll_uint8_ty() const { return llvm_type_of<uint8_t>(m_); }
  inline llvm::Type *ll_uint16_ty() const { return llvm_type_of<uint16_t>(m_); }
  inline llvm::Type *ll_uint32_ty() const { return llvm_type_of<uint32_t>(m_); }
  inline llvm::Type *ll_uint64_ty() const { return llvm_type_of<uint64_t>(m_); }

  inline llvm::Type *ll_bf16_ty() const {
    return llvm_type_of<cinn::common::bfloat16>(m_);
  }
  inline llvm::Type *ll_fp16_ty() const {
    return llvm_type_of<cinn::common::float16>(m_);
  }
  inline llvm::Type *ll_fp32_ty() const { return llvm_type_of<float>(m_); }
  inline llvm::Type *ll_fp64_ty() const { return llvm_type_of<double>(m_); }

  inline llvm::Type *ll_cinn_buffer_p_ty() const {
    return llvm_type_of<cinn_buffer_t *>(m_);
  }
  inline llvm::Type *ll_cinn_pod_ty() const {
    return llvm_type_of<cinn_pod_value_t>(m_);
  }
  inline llvm::Type *ll_cinn_pod_p_ty() const {
    return llvm_type_of<cinn_pod_value_t *>(m_);
  }
  // @}

  //! get a llvm type equivalent to a CINN type.
  inline llvm::Type *ll_type_of(Type type) {
    return CinnTypeToLLVMType(type, m_);
  }

  // Common methods to get a constant
  // @{
  inline llvm::Constant *ll_const_int32(int v) const {
    return llvm::ConstantInt::get(b_->getInt32Ty(), v);
  }
  inline llvm::Constant *ll_const_int64(int v) const {
    return llvm::ConstantInt::get(b_->getInt64Ty(), v);
  }
  // @}

  //! Get the bound LLVM module.
  llvm::Module *m() { return m_; }
  //! Get the bound LLVM ir builder.
  llvm::IRBuilder<> *b() { return b_; }

  void Compile(const ir::Module &module);

  using LLVMIRVisitor::Visit;

  virtual llvm::Value *Visit(const ir::_Module_ *op);

  virtual llvm::Value *Visit(const ir::_LoweredFunc_ *op);

#define __(op__) llvm::Value *Visit(const ir::op__ *) override;
  NODETY_FORALL(__)
#undef __

#define __(op__) llvm::Value *Visit(const ir::intrinsics::op__ *);
  INTRINSIC_KIND_FOR_EACH(__)
#undef __

  //! Used for the ExternFuncEmitter to store temporary result.
  mutable llvm::Value *extern_func_emit_res_{};

  std::shared_ptr<SymbolTable> named_vars() { return symbol_table_; }

  llvm::FunctionType *GenFunctionTypeFromCinnFunction(
      const ir::_LoweredFunc_ *func, bool with_buffer_type);

  virtual llvm::Value *GetVar(const std::string &name, bool lazy = true);

  llvm::Function *GetIntrinsicDecl(llvm::Intrinsic::ID id,
                                   llvm::Type *ret_type,
                                   llvm::ArrayRef<llvm::Type *> arg_types);

  // Constants
  // @{
  inline llvm::Value *llvm_int32_constant(int v) {
    return llvm::ConstantInt::get(ll_int32_ty(), v);
  }
  // @}

  virtual ~CodeGenLLVM();

 protected:
  // TODO(Superjomn) When to clear the existing local variables when switch to
  // another function?
  llvm::Value *SetVar(const std::string &name, llvm::Value *val);
  llvm::Value *EmitVectorSlice(llvm::Value *vec, int begin, int extent);
  llvm::Value *EmitVectorPad(llvm::Value *vec, int lanes);
  llvm::Value *EmitVectorConcat(std::vector<llvm::Value *> vecs);

  //! Visit different kinds of Calls, the following methods are analogous to
  //! those in CodeGenC.
  // @{
  llvm::Value *EmitCall_buffer_create(const ir::Call *op);
  llvm::Value *EmitCall_buffer_malloc(const ir::Call *op);
  llvm::Value *EmitCall_get_address(const ir::Call *op);
  llvm::Value *EmitCall_debug_info(const ir::Call *op);
  // @}

  llvm::Value *EmitBinaryOp(llvm::Value *lhs,
                            llvm::Value *rhs,
                            char opcode,
                            bool is_integral,
                            bool is_signed = true);

  llvm::Value *LLVMGenGlobalStringVar(const std::string &data);

  llvm::Value *CreateBufferPtr(Type t, llvm::Value *buffer, llvm::Value *index);
  llvm::Value *CreateBufferVecPtr(Type t,
                                  llvm::Value *buffer,
                                  llvm::Value *index);
  llvm::Value *CreateVecSlice(llvm::Value *vec, int begin, int lanes);

  llvm::Value *DenseVectorLoad(const ir::Load *load);
  llvm::Value *CreateSerialFor(const ir::For *op, int stride = 1);

  /**
   * Mark a load or store with type-based-alias-analysis metadata so that LLVM
   * can optimize by reordering loads and stores across different buffers.
   */
  void AddTbaaMetadata(llvm::Instruction *inst,
                       absl::string_view buffer,
                       Expr index);

  void InitTarget(const Target &target);

  void Scalarize(const Expr &e,
                 std::function<void(int i, llvm::Value *v)> flambda);

  llvm::Module *m_;
  llvm::IRBuilder<> *b_;
  // Current function
  llvm::Function *f_;

  std::unique_ptr<llvm::MDBuilder> md_builder_;

  // std::shared_ptr<absl::flat_hash_map<std::string, llvm::Value *>>
  // named_vars_;
  std::shared_ptr<SymbolTable> symbol_table_;
  std::unordered_set<ir::_Var_ *> alias_vars_;

  llvm::MDNode *md_tbaa_root_{nullptr};
  llvm::MDNode *md_tbaa_alias_set_{nullptr};

  int naive_vec_alignment_{0};
  Target target_;
};
namespace detail {
Expr StridedRampBase(Expr e, int stride);
}  // namespace detail

}  // namespace backends
}  // namespace cinn
