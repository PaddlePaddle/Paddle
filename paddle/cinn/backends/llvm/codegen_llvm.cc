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

#include "paddle/cinn/backends/llvm/codegen_llvm.h"

#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include "paddle/common/enforce.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "paddle/cinn/backends/extern_func_emitter.h"
#include "paddle/cinn/backends/extern_func_emitter_builtin.h"
#include "paddle/cinn/backends/llvm/llvm_util.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_verify.h"
#include "paddle/cinn/optim/var_mod_simplify.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace backends {

using BinaryInstruction = llvm::Instruction::BinaryOps;
using cinn::common::bfloat16;
using cinn::common::float16;

namespace {

template <typename T>
auto NodeToExpr(const T *node) {
  std::ostringstream oss;
  // oss << "\033[32m";
  oss << ir::Expr(const_cast<T *>(node));
  // oss << "\033[0m";
  return oss.str();
}

bool is_integral_type(cinn::common::Type t) {
  return t.is_int() || t.is_uint();
}

bool is_floating_type(cinn::common::Type t) { return t.is_float(); }

llvm::Value *EmitComparison(llvm::CmpInst::Predicate predicate,
                            llvm::Value *lhs,
                            llvm::Value *rhs,
                            llvm::IRBuilder<> *b) {
  llvm::Value *comparison_result{nullptr};
  if (lhs->getType()->isIntegerTy()) {
    comparison_result = b->CreateICmp(predicate, lhs, rhs);
  } else {
    comparison_result = b->CreateFCmp(predicate, lhs, rhs);
  }

  return comparison_result;
}

#define __IR_EMITTER_NOT_IMPLEMENTED(__op) CINN_NOT_IMPLEMENTED

int NextPowerOfTwo(int x) {
  for (int p2 = 1;; p2 *= 2) {
    if (p2 >= x) {
      return p2;
    }
  }
  return 0;
}

}  // namespace

CodeGenLLVM::CodeGenLLVM(llvm::Module *m,
                         llvm::IRBuilder<> *b,
                         const std::shared_ptr<SymbolTable> &symbol_table,
                         const Target &target)
    : m_(m), b_(b), symbol_table_(symbol_table), target_(target) {
  if (!symbol_table.get()) {
    symbol_table_ = std::make_shared<SymbolTable>();
  }
  symbol_table_->PushScope();  // Create a new scope by default.

  md_builder_ = std::make_unique<llvm::MDBuilder>(b_->getContext());
  md_tbaa_root_ = md_builder_->createTBAARoot("cinn-tbaa");
  md_tbaa_alias_set_ = md_builder_->createTBAANode("cinn-alias", md_tbaa_root_);
  InitTarget(target_);
}

CodeGenLLVM::~CodeGenLLVM() {}

llvm::Value *CodeGenLLVM::EmitVectorSlice(llvm::Value *vec,
                                          int begin,
                                          int extent) {
  int numel =
      llvm::dyn_cast<llvm::VectorType>(vec->getType())->getNumElements();
  if (extent == numel && begin == 0) return vec;

  CHECK(begin >= 0 && extent <= numel) << "Slicing out of bound!";

  std::vector<llvm::Constant *> indices(extent);
  for (int i = 0; i < extent; i++) {
    llvm::Constant **v = &indices[i];
    if (begin + i >= 0 && begin + i < numel) {
      *v = llvm::ConstantInt::get(b_->getInt32Ty(), begin + i);
    } else {
      *v = llvm::UndefValue::get(b_->getInt32Ty());
    }
  }
  return ShuffleVector(vec, vec, llvm::ConstantVector::get(std::move(indices)));
}

llvm::Value *CodeGenLLVM::EmitVectorPad(llvm::Value *vec, int lanes) {
#if LLVM_VERSION_MAJOR <= 10
  llvm::Value *mask =
      llvm::UndefValue::get(llvm::VectorType::get(b_->getInt32Ty(), lanes));
#else
  llvm::Value *mask = llvm::UndefValue::get(llvm::VectorType::get(
      b_->getInt32Ty(), llvm::ElementCount(lanes, false /*Scalable*/)));
#endif
  int numel =
      llvm::dyn_cast<llvm::VectorType>(vec->getType())->getNumElements();

  CHECK(numel <= lanes);
  if (numel == lanes) return vec;
  for (int i = 0; i < numel; i++) {
    mask = InsertElement(mask,
                         llvm::ConstantInt::get(b_->getInt32Ty(), i),
                         llvm::ConstantInt::get(b_->getInt32Ty(), i));
  }

  return ShuffleVector(vec, vec, mask);
}

llvm::Value *CodeGenLLVM::EmitVectorConcat(std::vector<llvm::Value *> vecs) {
  int lanes = 0;
  for (auto *v : vecs) {
    lanes += llvm::dyn_cast<llvm::VectorType>(v->getType())->getNumElements();
  }
  while (vecs.size() > 1) {
    std::vector<llvm::Value *> new_vecs;
    for (size_t i = 0; i < vecs.size() - 1; i += 2) {
      auto *lhs = vecs[i];
      auto *rhs = vecs[i + 1];
      const auto lhs_lanes =
          llvm::dyn_cast<llvm::VectorType>(lhs->getType())->getNumElements();
      const auto rhs_lanes =
          llvm::dyn_cast<llvm::VectorType>(rhs->getType())->getNumElements();
      if (lhs_lanes < rhs_lanes) {
        lhs = EmitVectorPad(lhs, rhs_lanes);
      } else if (lhs_lanes > rhs_lanes) {
        rhs = EmitVectorPad(rhs, lhs_lanes);
      }

      const auto shared_lanes = std::max(lhs_lanes, rhs_lanes);
      std::vector<unsigned> mask(lhs_lanes + rhs_lanes);
      std::iota(mask.begin(), std::next(mask.begin(), lhs_lanes), 0);
      std::iota(std::next(mask.begin(), lhs_lanes), mask.end(), shared_lanes);
      new_vecs.push_back(ShuffleVector(lhs, rhs, mask));
    }
    if (vecs.size() % 2) {
      new_vecs.push_back(vecs.back());
    }

    vecs = std::move(new_vecs);
  }

  return EmitVectorSlice(vecs[0], 0, lanes);
}

llvm::Value *CodeGenLLVM::EmitBinaryOp(llvm::Value *lhs,
                                       llvm::Value *rhs,
                                       char opcode,
                                       bool is_integral,
                                       bool is_signed) {
  llvm::Instruction::BinaryOps ops;
  PADDLE_ENFORCE_EQ(
      lhs->getType(),
      rhs->getType(),
      ::common::errors::InvalidArgument(
          "the types of operands of binary operation are mismatch"));

  switch (opcode) {
    case '+':
      ops = is_integral ? llvm::Instruction::BinaryOps::Add
                        : llvm::Instruction::BinaryOps::FAdd;
      break;
    case '-':
      ops = is_integral ? llvm::Instruction::BinaryOps::Sub
                        : llvm::Instruction::BinaryOps::FSub;
      break;
    case '*':
      ops = is_integral ? llvm::Instruction::BinaryOps::Mul
                        : llvm::Instruction::BinaryOps::FMul;
      break;
    case '/':
      ops = is_integral ? (is_signed ? llvm::Instruction::BinaryOps::SDiv
                                     : llvm::Instruction::BinaryOps::UDiv)
                        : llvm::Instruction::BinaryOps::FDiv;
      break;
    case '%':
      ops = is_integral ? (is_signed ? llvm::Instruction::BinaryOps::SRem
                                     : llvm::Instruction::BinaryOps::URem)
                        : llvm::Instruction::BinaryOps::FRem;
      break;
    default:
      return nullptr;
  }
  return BinOp(ops, lhs, rhs);
}

llvm::Value *CodeGenLLVM::Visit(const ir::IntImm *op) {
  auto *type = b_->getIntNTy(op->type().bits());
  return llvm::ConstantInt::get(type, op->value, true);
}

llvm::Value *CodeGenLLVM::Visit(const ir::UIntImm *op) {
  if (op->type().is_bool()) {
    auto *type = b_->getInt1Ty();
    return llvm::ConstantInt::get(type, op->value, false);
  }
  auto *type = b_->getIntNTy(op->type().bits());
  return llvm::ConstantInt::get(type, op->value, false);
}

llvm::Value *CodeGenLLVM::Visit(const ir::FloatImm *op) {
  if (op->type().is_float(64)) {
    return llvm::ConstantFP::get(b_->getDoubleTy(), op->value);
  } else if (op->type().is_float(32)) {
    return llvm::ConstantFP::get(b_->getFloatTy(), op->value);
  } else if (op->type().is_bfloat16()) {
    return llvm::ConstantFP::get(b_->getBFloatTy(), op->value);
  } else if (op->type().is_float16()) {
    return llvm::ConstantFP::get(b_->getHalfTy(), op->value);
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument("illegal float type."));
  }
  return nullptr;
}

llvm::Value *CodeGenLLVM::LLVMGenGlobalStringVar(const std::string &data) {
  return b_->CreateGlobalStringPtr(data);
}

llvm::Value *CodeGenLLVM::Visit(const ir::StringImm *op) {
  return LLVMGenGlobalStringVar(op->value);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Add *op) {
  ir::TryElevateInt32ToInt64({op->a(), op->b()});
  return EmitBinaryOp(
      Visit(&op->a()), Visit(&op->b()), '+', is_integral_type(op->type()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Sub *op) {
  ir::TryElevateInt32ToInt64({op->a(), op->b()});
  return EmitBinaryOp(
      Visit(&op->a()), Visit(&op->b()), '-', is_integral_type(op->type()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Mul *op) {
  ir::TryElevateInt32ToInt64({op->a(), op->b()});
  auto *lhs = Visit(&op->a());
  auto *rhs = Visit(&op->b());
  return EmitBinaryOp(lhs, rhs, '*', is_integral_type(op->type()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Div *op) {
  ir::TryElevateInt32ToInt64({op->a(), op->b()});
  return EmitBinaryOp(
      Visit(&op->a()), Visit(&op->b()), '/', is_integral_type(op->type()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Mod *op) {
  ir::TryElevateInt32ToInt64({op->a(), op->b()});
  return EmitBinaryOp(
      Visit(&op->a()), Visit(&op->b()), '%', is_integral_type(op->type()));
}

#define __IR_EMITTER_DEFINE_CMP_VISITOR(__sop, __uop, __fop) \
  ir::TryElevateInt32ToInt64({op->a(), op->b()});            \
  auto *lhs = Visit(&op->a());                               \
  auto *rhs = Visit(&op->b());                               \
  CHECK(op->a().type() == op->b().type());                   \
  llvm::CmpInst::Predicate predicate;                        \
  if (op->a().type().is_int()) {                             \
    predicate = llvm::CmpInst::ICMP_##__sop;                 \
  } else if (op->a().type().is_uint()) {                     \
    predicate = llvm::CmpInst::ICMP_##__uop;                 \
  } else /*float*/ {                                         \
    predicate = llvm::CmpInst::FCMP_##__fop;                 \
  }                                                          \
  return EmitComparison(predicate, lhs, rhs, b_)

llvm::Value *CodeGenLLVM::Visit(const ir::EQ *op) {
  __IR_EMITTER_DEFINE_CMP_VISITOR(EQ, EQ, OEQ);
}

llvm::Value *CodeGenLLVM::Visit(const ir::NE *op) {
  __IR_EMITTER_DEFINE_CMP_VISITOR(NE, NE, ONE);
}

llvm::Value *CodeGenLLVM::Visit(const ir::LT *op) {
  __IR_EMITTER_DEFINE_CMP_VISITOR(SLT, ULT, OLT);
}

llvm::Value *CodeGenLLVM::Visit(const ir::LE *op) {
  __IR_EMITTER_DEFINE_CMP_VISITOR(SLE, ULE, OLE);
}

llvm::Value *CodeGenLLVM::Visit(const ir::GT *op) {
  __IR_EMITTER_DEFINE_CMP_VISITOR(SGT, UGT, OGT);
}

llvm::Value *CodeGenLLVM::Visit(const ir::GE *op) {
  __IR_EMITTER_DEFINE_CMP_VISITOR(SGE, UGE, OGE);
}

#undef __IR_EMITTER_DEFINE_CMP_VISITOR

llvm::Value *CodeGenLLVM::Visit(const ir::And *op) {
  return And(Visit(&op->a()), Visit(&op->b()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Or *op) {
  return Or(Visit(&op->a()), Visit(&op->b()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Min *op) {
  auto *lhs = Visit(&op->a());
  auto *rhs = Visit(&op->b());

  llvm::Value *p{nullptr};
  if (op->type().is_int()) {
    p = ICmpSLT(lhs, rhs);
  } else if (op->type().is_uint()) {
    p = ICmpULT(lhs, rhs);
  } else /*float*/ {
    p = FCmpOLT(lhs, rhs);
  }

  return Select(p, lhs, rhs);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Max *op) {
  auto *lhs = Visit(&op->a());
  auto *rhs = Visit(&op->b());

  llvm::Value *p = nullptr;
  if (op->type().is_int()) {
    p = ICmpSGT(lhs, rhs);
  } else if (op->type().is_uint()) {
    p = ICmpUGT(lhs, rhs);
  } else /*float*/ {
    p = FCmpOGT(lhs, rhs);
  }

  return Select(p, lhs, rhs);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Minus *op) {
  auto *v = Visit(&op->v());
  return (op->type().is_int() || op->type().is_uint()) ? Neg(v) : FNeg(v);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Not *op) {
  return Not(Visit(&op->v()));
}

llvm::Value *CodeGenLLVM::Visit(const ir::Cast *op) {
  auto from = op->v().type();
  auto to = op->type();

  llvm::Type *source = CinnTypeToLLVMType(from, m_);
  llvm::Type *target = CinnTypeToLLVMType(to, m_);
  CHECK(source) << "source ir type is null";
  CHECK(target) << "target ir type is null";

  llvm::Value *value = Visit(&op->v());
  CHECK(value) << "value is null";

  // pod_value_t cast to a value.
  if (op->v().type().is_customized_type() &&
      op->v().type().customized_type() ==
          cinn::common::customized_type::kpod_value_t) {  // pod_value_t
                                                          // operator
    llvm::Function *callee{};
    if (op->type().is_bool()) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_bool);
    } else if (op->type().is_int(8)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_int8);
    } else if (op->type().is_int(16)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_int16);
    } else if (op->type().is_int(32)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_int32);
    } else if (op->type().is_int(64)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_int64);
    } else if (op->type().is_uint(8)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_uint8);
    } else if (op->type().is_uint(16)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_uint16);
    } else if (op->type().is_uint(32)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_uint32);
    } else if (op->type().is_uint(64)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_uint64);
    } else if (op->type().is_float(32)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_float);
    } else if (op->type().is_float(64)) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_double);
    } else if (op->type().is_bfloat16()) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_bfloat16);
    } else if (op->type().is_float16()) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_float16);
    } else if (op->type() == type_of<void *>()) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_void_p);
    } else if (op->type() == type_of<cinn_buffer_t *>() ||
               op->type() == type_of<const cinn_buffer_t *>()) {
      callee = m_->getFunction(runtime::intrinsic::pod_value_to_buffer_p);
    } else {
      LOG(ERROR) << "can't cast cinn_pod_value_t to " << op->type();
      CINN_NOT_IMPLEMENTED
    }

    CHECK(callee);
    CHECK(op->v().as_var()) << "argument to the intrinsic function "
                               "cinn_pod_value_to_x should be a Var";
    value = GetVar(op->v().as_var()->name);
    return Call(callee, std::vector<llvm::Value *>({value}), "pod_value_cast");
  }

  do {
    if (value->getType() == target) break;

    if (to.is_cpp_handle() || to.is_cpp_handle2()) {
      value = BitCast(value, target, "cast_to_cpp_handle");
      break;
    }

    if (to.is_bool()) {
      if (from.is_float()) {
        llvm::Constant *zero = llvm::ConstantFP::get(source, 0.);
        value = FCmpONE(value, zero);
      } else {
        llvm::Constant *zero = llvm::ConstantInt::get(source, 0);
        value = ICmpNE(value, zero);
      }
      break;
    }

    if (from.is_float() == false && to.is_float() == false) {
      value = IntCast(value, target, from.is_int());
      break;
    }

    if (from.is_float() && to.is_int()) {
      value = FPToSI(value, target);
      break;
    }

    if (from.is_float() && to.is_uint()) {
      value = FPToUI(value, target);
      if (to.bits() < 8) {
        value = IntCast(value, target, false);
      }
      break;
    }

    if (from.is_int() && to.is_float()) {
      value = SIToFP(value, target);
      break;
    }

    if (from.is_uint() && to.is_float()) {
      value = UIToFP(value, target);
      break;
    }

    CHECK(from.is_float() && to.is_float());
    value = FPCast(value, target);
  } while (false);

  return value;
}

llvm::Value *CodeGenLLVM::CreateSerialFor(const ir::For *op, int stride) {
  SymbolTableGuard symbol_table_guard(*symbol_table_);

  llvm::BasicBlock *preheader_bb = b_->GetInsertBlock();
  llvm::BasicBlock *exit_bb = nullptr;

  llvm::BasicBlock::iterator insert_point = b_->GetInsertPoint();

  if (insert_point == preheader_bb->end()) {
    CHECK(!preheader_bb->getTerminator());
    exit_bb = llvm::BasicBlock::Create(b_->getContext(),
                                       "loop_exit",
                                       b_->GetInsertBlock()->getParent(),
                                       nullptr);
  } else {
    CHECK(preheader_bb->getTerminator());
    exit_bb = preheader_bb->splitBasicBlock(insert_point, "loop_exit");
    preheader_bb->getTerminator()->eraseFromParent();
  }

  llvm::BasicBlock *header_bb =
      llvm::BasicBlock::Create(b_->getContext(),
                               "loop_header",
                               b_->GetInsertBlock()->getParent(),
                               nullptr);
  llvm::BasicBlock *body_bb =
      llvm::BasicBlock::Create(b_->getContext(),
                               "loop_body",
                               b_->GetInsertBlock()->getParent(),
                               nullptr);

  llvm::Function *func = preheader_bb->getParent();
  b_->SetInsertPoint(&func->getEntryBlock(),
                     func->getEntryBlock().getFirstInsertionPt());

  llvm::Value *old_var = GetVar(op->loop_var->name);
  // loop iterator
  llvm::AllocaInst *loop_var = Alloca(
      b_->getIntNTy(op->min->type().bits()), nullptr, op->loop_var->name);
  loop_var->setAlignment(llvm::Align(4));
  SetVar(op->loop_var->name, loop_var);

  b_->SetInsertPoint(preheader_bb);
  llvm::Value *start_index = Visit(&op->min);
  llvm::Value *end_index = Visit(&op->extent);
  Store(start_index, loop_var);
  CHECK(!preheader_bb->getTerminator());
  Br(header_bb);

  // loop_header
  b_->SetInsertPoint(header_bb);
  llvm::Value *indvar = Load(loop_var, "indvar");
  llvm::Value *exit_cond = ICmpSGE(indvar, end_index);
  CondBr(/*Cond=*/exit_cond,
         /*True=*/exit_bb,
         /*False=*/body_bb);

  // loop_body
  b_->SetInsertPoint(body_bb);
  llvm::Value *step =
      llvm::ConstantInt::get(b_->getIntNTy(op->min->type().bits()), stride);

  Visit(&op->body);
  llvm::Value *indvar_inc = Add(indvar,
                                step,
                                "indvar.inc",
                                /*HasNUW=*/true,
                                /*HasNSW=*/true);
  Store(indvar_inc, loop_var);
  llvm::BranchInst *back_branch = Br(header_bb);

  // Add loop metadata
  decltype(auto) ctx = b_->getContext();
  std::vector<llvm::Metadata *> loop_metadata;
  auto temp_node = llvm::MDNode::getTemporary(ctx, llvm::None);
  loop_metadata.push_back(temp_node.get());

  // TODO(fc500110): Loop vectorize
  // auto *vectorization = op->metadata.vectorization ? b_->getTrue() :
  // b_->getFalse(); loop_metadata.push_back(llvm::MDNode::get(
  //        ctx, {llvm::MDString::get(ctx, "llvm.loop.vectorize.enable"),
  //        llvm::ConstantAsMetadata::get(b_->getFalse())}));

  // Loop unroll
  std::string llvm_unroll_metadata{"llvm.loop.unroll."};
  switch (op->metadata.unroll_mode) {
    case ir::LLVMForLoopMeta::FullyUnroll:
      llvm_unroll_metadata += "full";
      break;
    case ir::LLVMForLoopMeta::NoUnroll:
      llvm_unroll_metadata += "disable";
      break;
    default:
      llvm_unroll_metadata += "enable";
  }

  /*
  loop_metadata.push_back(llvm::MDNode::get(ctx, {llvm::MDString::get(ctx,
  llvm_unroll_metadata)})); auto loop_id = llvm::MDNode::get(ctx,
  loop_metadata); loop_id->replaceOperandWith(0, loop_id);
  back_branch->setMetadata(llvm::LLVMContext::MD_loop, loop_id);
  */

  if (old_var) {
    SetVar(op->loop_var->name, old_var);
  } else {
    symbol_table_->Erase(op->loop_var->name);
  }

  b_->SetInsertPoint(exit_bb);
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::For *op) {
  return CreateSerialFor(op);
}

llvm::Value *CodeGenLLVM::Visit(const ir::PolyFor *op) {
  CINN_NOT_IMPLEMENTED
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Select *op) {
  return Select(
      Visit(&op->condition), Visit(&op->true_value), Visit(&op->false_value));
}

llvm::Value *CodeGenLLVM::Visit(const ir::IfThenElse *op) {
  SymbolTableGuard symbol_table_guard(*symbol_table_);

  bool emit_else = op->false_case.defined();

  auto &ll_ctx = b_->getContext();
  auto *ll_function = b_->GetInsertBlock()->getParent();

  llvm::Value *cond = Visit(&op->condition);
  llvm::BasicBlock *then_block =
      llvm::BasicBlock::Create(ll_ctx, "if-then", ll_function);
  llvm::BasicBlock *end_block =
      llvm::BasicBlock::Create(ll_ctx, "if-end", ll_function);

  if (op->false_case.defined()) {
    llvm::BasicBlock *else_block =
        llvm::BasicBlock::Create(ll_ctx, "if-else", ll_function);
    CondBr(cond, then_block, else_block);

    // true case
    b_->SetInsertPoint(then_block);
    Visit(&op->true_case);
    Br(end_block);

    // false case
    b_->SetInsertPoint(else_block);
    Visit(&op->false_case);
    Br(end_block);
  } else {
    CondBr(cond, then_block, end_block);
    b_->SetInsertPoint(then_block);
    Visit(&op->true_case);
    Br(end_block);
  }
  b_->SetInsertPoint(end_block);

  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Block *op) {
  // Create a new scope holding the temporary variables.
  SymbolTableGuard symbol_table_guard(*symbol_table_);

  llvm::Value *ret = nullptr;

  llvm::BasicBlock *block = llvm::BasicBlock::Create(
      b_->getContext(), "block", b_->GetInsertBlock()->getParent(), nullptr);

  Br(block);
  b_->SetInsertPoint(block);

  for (const auto &expr : op->stmts) {
    ret = Visit(&expr);
  }

  return ret;
}

llvm::Value *CodeGenLLVM::Visit(const ir::PrimitiveNode *) {
  CINN_NOT_IMPLEMENTED return nullptr;
}
llvm::Value *CodeGenLLVM::Visit(const ir::_BufferRange_ *) {
  CINN_NOT_IMPLEMENTED return nullptr;
}
llvm::Value *CodeGenLLVM::Visit(const ir::ScheduleBlock *) {
  CINN_NOT_IMPLEMENTED return nullptr;
}
llvm::Value *CodeGenLLVM::Visit(const ir::ScheduleBlockRealize *) {
  CINN_NOT_IMPLEMENTED return nullptr;
}
llvm::Value *CodeGenLLVM::Visit(const ir::_Dim_ *) {
  CINN_NOT_IMPLEMENTED return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Call *op) {
  if (op->name == runtime::intrinsic::debug_log_repr) {
    return EmitCall_debug_info(op);
  } else if (op->is_extern_call()) {
    auto emitter_id = ExternFuncID{backend_llvm_host, op->name.c_str()};
    const auto &fn_name =
        ExternFunctionEmitterRegistry::Global().Lookup(emitter_id);
    if (!fn_name.empty()) {
      ExternFunctionLLVMEmitter emitter(fn_name);
      emitter.BindCodeGen(this);
      emitter.Emit(op);
      return extern_func_emit_res_;
    }
  }

  llvm::Function *callee = m_->getFunction(op->name);
  CHECK(callee) << "Unknown function referenced. [" << op->name << "]";

  std::vector<llvm::Value *> args;
  for (const auto &e : op->read_args) {
    auto *arg = Visit(&e);
    CHECK(arg) << "argument " << e << " is null";
    args.push_back(arg);
  }
  for (const auto &e : op->write_args) {
    auto *arg = Visit(&e);
    CHECK(arg) << "argument " << e << " is null";
    args.push_back(arg);
  }

  if (op->is_cinn_call()) {
    auto arg = ir::intrinsics::GetAddr::Make(op->read_args[0]);
    args[0] = Visit(&arg);
    args[0] = BitCast(args[0], ll_void_p_ty(), "cast_to_void_p");
  }

  return Call(callee, std::move(args));
}

llvm::Value *CodeGenLLVM::Visit(const ir::_Module_ *op) {
  { ir::ir_utils::IrVerify(op); }

  for (auto &fn : op->functions) {
    VLOG(3) << "JIT Linking function [" << fn.As<ir::_LoweredFunc_>()->name
            << "]";
    auto fnll = Visit(fn.As<ir::_LoweredFunc_>());

    VLOG(5) << "fn llvm:\n" << DumpToString(*fnll);
  }
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::_Var_ *op) {
  llvm::Value *value = GetVar(op->name, /* lazy= */ false);
  // When visiting a Var that is allocated on the stack, we are actually
  // reading its value instead of its address.
  if (llvm::AllocaInst::classof(value)) {
    return Load(value, op->name + "_load");
  }
  return value;
}

void CodeGenLLVM::Scalarize(
    const Expr &e, std::function<void(int i, llvm::Value *v)> flambda) {
  if (const ir::Ramp *ramp = e.As<ir::Ramp>()) {
    for (int i = 0; i < ramp->type().lanes(); ++i) {
      Expr offset = ramp->base + (ramp->stride * i);
      VLOG(3) << "offset: " << offset;
      flambda(i, Visit(&offset));
    }
  } else {
    llvm::Value *value = Visit(&e);
    for (int i = 0; i < e->type().lanes(); ++i) {
      flambda(i, b_->CreateExtractElement(value, i));
    }
  }
}

llvm::Value *CodeGenLLVM::Visit(const ir::Load *op) {
  llvm::Value *array{nullptr};
  bool is_alias{false};
  if (auto *tensor_op = op->tensor.As<ir::_Tensor_>()) {
    array = GetVar(tensor_op->name);
  } else if (auto *var_op = op->tensor.As<ir::_Var_>()) {
    array = GetVar(var_op->name);
    is_alias = alias_vars_.count(const_cast<ir::_Var_ *>(var_op));
  } else {
    array = Visit(&op->tensor);
  }
  CHECK(array) << "fail to Visit Load node: "
               << Expr(const_cast<ir::Load *>(op));

  ir::Expr index = op->index();
  if (index.type().lanes() <= 1) {
    std::vector<llvm::Value *> indices;
    indices.push_back(Visit(&index));

    // auto load_inst = Load(InBoundsGEP(array, std::move(indices)));
    auto *load_inst =
        AlignedLoad(InBoundsGEP(array, std::move(indices)), llvm::MaybeAlign());
    /*
    if (is_alias) {
      llvm::MDNode *meta = md_builder_->createTBAANode("cinn-alias",
    md_tbaa_root_); load_inst->setMetadata("tbaa",
    md_builder_->createTBAAStructTagNode(meta, meta, 0));
    }
     */
    if (auto *load_tensor = op->tensor.as_tensor()) {
      AddTbaaMetadata(load_inst, load_tensor->name, op->index());
    }

    {
      int alignment = op->type().bits();
      alignment = 8;
      PADDLE_ENFORCE_GT(alignment,
                        0,
                        ::common::errors::InvalidArgument(
                            "alignment should be greater than 0"));
      load_inst->setAlignment(llvm::Align(std::min(alignment, 8)));
    }

    // TODO(fc500110): tbaa AliasAnalysis
    // auto md_tbaa_root      = md_builder_->createTBAARoot("cinn-tbaa");
    // auto md_tbaa_alias_set = md_builder_->createTBAANode("cinn-alias",
    // md_tbaa_root); llvm::MDNode *meta     = md_tbaa_alias_set;
    // load_inst->setMetadata("tbaa", md_builder_->createTBAAStructTagNode(meta,
    // meta, 0));
    return load_inst;
  } else {  // vector load
    Expr dense_strided_ramp = detail::StridedRampBase(op->index(), 1);
    llvm::Value *buffer = Visit(&op->tensor);
    if (dense_strided_ramp.defined()) {
      CHECK(op->type().is_vector());
      return DenseVectorLoad(op);
    }
    // scalarize load
    Type type = op->type();
    int alignment = type.bits() / 8;
    llvm::Value *ret =
        llvm::UndefValue::get(CinnTypeToLLVMType(type, m_, true));
    auto flambda = [&](int i, llvm::Value *index) {
      auto *ptr = CreateBufferPtr(type.ElementOf(), buffer, index);
      llvm::LoadInst *load_inst =
          b_->CreateAlignedLoad(ptr, llvm::Align(alignment), "load_vec");
      ret = b_->CreateInsertElement(ret, load_inst, ll_const_int32(i));
      if (auto *load_tensor = op->tensor.as_tensor()) {
        AddTbaaMetadata(load_inst, load_tensor->name, op->index());
      }
    };
    Scalarize(op->index(), flambda);
    return ret;
  }
}

llvm::Value *CodeGenLLVM::Visit(const ir::Store *op) {
  llvm::Value *array{nullptr};
  bool is_alias{false};
  if (auto *tensor_op = op->tensor.As<ir::_Tensor_>()) {
    array = GetVar(tensor_op->name);
  } else if (auto *var_op = op->tensor.As<ir::_Var_>()) {
    array = GetVar(var_op->name);
    is_alias = alias_vars_.count(const_cast<ir::_Var_ *>(var_op));
  }
  CHECK(array) << "array is null";

  ir::Expr index = op->index();

  if (op->type().is_scalar()) {
    std::vector<llvm::Value *> indices;
    indices.push_back(Visit(&index));

    // auto *store_inst = Store(Visit(&op->value), InBoundsGEP(array,
    // std::move(indices)));
    auto *store_inst = AlignedStore(Visit(&op->value),
                                    InBoundsGEP(array, std::move(indices)),
                                    llvm::MaybeAlign());
    /*
    if (is_alias) {
      llvm::MDNode *meta = md_builder_->createTBAANode("cinn-alias",
    md_tbaa_root_); store_inst->setMetadata("tbaa",
    md_builder_->createTBAAStructTagNode(meta, meta, 0));
    }
     */
    {
      int alignment = op->type().bits();
      alignment = 8;
      PADDLE_ENFORCE_GT(alignment,
                        0,
                        ::common::errors::InvalidArgument(
                            "alignment should be greater than 0"));
      store_inst->setAlignment(llvm::Align(std::min(alignment, 8)));
    }
    // TODO(fc500110): tbaa AliasAnalysis
    // auto md_tbaa_root      = md_builder_->createTBAARoot("cinn-tbaa");
    // auto md_tbaa_alias_set = md_builder_->createTBAANode("cinn-alias",
    // md_tbaa_root); llvm::MDNode *meta     = md_tbaa_alias_set;
    // store_inst->setMetadata("tbaa",
    // md_builder_->createTBAAStructTagNode(meta, meta, 0));
    AddTbaaMetadata(store_inst, op->tensor.as_tensor()->name, op->index());
    return store_inst;
  } else {  // vector store
    Expr dense_strided_ramp = detail::StridedRampBase(op->index(), 1);
    auto ramp_expr = op->index();
    auto *ramp = index.As<ir::Ramp>();
    auto *buffer = Visit(&op->tensor);
    auto *value = Visit(&op->value);

    if (dense_strided_ramp.defined()) {  // stride 1
      int total_lanes = op->type().lanes();
      int step = naive_vec_alignment_ / op->type().ElementOf().bits();

      // fit the total_lanes in native_lanes(split into multiple native steps)
      for (int offset = 0; offset < total_lanes; offset += total_lanes) {
        int lanes = total_lanes;
        Expr base = optim::ArithSimplify(ramp->base + offset);
        optim::VarModSimplify(&base);
        auto *ptr =
            CreateBufferPtr(op->type().ElementOf(), buffer, Visit(&base));
        auto *vtype = llvm::VectorType::get(
                          CinnTypeToLLVMType(op->type().ElementOf(), m_, true),
                          llvm::ElementCount(lanes, false /*Scalable*/))
                          ->getPointerTo();
        int alignment = std::max(op->type().ElementOf().bits() / 8, 1);
        llvm::StoreInst *inst =
            b_->CreateAlignedStore(CreateVecSlice(value, offset, lanes),
                                   b_->CreatePointerCast(ptr, vtype),
                                   alignment);
        AddTbaaMetadata(inst, op->tensor.as_tensor()->name, base);
        return inst;
      }
    }
    // scalarize store
    Type type = op->type();
    int alignment = type.bits() / 8;
    llvm::Value *ret =
        llvm::UndefValue::get(CinnTypeToLLVMType(type, m_, true));
    auto flambda = [&](int i, llvm::Value *index) {
      auto *ptr = CreateBufferPtr(type.ElementOf(), buffer, index);
      llvm::StoreInst *store_inst =
          b_->CreateAlignedStore(b_->CreateExtractElement(value, i),
                                 ptr,
                                 llvm::Align(alignment),
                                 "store_vec");
      ret = b_->CreateInsertElement(ret, store_inst, ll_const_int32(i));
      if (auto *store_tensor = op->tensor.as_tensor()) {
        AddTbaaMetadata(store_inst, store_tensor->name, op->index());
      }
    };
    Scalarize(op->index(), flambda);
    return ret;
  }
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Alloc *op) {
  auto *buffer_op = op->destination.As<ir::_Buffer_>();
  auto *buffer = GetVar(buffer_op->name);
  CHECK(buffer);

  return buffer;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Free *op) {
  auto *buffer_op = op->destination.As<ir::_Buffer_>();
  CHECK(symbol_table_->Lookup(buffer_op->name));
  symbol_table_->Erase(buffer_op->name);
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::_Buffer_ *op) {
  return GetVar(op->name);
}

llvm::Value *CodeGenLLVM::Visit(const ir::_Tensor_ *op) {
  return GetVar(op->name);
}

template <typename T,
          std::enable_if_t<std::is_same<const ir::Expr &, T>::value, int> = 0>
void appendBody(std::vector<Expr> &new_body, T &&v) {  // NOLINT
  new_body.push_back(v);
}

template <typename T,
          std::enable_if_t<!std::is_same<const ir::Expr &, T>::value, int> = 1>
void appendBody(std::vector<Expr> &new_body, T &&v) {  // NOLINT
  new_body.insert(new_body.end(), v.begin(), v.end());
}

llvm::Value *CodeGenLLVM::Visit(const ir::_LoweredFunc_ *op) {
  auto init_function_state = [this]() { alias_vars_.clear(); };
  init_function_state();

  PADDLE_ENFORCE_EQ(
      op->alloc_output_buffer_exprs.size(),
      op->dealloc_output_buffer_exprs.size(),
      ::common::errors::InvalidArgument(
          "the count of allocation and deallocation expressions is not "
          "match"));

  std::vector<Expr> new_body;
  auto create_temp_buffers = op->PrepareCreateTempBufferExprs();
  auto alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  auto dealloca_temp_buffers = op->PrepareDeallocTempBufferExprs();

  appendBody(new_body, op->argument_prepare_exprs);
  appendBody(new_body, create_temp_buffers);
  appendBody(new_body, alloca_temp_buffers);
  appendBody(new_body, op->alloc_output_buffer_exprs);
  appendBody(new_body, op->buffer_data_cast_exprs);
  appendBody(new_body, op->body);
  appendBody(new_body, dealloca_temp_buffers);
  appendBody(new_body, op->dealloc_output_buffer_exprs);

  ir::Expr function_body = ir::Block::Make(new_body);

  // Emit Function
  std::vector<llvm::Type *> arg_types = {b_->getInt8PtrTy(), b_->getInt32Ty()};

  llvm::FunctionType *function_type = llvm::FunctionType::get(
      /*Result=*/b_->getVoidTy(),
      /*Params=*/std::move(arg_types),
      /*isVarArg=*/false);
  CHECK(m_->getFunction(op->name) == nullptr)
      << "function[" << op->name << "] exists";

  f_ = llvm::Function::Create(
      /*FunctionType=*/function_type,
      /*LinkageTypes=*/llvm::Function::ExternalLinkage,
      /*Name=*/op->name,
      /*Module=*/m_);
  f_->setCallingConv(llvm::CallingConv::C);
  f_->setHasUWTable();  // GDB

  std::vector<llvm::Value *> args;
  args.reserve(f_->arg_size());
  std::transform(
      f_->arg_begin(), f_->arg_end(), std::back_inserter(args), [](auto &arg) {
        return std::addressof(arg);
      });

  llvm::BasicBlock *entry = llvm::BasicBlock::Create(
      /*Context=*/b_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/f_,
      /*InsertBefore=*/nullptr);

  SetVar("_args", args[0]);
  b_->SetInsertPoint(entry);
  Visit(&function_body);
  symbol_table_->Erase("_args");
  RetVoid();
  return f_;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Let *op) {
  CHECK(op->type().valid());
  auto name = op->symbol.As<ir::_Var_>()->name;
  if (op->symbol.As<ir::_Var_>()->type().is_cpp_handle()) {
    alias_vars_.insert(const_cast<ir::_Var_ *>(op->symbol.As<ir::_Var_>()));
  }
  if (op->body.defined()) {
    SetVar(name, Visit(&op->body));
  } else {
    llvm::AllocaInst *inst =
        Alloca(CinnTypeToLLVMType(op->type(), m_), nullptr, name);
    auto get_align = [](int n) {
      int i{0}, r{1};
      while (n > r) {
        r *= 2;
        ++i;
      }
      return r / 8;
    };
    int align_bits = std::max<int>(op->type().bits(), 8);
    int align = get_align(align_bits);
    inst->setAlignment(llvm::Align(align));
    SetVar(name, inst);
  }

  return GetVar(name);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Reduce *op) {
  __IR_EMITTER_NOT_IMPLEMENTED(op);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Ramp *op) {
  __IR_EMITTER_NOT_IMPLEMENTED(op);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Broadcast *op) {
#if LLVM_VERSION_MAJOR >= 11
  const llvm::ElementCount elem_count(op->lanes, /*scalable*/ false);
#else
  const int elem_count = op->lanes;
#endif
  llvm::Value *value = Visit(&op->value);
  llvm::Constant *undef = llvm::UndefValue::get(
      llvm::VectorType::get(value->getType(), elem_count));
  llvm::Constant *zero = llvm::ConstantInt::get(ll_int32_ty(), 0);
  value = b_->CreateInsertElement(undef, value, zero, "broadcast");
  llvm::Constant *zeros = llvm::ConstantVector::getSplat(elem_count, zero);
  return b_->CreateShuffleVector(value, undef, zeros, "broadcast_shuffle");
}

llvm::Value *CodeGenLLVM::Visit(const ir::FracOp *op) {
  __IR_EMITTER_NOT_IMPLEMENTED(op);
}

llvm::Value *CodeGenLLVM::Visit(const ir::Product *op) {
  auto size = op->operands().size();
  if (size == 0) return nullptr;

  llvm::Value *ret = Visit(&op->operand(0));
  for (int i = 1; i < size; i++) {
    llvm::Value *v = Visit(&op->operand(i));
    if (is_integral_type(op->type())) {
      ret = Mul(ret, v);
    } else {
      ret = FMul(ret, v);
    }
  }

  return ret;
}

llvm::Value *CodeGenLLVM::Visit(const ir::Sum *op) {
  auto size = op->operands().size();
  if (size == 0) return nullptr;

  llvm::Value *ret = Visit(&op->operand(0));
  for (int i = 1; i < size; i++) {
    llvm::Value *v = Visit(&op->operand(i));
    if (is_integral_type(op->type())) {
      ret = Add(ret, v);
    } else {  // float
      ret = FAdd(ret, v);
    }
  }

  return ret;
}

#undef __IR_EMITTER_CINN_NOT_IMPLEMENTED

void CodeGenLLVM::Compile(const ir::Module &module) { Visit(module.self()); }

llvm::Value *CodeGenLLVM::EmitCall_buffer_malloc(const ir::Call *op) {
  return nullptr;
}

llvm::Value *CodeGenLLVM::EmitCall_get_address(const ir::Call *op) {
  if (auto *read_var = op->read_args.front().as_var()) {
    return GetVar(read_var->name);
  }

  if (auto *read_buf = op->read_args.front().as_buffer()) {
    return GetVar(read_buf->name);
  }
  return nullptr;
}

llvm::Value *CodeGenLLVM::EmitCall_debug_info(const ir::Call *op) {
  auto callee = m_->getFunction(runtime::intrinsic::debug_log_repr);
  PADDLE_ENFORCE_GE(op->read_args.size(),
                    1UL,
                    ::common::errors::InvalidArgument(
                        "The arguments of debug_log_repr should be greater "
                        "than 1"));
  std::vector<llvm::Value *> args;
  for (auto &arg : op->read_args) {
    args.push_back(Visit(&arg));
  }
  return Call(callee, args, "call debug_info");
}

llvm::Value *CodeGenLLVM::GetVar(const std::string &name, bool lazy) {
  auto symbol = symbol_table_->Lookup(name);
  if (!lazy) {
    CHECK(symbol) << "No var [" << name << "] found";
  }
  return symbol;
}

llvm::Value *CodeGenLLVM::SetVar(const std::string &name, llvm::Value *val) {
  symbol_table_->Insert(name, val);
  CHECK(GetVar(name));
  return val;
}

llvm::FunctionType *CodeGenLLVM::GenFunctionTypeFromCinnFunction(
    const ir::_LoweredFunc_ *func, bool with_buffer_type) {
  auto func_ret_type = CinnTypeToLLVMType(Void(), m_);
  std::vector<llvm::Type *> arg_types;
  for (auto &arg : func->args) {
    if (arg.is_buffer() && arg.is_var()) {
      alias_vars_.insert(arg.var_arg().get());
    }
    if (arg.is_var()) {
      arg_types.push_back(CinnTypeToLLVMType(arg.var_arg()->type(), m_));
    } else if (arg.is_buffer()) {
      if (with_buffer_type) {
        arg_types.push_back(ll_cinn_buffer_p_ty());
      } else {
        arg_types.push_back(CinnTypeToLLVMType(arg.buffer_arg()->type(), m_));
      }
    }
  }

  return llvm::FunctionType::get(func_ret_type, arg_types, false);
}

llvm::Value *CodeGenLLVM::DenseVectorLoad(const ir::Load *op) {
  auto index = op->index();
  auto *ramp = index.As<ir::Ramp>();
  CHECK(ramp);

  int load_lanes = op->type().lanes();
  int native_lanes = naive_vec_alignment_ / op->type().bits();

  std::vector<llvm::Value *> slices;

  llvm::Value *buffer = Visit(&op->tensor);
  buffer->setName("buffer");

  for (int i = 0; i < load_lanes; i += load_lanes) {
    int slice_lanes = load_lanes;
    auto slice_base = optim::ArithSimplify(ramp->base + i);
    optim::VarModSimplify(&slice_base);

#if LLVM_VERSION_MAJOR >= 11
    const llvm::ElementCount elem_count(slice_lanes, /*scalable*/ false);
#else
    const int elem_count = slice_lanes;
#endif

    llvm::Type *slice_type = llvm::VectorType::get(
        CinnTypeToLLVMType(op->type().ElementOf(), m_, true), elem_count);

    llvm::Value *elt_ptr =
        CreateBufferPtr(op->type().ElementOf(), buffer, Visit(&slice_base));
    llvm::Value *vec_ptr = b_->CreatePointerCast(
        elt_ptr, slice_type->getPointerTo(), "get_vec_ptr");

    int alignment = std::max(op->type().ElementOf().bits() / 8, 1);

    llvm::Instruction *load_inst =
        b_->CreateAlignedLoad(vec_ptr, llvm::Align(alignment), "load_vec");
    AddTbaaMetadata(load_inst, op->tensor.as_tensor()->name, op->index());

    slices.push_back(load_inst);
  }

  PADDLE_ENFORCE_EQ(
      slices.size(),
      1UL,
      ::common::errors::InvalidArgument("slices size should be 1."));

  return slices[0];
}

llvm::Value *CodeGenLLVM::CreateBufferVecPtr(Type t,
                                             llvm::Value *buffer,
                                             llvm::Value *index) {
  PADDLE_ENFORCE_GT(
      t.lanes(),
      1,
      ::common::errors::InvalidArgument("type lanes should be greater "
                                        "than 1, but received %d",
                                        t.lanes()));
  llvm::PointerType *btype =
      llvm::dyn_cast<llvm::PointerType>(buffer->getType());
  CHECK(btype);
  llvm::PointerType *ptype =
      CinnTypeToLLVMType(t, m_)->getPointerTo(btype->getAddressSpace());
  if (btype != ptype) {
    buffer = b_->CreatePointerCast(buffer, ptype);
  }
  return b_->CreateInBoundsGEP(buffer, index);
}

llvm::Value *CodeGenLLVM::CreateBufferPtr(Type t,
                                          llvm::Value *buffer,
                                          llvm::Value *index) {
  PADDLE_ENFORCE_EQ(
      t.lanes(),
      1,
      ::common::errors::InvalidArgument("type lanes should be 1, but "
                                        "received %d",
                                        t.lanes()));
  auto *btype = llvm::dyn_cast<llvm::PointerType>(buffer->getType());
  CHECK(btype);
  auto *ptype =
      CinnTypeToLLVMType(t, m_)->getPointerTo(btype->getAddressSpace());
  CHECK(ptype);
  if (btype != ptype) {
    buffer = b_->CreatePointerCast(buffer, ptype, "pointer_cast");
  }
  return b_->CreateInBoundsGEP(buffer, index, "buffer_ptr");
}

llvm::Value *CodeGenLLVM::CreateVecSlice(llvm::Value *vec,
                                         int begin,
                                         int lanes) {
  int total_lanes =
      llvm::dyn_cast<llvm::VectorType>(vec->getType())->getNumElements();
  PADDLE_ENFORCE_LE(begin + lanes,
                    total_lanes,
                    ::common::errors::InvalidArgument(
                        "begin + lanes should be less than total_lanes"));
  if (lanes == total_lanes && begin == 0) return vec;  // full slice
  std::vector<llvm::Constant *> indices;
  for (int i = 0; i < lanes; ++i) {
    indices.push_back(ll_const_int32(begin + i));
  }
  llvm::Constant *undef = llvm::UndefValue::get(vec->getType());
  return b_->CreateShuffleVector(
      vec, undef, llvm::ConstantVector::get(indices));
}

int GetNaiveVecAlignmentImpl(common::UnknownArch, const Target &target) {
  PADDLE_THROW(::common::errors::InvalidArgument("unknown Arch found"));
}

int GetNaiveVecAlignmentImpl(common::X86Arch, const Target &target) {
  if (target.bits == Target::Bit::k32) {
    return 256;
  } else if (target.bits == Target::Bit::k64) {
    return 512;
  }
  PADDLE_THROW(::common::errors::InvalidArgument("get unknown bits"));
}

int GetNaiveVecAlignmentImpl(common::ARMArch, const Target &target) {
  return 128;
}

int GetNaiveVecAlignmentImpl(common::NVGPUArch, const Target &target) {
  return 128;
}

int GetNaiveVecAlignmentImpl(common::HygonDCUArchHIP, const Target &target) {
  return 128;
}

int GetNaiveVecAlignmentImpl(common::HygonDCUArchSYCL, const Target &target) {
  return 128;
}

int GetNaiveVecAlignment(const Target &target) {
  return std::visit(
      [&](const auto &impl) { return GetNaiveVecAlignmentImpl(impl, target); },
      target.arch.variant());
}

void CodeGenLLVM::InitTarget(const Target &target) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  naive_vec_alignment_ = GetNaiveVecAlignment(target);
}

void CodeGenLLVM::AddTbaaMetadata(llvm::Instruction *inst,
                                  absl::string_view buffer,
                                  Expr index) {
  // If the index is constant, generate some TBAA info that helps LLVM
  // understand our loads/stores aren't aliased.
  bool constant_index = false;
  int base = 0;
  int width = 1;

  if (index.defined()) {
    if (const ir::Ramp *ramp = index.As<ir::Ramp>()) {
      auto *pstride_int = ramp->stride.As<ir::IntImm>();
      auto *pbase_int = ramp->base.As<ir::IntImm>();
      if (pstride_int && pbase_int) {
        int stride = pstride_int->value;
        base = pbase_int->value;
        PADDLE_ENFORCE_GE(
            base,
            0,
            ::common::errors::InvalidArgument("base should be greater than 0"));
        width = NextPowerOfTwo(ramp->lanes * stride);

        while (base % width) {
          base -= base % width;
          width *= 2;
        }
        constant_index = true;
      }
    } else {
      auto *pbase_int = index.As<ir::IntImm>();
      if (pbase_int) {
        int pbase = pbase_int->value;
        base = pbase;
        constant_index = true;
      }
    }
  }

  llvm::MDBuilder builder(b_->getContext());

  // Add type-based-alias-analysis metadata to the pointer, so that loads and
  // stores to different buffers can get reordered.
  llvm::MDNode *tbaa = builder.createTBAARoot("cinn buffer");
  tbaa = builder.createTBAAScalarTypeNode(std::string(buffer), tbaa);

  // Add metadata for constant indices to allow loads and stores to the same
  // buffer to get reordered.
  if (constant_index) {
    for (int w = 1024; w >= width; w /= 2) {
      int b = (base / w) * w;
      tbaa = builder.createTBAAScalarTypeNode(
          utils::StringFormat("%s.width%d.base%d", buffer.data(), w, b), tbaa);
    }
  }

  tbaa = builder.createTBAAStructTagNode(tbaa, tbaa, 0);
  inst->setMetadata("tbaa", tbaa);
}

llvm::Value *CodeGenLLVM::Visit(const ir::IntrinsicOp *op) {
  switch (op->getKind()) {
#define __(op__)                   \
  case ir::IntrinsicKind::k##op__: \
    return Visit(llvm::dyn_cast<ir::intrinsics::op__>(op));
    INTRINSIC_KIND_FOR_EACH(__)
#undef __
  }
}

llvm::Value *CodeGenLLVM::Visit(const ir::intrinsics::BufferGetDataHandle *op) {
  std::vector<llvm::Value *> args({Visit(&op->buffer)});
  auto *callee = m_->getFunction("cinn_buffer_get_data_handle");
  return Call(callee, std::move(args));
}

llvm::Value *CodeGenLLVM::Visit(
    const ir::intrinsics::BufferGetDataConstHandle *op) {
  std::vector<llvm::Value *> args({Visit(&op->buffer)});
  auto *callee = m_->getFunction("cinn_buffer_get_data_const_handle");
  return Call(callee, std::move(args));
}

llvm::Value *CodeGenLLVM::Visit(const ir::intrinsics::BufferCreate *op) {
  auto *callee = m_->getFunction(runtime::intrinsic::buffer_create_default);
  auto buffer_node = op->buffer.as_buffer();
  CHECK(buffer_node);
  std::vector<llvm::Value *> args(
      {ll_const_int32(buffer_node->target.runtime_arch())});
  int64_t memory_size = (buffer_node->dtype.ElementOf().bits() + 7) / 8;
  // Calculate buffer size and determine if it contains a symbolic constant
  Expr buffer_size(static_cast<int64_t>(1));
  buffer_size = buffer_size * ir::Expr(memory_size);
  for (int i = 0; i < buffer_node->shape.size(); i++) {
    buffer_size = buffer_size * buffer_node->shape[i];
  }
  ir::TryElevateInt32ToInt64({buffer_size});
  args.push_back(Visit(&buffer_size));
  args.push_back(ll_const_int32(32));

  return Call(callee, args);
}

llvm::Value *CodeGenLLVM::Visit(const ir::intrinsics::GetAddr *op) {
  if (auto *n = op->data.as_var()) {
    return GetVar(n->name);
  } else if (auto *n = op->data.as_buffer()) {
    return GetVar(n->name);
  }
  if (auto *n =
          op->data
              .As<ir::Load>()) {  // get the address to an element in a buffer
    auto *e = Visit(&op->data);
    if (auto *e_load = llvm::dyn_cast<llvm::LoadInst>(e)) {
      return e_load->getPointerOperand();
    }
    return e;
  }
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::intrinsics::ArgsConstruct *op) {
  llvm::SmallVector<llvm::Value *, 7> args;
  Expr var(op->var);
  var->set_type(type_of<cinn_pod_value_t>());
  var = ir::intrinsics::GetAddr::Make(var);

  llvm::Value *ll_var = Visit(&var);
  var = ir::Cast::Make(type_of<cinn_pod_value_t *>(), var);

  Expr num_args(static_cast<int>(op->args.size()));
  args.push_back(
      BitCast(ll_var, ll_cinn_pod_p_ty(), "cast_to_pod_value_t_ptr"));
  args.push_back(Visit(&num_args));
  for (auto &arg : op->args) {
    args.push_back(Visit(&arg));
  }

  auto *callee = m_->getFunction(runtime::intrinsic::args_construct_repr);
  return Call(callee, std::move(args));
}

llvm::Function *CodeGenLLVM::GetIntrinsicDecl(
    llvm::Intrinsic::ID id,
    llvm::Type *ret_type,
    llvm::ArrayRef<llvm::Type *> arg_types) {
  llvm::Module *module = m_;

  if (!llvm::Intrinsic::isOverloaded(id)) {
    return llvm::Intrinsic::getDeclaration(module, id, {});
  }

  llvm::SmallVector<llvm::Intrinsic::IITDescriptor, 4> infos;
  llvm::Intrinsic::getIntrinsicInfoTableEntries(id, infos);
  llvm::SmallVector<llvm::Type *, 4> overload_types;

  auto try_match = [&](llvm::FunctionType *f_ty, bool var_arg) {
    overload_types.clear();
    llvm::ArrayRef<llvm::Intrinsic::IITDescriptor> ref(infos);
    auto match =
        llvm::Intrinsic::matchIntrinsicSignature(f_ty, ref, overload_types);
    if (match == llvm::Intrinsic::MatchIntrinsicTypes_Match) {
      if (llvm::Intrinsic::matchIntrinsicVarArg(var_arg, ref)) {
        return llvm::Intrinsic::MatchIntrinsicTypes_NoMatchArg;
      }
    }
    return match;
  };

  auto *fn_ty = llvm::FunctionType::get(ret_type, arg_types, false);
  switch (try_match(fn_ty, false)) {
    case llvm::Intrinsic::MatchIntrinsicTypes_Match:
      return llvm::Intrinsic::getDeclaration(module, id, overload_types);
    case llvm::Intrinsic::MatchIntrinsicTypes_NoMatchRet:
      return nullptr;
    case llvm::Intrinsic::MatchIntrinsicTypes_NoMatchArg:
      break;
  }

  // try matching the var arg signature.
  llvm::SmallVector<llvm::Type *, 4> var_types;
  for (int i = 0; i <= arg_types.size(); ++i) {
    if (i > 0) {
      var_types.push_back(arg_types[i - 1]);
    }
    auto *ft = llvm::FunctionType::get(ret_type, var_types, true);
    if (try_match(ft, true) == llvm::Intrinsic::MatchIntrinsicTypes_Match) {
      return llvm::Intrinsic::getDeclaration(module, id, overload_types);
    }
  }
  return nullptr;
}

llvm::Value *CodeGenLLVM::Visit(const ir::intrinsics::BuiltinIntrin *op) {
  std::string func_name = op->name;
  if (op->id == -1) {
    if (func_name == "bitwise_and") {
      PADDLE_ENFORCE_GE(op->args.size(),
                        2U,
                        ::common::errors::InvalidArgument(
                            "bitwise_and should have at least 2 arguments"));
      return b_->CreateAnd(Visit(&op->args[0]), Visit(&op->args[1]));
    } else if (func_name == "bitwise_or") {
      PADDLE_ENFORCE_GE(op->args.size(),
                        2U,
                        ::common::errors::InvalidArgument(
                            "bitwise_or should have at least 2 arguments"));
      return b_->CreateOr(Visit(&op->args[0]), Visit(&op->args[1]));
    } else if (func_name == "bitwise_xor") {
      PADDLE_ENFORCE_GE(op->args.size(),
                        2U,
                        ::common::errors::InvalidArgument(
                            "bitwise_xor should have at least 2 arguments"));
      return b_->CreateXor(Visit(&op->args[0]), Visit(&op->args[1]));
    } else if (func_name == "bitwise_not") {
      PADDLE_ENFORCE_GE(op->args.size(),
                        1U,
                        ::common::errors::InvalidArgument(
                            "bitwise_not should have at least 1 argument"));
      return b_->CreateNot(Visit(&op->args[0]));
    } else if (func_name == "left_shift") {
      PADDLE_ENFORCE_GE(op->args.size(),
                        2U,
                        ::common::errors::InvalidArgument(
                            "left_shift should have at least 2 arguments"));
      return b_->CreateShl(Visit(&op->args[0]), Visit(&op->args[1]));
    } else if (func_name == "right_shift") {
      PADDLE_ENFORCE_GE(op->args.size(),
                        2U,
                        ::common::errors::InvalidArgument(
                            "right_shift should have at least 2 arguments"));
      if (op->args[0]->type().is_int()) {
        return b_->CreateAShr(Visit(&op->args[0]), Visit(&op->args[1]));
      } else {
        return b_->CreateLShr(Visit(&op->args[0]), Visit(&op->args[1]));
      }
    } else if (func_name == "isnan") {
      PADDLE_ENFORCE_GE(op->args.size(),
                        1U,
                        ::common::errors::InvalidArgument(
                            "isnan should have at least 1 argument"));
      llvm::Value *v = Visit(&op->args[0]);
      return b_->CreateFCmpUNO(v, v);
    }
  }

  llvm::Intrinsic::ID id = op->id;
  int64_t num_signature = op->arg_nums;
  std::vector<llvm::Value *> arg_value;
  std::vector<llvm::Type *> arg_type;
  for (size_t i = 0; i < op->args.size(); ++i) {
    arg_value.push_back(Visit(&op->args[i]));
    if (i < static_cast<size_t>(num_signature)) {
      arg_type.push_back(arg_value.back()->getType());
    }
  }
  CHECK(!op->args.empty());
  llvm::Type *return_type = CinnTypeToLLVMType(op->type(), m_, true);
  llvm::Function *fn = GetIntrinsicDecl(id, return_type, arg_type);
  CHECK(fn) << "Cannot find intrinsic declaration, possible type mismatch: "
            << llvm::Intrinsic::getName(id, {});
  return b_->CreateCall(fn, arg_value);
}

llvm::Value *CodeGenLLVM::Visit(const ir::intrinsics::PodValueToX *op) {
  auto to_type = op->GetOutputType(0);
  llvm::Function *callee{};

  if (to_type == type_of<float>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_float);
  } else if (to_type == type_of<double>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_double);
  } else if (to_type == type_of<bfloat16>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_bfloat16);
  } else if (to_type == type_of<float16>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_float16);
  } else if (to_type == type_of<bool>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_bool);
  } else if (to_type == type_of<int8_t>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_int8);
  } else if (to_type == type_of<int16_t>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_int16);
  } else if (to_type == type_of<int32_t>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_int32);
  } else if (to_type == type_of<int64_t>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_int64);
  } else if (to_type == type_of<uint8_t>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_uint8);
  } else if (to_type == type_of<uint16_t>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_uint16);
  } else if (to_type == type_of<uint32_t>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_uint32);
  } else if (to_type == type_of<uint64_t>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_uint64);
  } else if (to_type == type_of<void *>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_void_p);
  } else if (to_type == type_of<cinn_buffer_t *>()) {
    callee = m_->getFunction(runtime::intrinsic::pod_value_to_buffer_p);
  } else {
    std::stringstream ss;
    ss << "Not supported type: " << to_type;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }

  CHECK(callee);
  auto *value = Visit(&op->pod_value_ptr);
  CHECK(value);
  return Call(callee, std::vector<llvm::Value *>({value}), "pod_value_cast");
}

}  // namespace backends
}  // namespace cinn
