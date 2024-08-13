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

#include "paddle/cinn/backends/llvm/codegen_x86.h"

#include <absl/container/flat_hash_map.h>
#include <llvm/IR/LLVMContext.h>

#include <algorithm>
#include <utility>

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"
#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/common/enforce.h"
namespace cinn::backends {

CodeGenX86::CodeGenX86(llvm::Module* m,
                       llvm::IRBuilder<>* b,
                       const std::shared_ptr<SymbolTable>& vars)
    : CodeGenLLVM(m, b, vars) {}

CodeGenX86::~CodeGenX86() {}

llvm::Value* CodeGenX86::PackVars(const std::vector<std::string>& vars,
                                  uint64_t* num_bytes) {
  if (vars.empty()) {
    *num_bytes = 0U;
    return llvm::Constant::getNullValue(ll_void_p_ty());
  }
  std::vector<llvm::Type*> types;
  for (auto& v : vars) {
    types.push_back(GetVar(v, false)->getType());
  }
  llvm::StructType* t_data = llvm::StructType::create(types);
  llvm::Value* data = b_->CreateAlloca(t_data, llvm_int32_constant(1));
  for (size_t i = 0; i < vars.size(); ++i) {
    b_->CreateStore(
        GetVar(vars[i]),
        b_->CreateInBoundsGEP(
            data, {llvm_int32_constant(0), llvm_int32_constant(i)}));
  }
  *num_bytes = m_->getDataLayout().getTypeAllocSize(
      llvm::cast<llvm::PointerType>(data->getType())->getElementType());
  return data;
}

void CodeGenX86::UnpackVars(const std::vector<std::string>& vars,
                            llvm::Value* data) {
  for (size_t i = 0; i < vars.size(); ++i) {
    SetVar(vars[i],
           b_->CreateLoad(b_->CreateInBoundsGEP(
               data, {llvm_int32_constant(0), llvm_int32_constant(i)})));
  }
}

llvm::BasicBlock* CodeGenX86::CheckCallSuccess(llvm::Value* retcode) {
  llvm::BasicBlock* fail_block =
      llvm::BasicBlock::Create(b_->getContext(),
                               "call_fail",
                               b_->GetInsertBlock()->getParent(),
                               nullptr);
  llvm::BasicBlock* end_block = llvm::BasicBlock::Create(
      b_->getContext(), "call_end", b_->GetInsertBlock()->getParent(), nullptr);
  llvm::Value* succ =
      b_->CreateICmpEQ(retcode, llvm::ConstantInt::get(ll_int32_ty(), 0));
  b_->CreateCondBr(succ, end_block, fail_block);
  b_->SetInsertPoint(fail_block);
  RetVoid();
  b_->SetInsertPoint(end_block);
  return end_block;
}

void CodeGenX86::CreateParallelLaunch(Expr body, int num_task) {
  auto ftype_parallel_lambda = llvm::FunctionType::get(
      ll_int32_ty(),
      {ll_int32_ty(), ll_int32_ty(), ll_type_of(Float(32).PointerOf())},
      false);
  llvm::Function* f = llvm::Function::Create(ftype_parallel_lambda,
                                             llvm::Function::PrivateLinkage,
                                             "__parallel_lambda",
                                             m_);
  std::vector<std::string> vars = ir::ir_utils::CollectUndefinedVars(&body);
  uint64_t nbytes;
  auto* data = PackVars(vars, &nbytes);

  auto ftype_parallel_launch =
      llvm::FunctionType::get(ll_int32_ty(),
                              {ftype_parallel_lambda->getPointerTo(),
                               ll_type_of(Float(32).PointerOf()),
                               ll_int32_ty()},
                              false);
  auto* launch_callee = llvm::dyn_cast<llvm::Function>(
      m_->getOrInsertFunction(runtime::intrinsic::parallel_launch,
                              ftype_parallel_launch)
          .getCallee());
  launch_callee->setCallingConv(llvm::CallingConv::C);
  auto* launch_end = CheckCallSuccess(b_->CreateCall(
      launch_callee,
      {f,
       b_->CreatePointerCast(data, ll_type_of(Float(32).PointerOf())),
       llvm_int32_constant(num_task)}));

  auto* flambda = llvm::BasicBlock::Create(b_->getContext(), "flambda", f);
  b_->SetInsertPoint(flambda);
  auto it = f->arg_begin();
  auto* task_id = &(*it++);
  auto* penv = &(*it++);
  data = b_->CreatePointerCast(&(*it++), data->getType());
  symbol_table_->PushScope();
  UnpackVars(vars, data);
  ParallelEnv par_env;
  auto task_id_name = cinn::common::UniqName("task_id");
  auto num_task_name = cinn::common::UniqName("num_task");
  par_env.task_id = ir::Var(task_id_name, Int(32));
  par_env.num_task = ir::Var(num_task_name, Int(32));
  SetVar(task_id_name, task_id);
  SetVar(num_task_name, penv);
  par_env.penv = penv;
  std::swap(f_, f);
  std::swap(parallel_env_, par_env);
  this->Visit(&body);
  b_->CreateRet(ll_const_int32(0));
  symbol_table_->Erase(task_id_name);
  symbol_table_->Erase(num_task_name);
  symbol_table_->PopScope();
  std::swap(parallel_env_, par_env);
  std::swap(f_, f);
  PADDLE_ENFORCE_NE(par_env.parallel_loop_count,
                    0,
                    ::common::errors::InvalidArgument(
                        "find no parallel loop within parallel launch"));
  b_->SetInsertPoint(launch_end);
}

llvm::Value* CodeGenX86::Visit(const ir::For* op) {
  if (op->is_parallel()) {
    VLOG(3) << "parallel forloop";
    if (parallel_env_.penv == nullptr) {
      CreateParallelLaunch(ir::For::Make(op->loop_var,
                                         op->min,
                                         op->extent,
                                         op->for_type(),
                                         op->device_api,
                                         op->body,
                                         op->vectorize_info()),
                           0);
    } else {
      Expr num_task = parallel_env_.num_task;
      Expr task_id = parallel_env_.task_id;
      PADDLE_ENFORCE_EQ(parallel_env_.in_parallel_loop,
                        false,
                        ::common::errors::InvalidArgument(
                            "Nested parallel loop is not supported, "
                            "try to fuse them instead"));
      parallel_env_.in_parallel_loop = true;
      if (parallel_env_.stride_pattern) {
        auto new_for = ir::For::Make(op->loop_var,
                                     task_id,
                                     op->extent,
                                     op->for_type(),
                                     op->device_api,
                                     op->body,
                                     op->vectorize_info());
        auto for_node = new_for.As<ir::For>();
        PADDLE_ENFORCE_NOT_NULL(for_node,
                                ::common::errors::InvalidArgument(
                                    "the node new_for can't be nullptr"));
        CreateSerialFor(for_node, num_task.as_int32());
      } else {
        Expr extent = op->extent;
        Expr step = (extent + num_task - Expr(1)) / num_task;
        Expr begin = min(task_id * step, op->extent);
        Expr end = min((task_id + Expr(1)) * step, op->extent);
        auto new_for = ir::For::Make(op->loop_var,
                                     begin,
                                     end,
                                     op->for_type(),
                                     op->device_api,
                                     op->body,
                                     op->vectorize_info());
        auto for_node = new_for.As<ir::For>();
        PADDLE_ENFORCE_NOT_NULL(for_node,
                                ::common::errors::InvalidArgument(
                                    "the node new_for can't be null"));
        CreateSerialFor(for_node);
      }
      parallel_env_.in_parallel_loop = false;
      ++parallel_env_.parallel_loop_count;
    }
  } else {
    return CodeGenLLVM::Visit(op);
  }
  return nullptr;
}
}  // namespace cinn::backends
