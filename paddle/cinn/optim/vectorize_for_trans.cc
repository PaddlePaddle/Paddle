// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/vectorize_for_trans.h"

#include <stack>
#include <vector>
#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/optim/unroll_loops.h"

namespace cinn {
namespace optim {

namespace {

class VectorizeForTransMutator : public ir::IRMutator<ir::Expr *> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Load *op, ir::Expr *expr) override {
    if (in_vectorize_) {
      auto *node = expr->As<ir::Load>();
      auto *tensor = node->tensor.As<ir::_Tensor_>();
      if (node->is_addr_tensor()) {
        TensorVectorized(node, &node->indices, false);
      }
    }
  }

  void Visit(const ir::Store *op, ir::Expr *expr) override {
    auto *node = expr->As<ir::Store>();
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in Store, but received nullptr."));
    if (in_vectorize_) {
      TensorVectorized(node, &node->indices, true);
    }
    IRMutator::Visit(&node->value, &node->value);
  }

  void Visit(const ir::For *op, ir::Expr *expr) override {
    auto *forloop = expr->As<ir::For>();
    if (op->is_vectorized()) {
      auto loop_var_var_name = forloop->loop_var->name;
      vectorize_size_ = forloop->vectorize_info().factor;
      loop_var_ = op->loop_var;
      vec_for_ = expr->As<ir::For>();
      in_vectorize_ = true;
    }

    // deal with vectorize Tensor load and store
    IRMutator::Visit(forloop, expr);

    if (in_vectorize_) {
      const int factor = forloop->vectorize_info().factor;
      auto _new_forloop = SplitForLoop(forloop, factor);
      if (!_new_forloop.defined()) {
        IRMutator<>::Visit(&forloop->body, &forloop->body);
        return;
      }
      auto *new_forloop = _new_forloop.As<ir::For>();
      auto copied_loop =
          ir::ir_utils::IRCopy(new_forloop, /* copy_buffer_node = */ false);
      copied_loop.As<ir::For>()->set_unrolled();
      optim::UnrollLoop(&copied_loop);
      auto unroll_body = copied_loop.As<ir::Block>()->stmts;
      auto &body_stmts = new_forloop->body.As<ir::Block>()->stmts;
      if (!update_cast_stmts_.empty()) {
        body_stmts.assign(update_cast_stmts_.begin(), update_cast_stmts_.end());
        update_cast_stmts_.clear();
      }
      body_stmts.insert(
          body_stmts.end(), unroll_body.begin(), unroll_body.end());

      if (!update_store_stmts_.empty()) {
        body_stmts.insert(body_stmts.end(),
                          update_store_stmts_.begin(),
                          update_store_stmts_.end());
        update_store_stmts_.clear();
      }
      *expr = new_forloop->body;
    }

    in_vectorize_ = false;
  }

 private:
  std::string GetVectorTypeName(ir::Type type) {
    std::string name_prefix =
        cinn::common::customized_type::kcuda_builtin_vector_t;
#define GET_CUDA_VECTOR_TYPE_NAME(pred_expr, scalar_name)               \
  if (pred_expr) {                                                      \
    return name_prefix + scalar_name + std::to_string(vectorize_size_); \
  }

    GET_CUDA_VECTOR_TYPE_NAME(type.is_int(32), "int");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_uint(32), "uint");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_float(32), "float");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_float16(), "half");
    GET_CUDA_VECTOR_TYPE_NAME(type.is_bfloat16(), "bfloat16");
#undef GET_CUDA_VECTOR_TYPE_NAME

    // others are not implemented yet
    CINN_NOT_IMPLEMENTED
    return "";
  }

  void TensorVectorized(ir::LoadStoreAddrMnger *node,
                        std::vector<ir::Expr> *indices,
                        bool is_store) {
    auto *tensor = node->tensor.As<ir::_Tensor_>();
    VLOG(5) << "Vectorizing tensor:" << tensor->name;

    if (!tensor2vectorized_vars_.count(tensor->name)) {
      AppendCast(node->tensor, *indices, is_store);
    }

    auto vectorized_var = tensor2vectorized_vars_.at(tensor->name);
    // substitute a new tensor with the vector name and dtype
    auto t = vectorized_var->type().is_cpp_handle()
                 ? node->tensor->type().PointerOf()
                 : node->tensor->type();
    node->tensor = ir::Tensor(vectorized_var->name,
                              t,
                              {ir::Expr(vectorize_size_)},
                              {ir::Expr(vectorize_size_)},
                              tensor->operation);
    // remain the last iterative indice
    indices->assign({loop_var_});
  }

  void AppendCast(ir::Expr tensor,
                  const std::vector<ir::Expr> &indices,
                  bool is_store) {
    auto *node = tensor.As<ir::_Tensor_>();

    // generate the corresponding vector type
    Type scalar_type = tensor->type().ElementOf();
    Type vector_type_ptr(
        ir::Type::type_t::Customized, scalar_type.bits(), vectorize_size_);
    Type vector_type(
        ir::Type::type_t::Customized, scalar_type.bits(), vectorize_size_);
    vector_type_ptr.set_customized_type(GetVectorTypeName(scalar_type));
    vector_type_ptr.set_cpp_handle();
    vector_type_ptr.set_cpp_const(false);

    vector_type.set_customized_type(GetVectorTypeName(scalar_type));
    vector_type.set_cpp_const(false);

    // generate a local vector variable to be used in subsequent statements
    std::string vectorized_name = "vectorized_" + node->name;
    Var vectorized_var = ir::_Var_::Make(vectorized_name, vector_type);
    tensor2vectorized_vars_.emplace(node->name, vectorized_var);

    // generate a get_addr expr to get the address of the tensor
    Expr converted_tensor = ir::Load::Make(tensor, indices);
    cinn::ir::ir_utils::IrReplaceVarBroadcast(
        &converted_tensor, loop_var_, Expr(int32_t(0)));
    auto get_addr = ir::intrinsics::GetAddr::Make(converted_tensor);

    // generate a let expression to cast the tensor into the local vector
    auto cast = ir::Cast::Make(vector_type_ptr, get_addr);
    if (!is_store) {
      auto load = ir::Load::Make(cast, {cinn::common::make_const(0)});
      auto let = ir::Let::Make(vectorized_var, load);
      update_cast_stmts_.emplace_back(let);
      VLOG(5) << "Append a vectorized expr:" << let;
    } else {
      Var vectorized_ptr =
          ir::_Var_::Make(vectorized_name + "_ptr", vector_type_ptr);

      auto let1 = ir::Let::Make(vectorized_ptr, cast);
      auto let2 = ir::Let::Make(vectorized_var, ir::Expr(0));
      update_cast_stmts_.emplace_back(let1);
      update_cast_stmts_.emplace_back(let2);

      VLOG(5) << "Append a vectorized expr:" << let1;
      VLOG(5) << "Append a vectorized expr:" << let2;

      auto t = ir::Tensor(vectorized_ptr->name,
                          node->type().PointerOf(),
                          {ir::Expr(vectorize_size_)},
                          {ir::Expr(vectorize_size_)},
                          node->operation);
      auto store =
          ir::Store::Make(t, vectorized_var, {cinn::common::make_const(0)});
      update_store_stmts_.emplace_back(store);
      VLOG(5) << "Append a vectorized expr:" << store;
    }
  }

  Expr SplitForLoop(ir::For *forloop, int factor) {
    forloop->set_vectorized(false);
    Var new_iterator(Context::Global().NewName("vi"));
    Expr new_index;
    new_index = Expr(new_iterator);
    cinn::ir::ir_utils::IrReplaceVarBroadcast(
        &forloop->body, forloop->loop_var, new_index);
    auto new_forloop = ir::For::Make(new_iterator,
                                     forloop->min,
                                     cinn::common::make_const(factor),
                                     ir::ForType::Vectorized,
                                     ir::DeviceAPI::UNK,
                                     forloop->body,
                                     forloop->vectorize_info());
    forloop->body = ir::Block::Make({new_forloop});
    return new_forloop;
  }

  std::vector<ir::Expr> update_cast_stmts_;
  std::vector<ir::Expr> update_store_stmts_;
  absl::flat_hash_map<std::string, ir::Var> tensor2vectorized_vars_;
  int vectorize_size_{0};
  ir::Var loop_var_;
  ir::For *vec_for_{nullptr};
  bool in_vectorize_{false};
};

}  // namespace

void VectorizeForTrans(Expr *expr) {
  VectorizeForTransMutator collector;
  VLOG(5) << "before vectorize for trans " << *expr;
  collector(expr);
  VLOG(5) << "after vectorize for trans " << *expr;
}

}  // namespace optim
}  // namespace cinn
