// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/realize_composite_reduce_pass.h"
#include "paddle/cinn/hlir/pe/reduction.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/phi/core/enforce.h"

namespace cinn {
namespace optim {

using ir::stmt::Alloc;
using ir::stmt::BlockRef;
using ir::stmt::Evaluate;
using ir::stmt::For;
using ir::stmt::Free;
using ir::stmt::IfThenElse;
using ir::stmt::Let;
using ir::stmt::Schedule;
using ir::stmt::StmtRef;
using ir::stmt::Store;
using ReduceType = ir::Reduce::ReduceType;

namespace {
ReduceType GetReduceType(const ir::Expr& expr) {
  if (auto it = expr.As<ir::Call>()) {
    if (it->name == hlir::pe::kVarianceFuncName) {
      return ReduceType::kVariance;
    } else if (it->name == hlir::pe::kArgmaxFuncName) {
      return ReduceType::kArgmax;
    } else if (it->name == hlir::pe::kArgminFuncName) {
      return ReduceType::kArgmin;
    }
  }
  return ReduceType::kNone;
}

struct CompositeTypes : public std::vector<common::Type> {
  ReduceType type = ReduceType::kNone;
  explicit CompositeTypes(ReduceType _type = ReduceType::kNone) : type(_type) {
    this->reserve(2);
  }

  void Print() const {
    VLOG(4) << "[CompositeTypes]: ";
    for (auto _t : *this) {
      VLOG(4) << _t;
    }
  }
};

CompositeTypes GetArgReduceUnderlyingType(const ir::Expr& expr) {
  if (auto it = expr.As<ir::Call>()) {
    if (it->name == hlir::pe::kArgmaxFuncName ||
        it->name == hlir::pe::kArgminFuncName) {
      // for cinn_argxxx func, the arg1 is the argidx
      // we need to check the type of the input
      auto argidx_call = it->read_args[1].As<ir::Call>();
      if (argidx_call != nullptr && argidx_call->name.find("argidx_") == 0) {
        CompositeTypes comp_types(it->name == hlir::pe::kArgminFuncName
                                      ? ReduceType::kArgmin
                                      : ReduceType::kArgmax);
        comp_types.push_back(argidx_call->read_args[0]->type());
        comp_types.push_back(expr->type());
        return comp_types;
      }
    } else if (it->name == hlir::pe::kVarianceFuncName) {
      return CompositeTypes(ReduceType::kVariance);
    }
  }
  return CompositeTypes();
}

void SetInitValue(Store store_stmt,
                  common::Type new_type,
                  const CompositeTypes& comp_type) {
  ir::Expr init_value = store_stmt->value();
  if (comp_type.type == ReduceType::kVariance) {
    store_stmt->set_value(ir::Call::Make(new_type,
                                         new_type.customized_type(),
                                         {init_value, init_value, init_value},
                                         {},
                                         ir::CallType::Intrinsic));
  } else if (comp_type.type == ReduceType::kArgmax ||
             comp_type.type == ReduceType::kArgmin) {
    ir::Expr index_init = ir::Expr(0);
    index_init->set_type(common::Int(32));
    if (comp_type.at(1).is_int(64)) {
      index_init->set_type(common::Int(64));
    }
    store_stmt->set_value(ir::Call::Make(new_type,
                                         new_type.customized_type(),
                                         {init_value, index_init},
                                         {},
                                         ir::CallType::Intrinsic));
  } else {
    PADDLE_THROW(::common::errors::Unimplemented(
        "reduce_type '%d' not allowed.", static_cast<int>(comp_type.type)));
  }
}

template <typename StmtType>
std::set<ir::Buffer> CollectReduceBuffers(const BlockRef& body) {
  std::set<ir::Buffer> buffers;

  const auto VisitFn = [&](const StmtRef& stmt) {
    if (!stmt.isa<StmtType>()) return;
    StmtType store_stmt = stmt.as<StmtType>();
    if (GetReduceType(store_stmt->value()) != ReduceType::kNone) {
      buffers.insert(store_stmt->tensor().as_tensor()->buffer);
    }
  };

  ir::stmt::Visit(body, VisitFn, [](auto) {});
  return buffers;
}

CompositeTypes DetermineReduceType(const BlockRef& body) {
  CompositeTypes composite_reduce;

  // the blockref can only have one unique reduce type
  const auto VisitFn = [&composite_reduce](const StmtRef& stmt) {
    // to stop recursive visit: (1) rtype is defined
    // (2) stmt is not a Store stmt
    if (composite_reduce.type != ReduceType::kNone || !stmt.isa<Store>())
      return;
    Store store_stmt = stmt.as<Store>();
    composite_reduce = GetArgReduceUnderlyingType(store_stmt->value());
  };

  ir::stmt::Visit(body, VisitFn, [](auto) {});
  return composite_reduce;
}

Store GetStoreOfSchedule(const Schedule& stmt) {
  Store store_stmt;
  bool found = false;
  const auto VisitFn = [&](StmtRef stmt) {
    if (!found && stmt.isa<Store>()) {
      store_stmt = stmt.as<Store>();
      found = true;
    }
  };
  ir::stmt::Visit(stmt->body(), VisitFn, [](auto) {});
  PADDLE_ENFORCE_EQ(found,
                    true,
                    ::common::errors::PreconditionNotMet(
                        "One Schedule should have exactly one Store."));
  return store_stmt;
}

Type GetCompositeReduceType(const Type& elem_type,
                            const CompositeTypes& composite_reduce) {
  int type_bits = 0;
  std::string rtype_name = "";
  if (composite_reduce.type == ReduceType::kVariance) {
    type_bits = elem_type.bits() * 3;
    rtype_name = "welford" + hlir::pe::Type2StrForReduce(elem_type);
  } else if (composite_reduce.type == ReduceType::kArgmax ||
             composite_reduce.type == ReduceType::kArgmin) {
    PADDLE_ENFORCE_GT(
        composite_reduce.size(),
        1,
        ::common::errors::InvalidArgument("CompositeTypes for arg reduce "
                                          "must have at least two types"));
    int max_bits =
        std::max(composite_reduce[0].bits(), composite_reduce[1].bits());
    type_bits = max_bits * 2;
    rtype_name = "argidx" +
                 hlir::pe::Type2StrForArgReduce(composite_reduce[0]) +
                 hlir::pe::Type2StrForArgReduce(composite_reduce[1]);
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Unsupported composite reduce type: %d",
        static_cast<int>(composite_reduce.type)));
  }
  Type customized_type(ir::Type::type_t::Customized,
                       /* bits = */ type_bits,
                       /* width = */ 1);
  customized_type.set_customized_type(rtype_name);
  customized_type.set_cpp_const(false);
  return customized_type;
}

struct StageReduceResultMutator : public ir::stmt::StmtMutator<> {
  explicit StageReduceResultMutator(ir::LoweredFunc func) : func_(func) {
    for (auto& arg : func->args) {
      if (arg.is_buffer()) arg_buffers_.insert(arg.buffer_arg());
    }
  }

  void operator()(BlockRef block) { VisitBlock(block); }

 private:
  void VisitStmt(Schedule stmt) override {
    if (stmt->name().substr(0, 4) == "root") {
      ir::stmt::StmtMutator<>::VisitBlock(stmt->body());
      return;
    }
    Store store_stmt = GetStoreOfSchedule(stmt.as<Schedule>());
    auto* store_tensor = store_stmt->tensor().as_tensor();
    if (GetReduceType(store_stmt->value()) == ReduceType::kNone) return;
    if (arg_buffers_.count(store_tensor->buffer) == 0) return;

    // Create the staging buffer.
    // We only need one element for this buffer, so its shape is {1}.
    const std::vector<ir::Expr> shape = {ir::Expr(1)};
    const std::vector<ir::Expr> indices = {ir::Expr(0)};
    ir::Tensor staging_tensor =
        ir::_Tensor_::Make(common::UniqName(store_tensor->name + "_local"),
                           store_tensor->buffer->dtype,
                           shape,
                           shape);
    staging_tensor->WithBuffer("local", staging_tensor->name + "_buffer");
    func_->temp_bufs.push_back(staging_tensor->buffer);

    // Create the staging Schedule.
    Schedule staging_schedule(stmt->iter_vars(),
                              stmt->iter_values(),
                              stmt->read_buffers(),
                              stmt->write_buffers(),
                              staging_tensor->name,
                              ir::ir_utils::IRCopy(stmt->body()),
                              stmt->attrs(),
                              stmt->reduce_method());
    sibling_stmts_.push_back(staging_schedule);

    // Replace all uses of the composite reduce buffer with the staging buffer.
    Store staging_store = GetStoreOfSchedule(staging_schedule);
    staging_store->set_tensor(staging_tensor);
    staging_store->set_indices(indices);
    ir::Expr staging_value = staging_store->value();
    staging_value.As<ir::Call>()->read_args[0] =
        ir::Load::Make(staging_tensor, indices);
    staging_store->set_value(staging_value);
    store_stmt->set_value(ir::Load::Make(staging_tensor, indices));

    // Remove the reduction flags in the current Schedule, because reduction
    // has been done in the staging Schedule.
    std::vector<ir::Var> new_iter_vars;
    for (auto& var : stmt->iter_vars()) {
      ir::Var new_var = var->Copy().as_var_ref();
      new_var->is_reduce_axis = false;
      new_iter_vars.push_back(new_var);
    }
    stmt->set_iter_vars(new_iter_vars);
  }

  void VisitBlock(BlockRef block) override {
    std::vector<StmtRef> old_stmts;
    old_stmts.swap(sibling_stmts_);

    for (StmtRef stmt : block->stmts()) {
      ir::stmt::StmtMutator<>::VisitStmt(stmt);
      sibling_stmts_.push_back(stmt);
    }

    block->set_stmts(sibling_stmts_);
    sibling_stmts_ = std::move(old_stmts);
  }

  void VisitStmt(For stmt) override { VisitBlock(stmt->body()); }

  void VisitStmt(IfThenElse stmt) override {
    ir::stmt::BlockRef true_case = stmt->true_case();
    VisitBlock(true_case);
    stmt->set_true_case(true_case);
    if (stmt->false_case().defined()) {
      ir::stmt::BlockRef false_case = stmt->false_case();
      VisitBlock(false_case);
      stmt->set_false_case(false_case);
    }
  }

  void VisitStmt(Let stmt) override { return; }
  void VisitStmt(Store stmt) override { return; }
  void VisitStmt(Alloc stmt) override { return; }
  void VisitStmt(Free stmt) override { return; }
  void VisitStmt(Evaluate stmt) override { return; }

 private:
  ir::LoweredFunc func_;
  // buffers in the function's argument list
  std::set<ir::Buffer> arg_buffers_;
  // stmts at the same level with the currently visiting stmt
  std::vector<StmtRef> sibling_stmts_;
};

struct LoadTypeMutator : public ir::IRMutator<> {
  explicit LoadTypeMutator(const std::map<ir::Buffer, ir::Type>& buffer2type,
                           const CompositeTypes& ctype)
      : buffer2type_(buffer2type), comp_type_(ctype) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Load* op, ir::Expr* expr) override {
    ir::IRMutator<>::Visit(op, expr);
    auto* node = expr->As<ir::Load>();
    auto& buffer = node->tensor.as_tensor()->buffer;
    auto it = buffer2type_.find(buffer);
    if (it != buffer2type_.end()) {
      ir::Type new_type = GetCompositeReduceType(it->second, comp_type_);
      node->tensor.as_tensor()->set_type(new_type);
      buffer->dtype = new_type;
      *expr = ir::Cast::Make(it->second, *expr);
    }
  }

  void UncastType(ir::Expr* expr) {
    auto* cast_node = expr->As<ir::Cast>();
    if (!cast_node) return;
    auto* load_node = cast_node->v().As<ir::Load>();
    if (!load_node) return;
    if (buffer2type_.count(load_node->tensor.as_tensor()->buffer) > 0) {
      *expr = cast_node->v();
    }
  }

  void Visit(const ir::Call* op, ir::Expr* expr) override {
    // this function will cast the buffer from composite type
    // to an underlying type, for example welford_fp32 -> float
    // uncast will undo this process
    ir::IRMutator<>::Visit(op, expr);
    // By default, all tensors are casted back to their element type
    // before doing other computation. However, for the composite reduction
    // calls, we shouldn't cast the arguments back because they hold the
    // intermediate status.
    if (GetReduceType(*expr) != ReduceType::kNone) {
      auto* node = expr->As<ir::Call>();
      UncastType(&(node->read_args[0]));
      UncastType(&(node->read_args[1]));
    }
  }

  const std::map<ir::Buffer, ir::Type>& buffer2type_;
  const CompositeTypes& comp_type_;
};

void SetBufferType(ir::LoweredFunc func,
                   const std::set<ir::Buffer>& buffers,
                   const CompositeTypes& composite_type) {
  // Make a map from the buffers to their element types, otherwise it's hard to
  // know a buffer's original type.
  std::map<ir::Buffer, ir::Type> buffer2type;
  for (auto& buffer : buffers) {
    buffer2type.emplace(buffer, buffer->dtype);
  }

  // Set function's temp_bufs type
  for (auto& buffer : func->temp_bufs) {
    auto it = buffer2type.find(buffer);
    if (it != buffer2type.end()) {
      buffer->dtype = GetCompositeReduceType(it->second, composite_type);
    }
  }

  const auto VisitFn = [&](StmtRef stmt) {
    if (!stmt.isa<Store>()) return;
    Store store_stmt = stmt.as<Store>();
    auto* tensor = store_stmt->tensor().as_tensor();
    auto& buffer = tensor->buffer;

    // Set store buffer type
    auto it = buffer2type.find(buffer);
    if (it != buffer2type.end()) {
      ir::Expr new_tensor = ir::ir_utils::IRCopy(store_stmt->tensor());
      ir::Type new_type = GetCompositeReduceType(it->second, composite_type);
      new_tensor.as_tensor()->set_type(new_type);
      new_tensor.as_tensor()->buffer->dtype = new_type;
      store_stmt->set_tensor(new_tensor);

      if (ir::IsReduceInitTensorName(tensor->name)) {
        SetInitValue(store_stmt, new_type, composite_type);
      }
    }

    // Set load buffer type
    ir::Expr new_value = ir::ir_utils::IRCopy(store_stmt->value());
    LoadTypeMutator load_type_mutator(buffer2type, composite_type);
    load_type_mutator(&new_value);
    store_stmt->set_value(new_value);
  };

  ir::stmt::Mutate(func->body_block, VisitFn, [](auto) {});
}

struct ReduceExternCallMutator : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Call* op, ir::Expr* expr) override {
    ir::IRMutator<>::Visit(op, expr);
    auto reduce_type_ = GetReduceType(*expr);
    if (reduce_type_ == ReduceType::kNone) return;
    ir::Expr lhs = op->read_args[0];
    ir::Expr rhs = op->read_args[1];
    if (lhs.type() != rhs.type()) {
      if (auto call_op = rhs.As<ir::Call>()) {
        // for argidx type, avoid redundant type casting, but this is ugly
        if (call_op->name.find("argidx") != std::string::npos) {
          rhs->set_type(lhs.type());
        }
      } else {
        rhs = ir::Cast::Make(lhs.type(), rhs);
      }
    }
    if (reduce_type_ == ReduceType::kVariance) {
      // replace cinn_reduce_variance to operator+
      *expr = ir::Add::Make(lhs, rhs);
    } else if (reduce_type_ == ReduceType::kArgmax ||
               reduce_type_ == ReduceType::kArgmin) {
      // replace cinn_argmxx_iyy to max or min (overloaded)
      if (op->name.find("argmax") != std::string::npos) {
        *expr = ir::Max::Make(lhs, rhs);
      } else {
        *expr = ir::Min::Make(lhs, rhs);
      }
    }
  }
};

void ReplaceReduceExternCall(const BlockRef& body) {
  const auto VisitFn = [&](StmtRef stmt) {
    if (!stmt.isa<Store>()) return;
    Store store_stmt = stmt.as<Store>();
    ir::Expr new_value = ir::ir_utils::IRCopy(store_stmt->value());
    ReduceExternCallMutator()(&new_value);
    store_stmt->set_value(new_value);
  };

  ir::stmt::Mutate(body, VisitFn, [](auto) {});
}

}  // namespace

LogicalResult RealizeCompositeReducePass::Run(ir::LoweredFunc func) {
  BlockRef body = func->body_block;

  // Step 1. Create a staging buffer for composite reduction result if it is
  //   directly written to the function's argument. This is because the
  //   result and the argument have different data types, and we need a staging
  //   buffer to do casting properly.
  // Note: theoretically, we don't need this mutator if all reduction results
  //   are explicitly written back to global memory by yield_stores. However,
  //   current CINN frontend cannot guarantee this, so we need to do staging by
  //   ourself if the expected yield_store is missing.
  StageReduceResultMutator mutator(func);
  mutator(body);

  auto composite_type = DetermineReduceType(body);
  if (composite_type.type == ReduceType::kNone) {
    return LogicalResult::success();
  }

  // Step 2. Collect buffers that are used for reduce computation.
  std::set<ir::Buffer> buffers = CollectReduceBuffers<Store>(body);

  // Step 3. Change the data type of buffers to the corresponding type.
  SetBufferType(func, buffers, composite_type);

  // Step 4. Replace the `cinn_reduce_variance` calls to `operator+` in order to
  //   reuse the cross-thread/block reduction templates.
  ReplaceReduceExternCall(body);

  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateRealizeCompositeReducePass() {
  return std::make_unique<RealizeCompositeReducePass>();
}

}  // namespace optim
}  // namespace cinn
