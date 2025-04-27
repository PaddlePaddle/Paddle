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

  bool operator==(const CompositeTypes& other) const {
    if (this->type != other.type || other.size() != this->size()) return false;
    for (size_t i = 0; i < other.size(); i++) {
      if (this->at(i) != other.at(i)) return false;
    }
    return true;
  }

  void Print() const {
    VLOG(4) << "[CompositeTypes]: " << static_cast<int>(this->type);
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
                  const CompositeTypes& comp_type,
                  std::string prefix = "") {
  // prefix: if target is x86, we can not call constructor for POD struct
  // the intrinsic function for creating struct is usually "create_" + typename
  ir::Expr init_value = store_stmt->value();
  auto call_op = init_value.As<ir::Call>();
  // if the type is already a call
  if (call_op != nullptr) {
    call_op->set_type(new_type);
    if (call_op->name.find("argidx_") == 0 ||
        call_op->name.find("welford_") == 0) {
      call_op->name = prefix + call_op->name;
    }
    return;
  }
  if (comp_type.type == ReduceType::kVariance) {
    store_stmt->set_value(ir::Call::Make(new_type,
                                         prefix + new_type.customized_type(),
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
                                         prefix + new_type.customized_type(),
                                         {init_value, index_init},
                                         {},
                                         ir::CallType::Intrinsic));
  } else {
    PADDLE_THROW(::common::errors::Unimplemented(
        "reduce_type '%d' not allowed.", static_cast<int>(comp_type.type)));
  }
}

/**
 * This function resolves undefined argidx type, for example:
 * \code
 * spatial inner loop (argidx type defined)
 *    tensor_0[...] = cinn_argmax(tensor_1[...], argidx_f32_i64(tensor_2[...],
 * index))
 *
 * follow up cross thread reduce (argidx type undefined)
 *    tensor_3[...] = cinn_argmax(tensor_4[...], tensor_5[...])
 * \endcode
 * In the above undefined case, we can not extract value type, since both
 * tensors (4 and 5) in the arguments will be of index type, which lefts the
 * argidx type undefined. So this function basically checks whether tensor_5 is
 * in the typed_buffers map. Since cross thread reduction usually follows
 * spatial inner loop reduction, so normally, tensor_0 and tensor_5 will
 * normally be the same in one reduce block. and since tensor_0's type is
 * defined, we can use it to resolve tensor_5 (and thus, the undefined
 * tensor_3)'s type.
 */
std::map<ir::Buffer, CompositeTypes> ResolveUndefinedArgIdxType(
    std::map<ir::Buffer, CompositeTypes>&& typed_buffers,
    std::vector<Store>&& stores) {
  for (const auto& store_stmt : stores) {
    if (auto call_stmt = store_stmt->value().As<ir::Call>()) {
      if (call_stmt->name != hlir::pe::kArgmaxFuncName &&
          call_stmt->name != hlir::pe::kArgminFuncName)
        continue;
      auto load_stmt = call_stmt->read_args[1].As<ir::Load>();
      PADDLE_ENFORCE_NOT_NULL(load_stmt,
                              ::common::errors::PreconditionNotMet(
                                  "Non-spatial inner loop arg reduce func call "
                                  "second argument must be load."));
      auto it = typed_buffers.find(load_stmt->tensor.as_tensor()->buffer);
      PADDLE_ENFORCE_NE(it,
                        typed_buffers.end(),
                        ::common::errors::PreconditionNotMet(
                            "Referenced buffer '%s' should be defined.",
                            load_stmt->tensor.as_tensor()->buffer->name));
      auto composite_type = it->second;
      typed_buffers.emplace(store_stmt->tensor().as_tensor()->buffer,
                            composite_type);
    }
  }
  return typed_buffers;
}

std::map<ir::Buffer, CompositeTypes> CollectTypedReduceBuffers(
    const BlockRef& body, std::vector<Store>* arg_stores) {
  std::map<ir::Buffer, CompositeTypes> typed_buffers;
  const auto VisitFn = [&](const StmtRef& stmt) {
    if (!stmt.isa<Store>()) return;
    Store store_stmt = stmt.as<Store>();
    if (GetReduceType(store_stmt->value()) != ReduceType::kNone) {
      auto it = typed_buffers.find(store_stmt->tensor().as_tensor()->buffer);
      if (it == typed_buffers.end()) {
        auto composite_type = GetArgReduceUnderlyingType(store_stmt->value());
        if (composite_type.type == ReduceType::kNone) {
          arg_stores->emplace_back(store_stmt);
        } else {
          // defined composite type can be immediately stored
          typed_buffers.emplace(store_stmt->tensor().as_tensor()->buffer,
                                composite_type);
        }
      } else {
        // check whether we will have conflicted store types
        PADDLE_ENFORCE_EQ(
            it->second == GetArgReduceUnderlyingType(store_stmt->value()),
            true,
            ::common::errors::PreconditionNotMet(
                "Composite type conflict detected in the buffer map."));
      }
    }
  };

  ir::stmt::Visit(body, VisitFn, [](auto) {});
  return typed_buffers;
}

void ReplaceOutputBufferX86(
    const BlockRef& body,
    const std::set<ir::Buffer>& out_buffer_map,
    const std::map<ir::Buffer, CompositeTypes>& typed_buffers) {
  // re-route the reduce_init buffer to the local staging buffer
  // and set the type for the buffers correctly
  struct BufferRelationRecorder {
    Store reduce_init;
    Store write_back;
  };
  std::map<ir::Buffer, BufferRelationRecorder> buffer_relations;
  for (auto buffer : out_buffer_map) {
    buffer_relations.emplace(buffer, BufferRelationRecorder());
  }
  const auto VisitFn = [&](const StmtRef& stmt) {
    if (!stmt.isa<Store>()) return;
    Store store_stmt = stmt.as<Store>();

    auto* tensor = store_stmt->tensor().as_tensor();
    auto& buffer = tensor->buffer;
    auto buffer_it = buffer_relations.find(buffer);
    // check whether the buffer is related to output args
    if (buffer_it == buffer_relations.end()) return;
    if (ir::IsReduceInitTensorName(tensor->name)) {
      buffer_it->second.reduce_init = store_stmt;
    } else {
      buffer_it->second.write_back = store_stmt;
    }
  };

  ir::stmt::Visit(body, VisitFn, [](auto) {});

  for (auto& [_, buffer_rel] : buffer_relations) {
    // both should be defined
    if (!buffer_rel.reduce_init.defined() || !buffer_rel.write_back.defined())
      continue;
    auto wb_value = buffer_rel.write_back->value();
    if (auto load_node = wb_value.As<ir::Load>()) {
      auto wb_load_buffer = load_node->tensor.as_tensor()->buffer;
      auto wb_load_it = typed_buffers.find(wb_load_buffer);
      PADDLE_ENFORCE_NE(wb_load_it,
                        typed_buffers.end(),
                        ::common::errors::Fatal(
                            "Buffer '%s' should be defined in typed_buffers.",
                            wb_load_buffer->name));
      // set the buffer of the reduce_init to write back buffer
      ir::Expr new_tensor =
          ir::ir_utils::IRCopy(buffer_rel.reduce_init->tensor());
      new_tensor.as_tensor()->buffer = wb_load_buffer;
      buffer_rel.reduce_init->set_tensor(new_tensor);
    }
  }
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
  explicit LoadTypeMutator(
      const std::map<ir::Buffer, std::pair<ir::Type, CompositeTypes>>&
          buffer2type)
      : buffer2type_(buffer2type) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Load* op, ir::Expr* expr) override {
    ir::IRMutator<>::Visit(op, expr);
    auto* node = expr->As<ir::Load>();
    auto& buffer = node->tensor.as_tensor()->buffer;
    auto it = buffer2type_.find(buffer);
    if (it != buffer2type_.end()) {
      const auto& [buffer_type, composite_type] = it->second;
      ir::Type new_type = GetCompositeReduceType(buffer_type, composite_type);
      node->tensor.as_tensor()->set_type(new_type);
      buffer->dtype = new_type;
      *expr = ir::Cast::Make(buffer_type, *expr);
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

  const std::map<ir::Buffer, std::pair<ir::Type, CompositeTypes>>& buffer2type_;
};

void SetBufferType(ir::LoweredFunc func,
                   const std::map<ir::Buffer, CompositeTypes>& typed_buffers,
                   bool is_x86_arch) {
  // Make a map from the buffers to their element and composite reduce types,
  // otherwise it's hard to know a buffer's original type. The original type
  // must be known to perform casting (back) in LoadTypeMutator::Visit()
  std::map<ir::Buffer, std::pair<ir::Type, CompositeTypes>> buffer2type;
  for (auto& [buffer, reduce_type] : typed_buffers) {
    buffer2type.emplace(buffer, std::make_pair(buffer->dtype, reduce_type));
  }

  // Set function's temp_bufs type
  for (auto& buffer : func->temp_bufs) {
    auto it = buffer2type.find(buffer);
    if (it != buffer2type.end()) {
      const auto& [buffer_type, composite_type] = it->second;
      buffer->dtype = GetCompositeReduceType(buffer_type, composite_type);
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
      const auto& [buffer_type, composite_type] = it->second;
      ir::Type new_type = GetCompositeReduceType(buffer_type, composite_type);
      new_tensor.as_tensor()->set_type(new_type);
      new_tensor.as_tensor()->buffer->dtype = new_type;
      store_stmt->set_tensor(new_tensor);
      stmt->set_type(new_type);
      if (ir::IsReduceInitTensorName(new_tensor.as_tensor()->name)) {
        std::string call_prefix = is_x86_arch ? "create_" : "";
        SetInitValue(store_stmt, new_type, composite_type, call_prefix);
      }
    }

    // Set load buffer type
    ir::Expr new_value = ir::ir_utils::IRCopy(store_stmt->value());
    LoadTypeMutator load_type_mutator(buffer2type);
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

struct ReduceExternCallMutatorX86 : public ir::IRMutator<> {
  // unlike non x86 counterpart, we do not replace the call
  // by a arithmetic IR node, but instead call x86-exclusive funcs
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Call* op, ir::Expr* expr) override {
    ir::IRMutator<>::Visit(op, expr);
    auto reduce_type_ = GetReduceType(*expr);
    if (reduce_type_ == ReduceType::kNone) return;
    ir::Expr lhs = op->read_args[0];
    ir::Expr rhs = op->read_args[1];
    std::string lhs_type = lhs.type().to_string();
    if (lhs.type() != rhs.type()) {
      if (auto call_op = rhs.As<ir::Call>()) {
        // for argidx type, avoid redundant type casting, but this is ugly
        if (call_op->name.find("argidx") == 0) {
          call_op->name = "create_" + call_op->name;
          rhs->set_type(lhs.type());
        }
      } else {
        // welford pod type call create function on x86
        ir::Expr m2_init(0.f), weight_init(1.f);
        if (lhs_type == "welford_fp64") {
          m2_init->set_type(common::F64());
          weight_init->set_type(common::F64());
        }
        rhs = ir::Call::Make(lhs.type(),
                             "create_" + lhs_type,
                             {rhs, m2_init, weight_init},
                             {},
                             ir::CallType::Intrinsic);
      }
    }
    std::string call_prefix = "";
    switch (reduce_type_) {
      case ReduceType::kVariance:
        call_prefix = "sum_";
        break;
      case ReduceType::kArgmax:
        call_prefix = "max_";
        break;
      case ReduceType::kArgmin:
        call_prefix = "min_";
        break;
      default:
        break;
    }
    *expr = ir::Call::Make(lhs.type(),
                           call_prefix + lhs_type,
                           {lhs, rhs},
                           {},
                           ir::CallType::Intrinsic);
  }
};

void ReplaceReduceExternCall(const BlockRef& body, bool is_x86_arch = false) {
  const auto VisitFn = [&](StmtRef stmt) {
    if (!stmt.isa<Store>()) return;
    Store store_stmt = stmt.as<Store>();
    ir::Expr new_value = ir::ir_utils::IRCopy(store_stmt->value());
    if (is_x86_arch) {
      ReduceExternCallMutatorX86()(&new_value);
    } else {
      ReduceExternCallMutator()(&new_value);
    }
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

  // Step 2. Collect buffers that are used for reduce computation.
  std::vector<Store> arg_stores;
  auto typed_buffers = CollectTypedReduceBuffers(body, &arg_stores);
  if (typed_buffers.empty()) {
    // not a composite reduce func
    return LogicalResult::success();
  }
  typed_buffers = ResolveUndefinedArgIdxType(std::move(typed_buffers),
                                             std::move(arg_stores));

  bool is_x86_arch = false;
  target_.arch.Match(
      [&](std::variant<common::X86Arch>) {
        /**
         * trace the CPU buffer for reduce init. For x86 pass, schedule pass
         * will not be applied, therefore, the reduce_init buffer will be the
         * same as the output buffer, which leads to incorrect buffer type and
         * op type for codegen
         *
         * (1) we first extract the buffer for each output arg
         * (2) find all stores to the corresponding output buffer, this op is
         * prior to the output type cast, for x86 IR, reduce_init and the
         * writing back op uses the same buffer (output tensor buffer). (3)
         * create a mapping. if the buffer of a store (the value of the store)
         * is in the typed_buffer, we try finding the reduce_init related op,
         * and change the the buffer and op type of the reduce_init
         */
        is_x86_arch = true;
        std::set<ir::Buffer> output_buffers;
        for (auto& arg : func->args) {
          if (!arg.is_output()) continue;
          output_buffers.emplace(arg.buffer_arg());
        }
        ReplaceOutputBufferX86(body, output_buffers, typed_buffers);
      },
      [&](std::variant<common::NVGPUArch,
                       common::HygonDCUArchHIP,
                       common::HygonDCUArchSYCL,
                       common::ARMArch,
                       common::UnknownArch>) {});
  // Step 3. Change the data type of buffers to the corresponding type.
  SetBufferType(func, typed_buffers, is_x86_arch);

  // Step 4. Replace the `cinn_reduce_variance` and `cinn_argmax` calls
  // in order to reuse the cross-thread/block reduction templates.
  ReplaceReduceExternCall(body, is_x86_arch);

  return LogicalResult::success();
}

std::unique_ptr<FuncPass> CreateRealizeCompositeReducePass(Target target) {
  return std::make_unique<RealizeCompositeReducePass>(target);
}

}  // namespace optim
}  // namespace cinn
