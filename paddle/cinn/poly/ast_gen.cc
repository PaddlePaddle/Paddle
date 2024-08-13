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

#include "paddle/cinn/poly/ast_gen.h"

#include <llvm/Support/FormatVariadic.h>

#include <utility>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/poly/domain_add_unit_loop_mutator.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace poly {

struct AstGen::Impl {
  Impl(const isl::set& context, const poly::ScheduleGroup& schedule_group)
      : context_(context), schedule_group_(schedule_group) {}
  //! Set the ISL ast_gen configs.
  void InitIslAstConfig();

  //! Return a domain composed of all the elements.
  isl::union_set domain() const;

  //! Return a map composed of all the transforms.
  isl::union_map transform();

  isl::ctx ctx() const;

  /**
   * Help to collect the map from the axis(and the pos) in statement to the
   * transformed indice. e.g. If s[i,j] will be generated to something like
   * s[a+2, b] in the final AST, this will return
   * - a map { i->a+2, j->b, 0->a+2, 1->b }.
   */
  static std::map<std::string, isl::ast_expr> ExtractIslTransformedIndiceMap(
      const isl::set& iterator_domain, isl_ast_build* build);

  //! Get the polyhedral stages.
  const std::vector<Shared<Stage>>& stages() const { return stages_; }

 private:
  isl::set context_;
  std::vector<Shared<Stage>> stages_;
  const poly::ScheduleGroup& schedule_group_;
  std::vector<std::string> iterator_names_;
  //! tuple name -> { axis -> isl_ast }
  std::map<std::string, std::map<std::string, isl::ast_expr>>
      transformed_indice_map_;
  isl::union_map build_options_;

  friend class AstGen;
};

isl::union_set AstGen::domain() const { return impl_->domain(); }

isl::union_set AstGen::Impl::domain() const {
  PADDLE_ENFORCE_NE(stages_.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Stages vector is empty in AstGen::Impl::domain()."));
  auto sets = utils::Map<std::vector<Shared<Stage>>, isl::set>(
      stages_, [](const Shared<Stage>& e) { return e->domain(); });
  return isl_sets_to_union_set(sets);
}

isl::ctx AstGen::ctx() const { return impl_->ctx(); }

isl::ctx AstGen::Impl::ctx() const {
  PADDLE_ENFORCE_NE(stages_.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Stages vector is empty in AstGen::Impl::ctx()."));
  return stages_.front()->domain().ctx();
}

isl::set TransIdentityExtentToContextId(isl::set set) {
  std::vector<std::tuple<int, int>> iden_dim_offsets;
  for (int i = 0; i < isl_set_dim(set.get(), isl_dim_set); i++) {
    if (isl_set_axis_has_noparam_constant_bound(set.get(), i)) {
      auto range = isl_set_get_axis_range(set.get(), i);
      auto& minv = std::get<0>(range);
      auto& maxv = std::get<1>(range);

      int min_iv = minv.get_num_si();
      int max_iv = maxv.get_num_si();
      if (max_iv == min_iv) {
        iden_dim_offsets.emplace_back(i, max_iv);
      }
    }
  }

  isl::set res_set = set;
  for (auto offset_val : iden_dim_offsets) {
    auto& offset = std::get<0>(offset_val);
    auto& val = std::get<1>(offset_val);
    res_set = isl::manage(isl_set_drop_constraints_involving_dims(
        res_set.copy(), isl_dim_set, offset, 1));

    std::string const_param_name =
        llvm::formatv("{0}{1}", kIslParamConstPrefix, val);

    std::string cond_str =
        llvm::formatv("{0} <= {1} <= {2}",
                      val,
                      isl_set_get_dim_name(res_set.get(), isl_dim_set, offset),
                      const_param_name);
    std::string param_cond_str =
        llvm::formatv("{0} <= {1} < {2}", val, const_param_name, val + 2);

    std::string set_repr =
        llvm::formatv("[{0}] -> { {1}[{2}]: {3} and {4} }",
                      const_param_name,
                      isl_set_get_tuple_name(res_set.get()),
                      utils::Join(isl_get_dim_names(res_set.get()), ","),
                      cond_str,
                      param_cond_str);

    VLOG(4) << "repr: " << set_repr;

    isl::set new_set(res_set.ctx(), set_repr);

    res_set = res_set.intersect(new_set);
  }
  return res_set;
}

isl::union_set TransIdentityExtentToContextId(isl::union_set set) {
  auto* set_list = isl_union_set_get_set_list(set.release());
  llvm::SmallVector<isl::set, 4> sets;
  for (int i = 0; i < isl_set_list_n_set(set_list); i++) {
    auto set = isl::manage(isl_set_list_get_set(set_list, i));
    set = TransIdentityExtentToContextId(set);
    sets.push_back(set);
  }
  isl_set_list_free(set_list);

  return isl_union_set_from_sets(sets);
}

isl::ast_node AstGen::Build() {
  // Collect schedule from scheduler.
  auto schedule_map = CollectScheduleMapFromGroup(impl_->schedule_group_);
  std::vector<isl::map> maps;
  for (auto& stage : impl_->stages_) {
    auto it = schedule_map.find(stage->id());
    PADDLE_ENFORCE_EQ(it != std::end(schedule_map),
                      true,
                      ::common::errors::InvalidArgument(
                          "Stage %s not found in the map.", stage->id()));
    maps.push_back(it->second);
  }
  auto schedule = isl_maps_to_union_map(maps);

  // Build it.
  auto ast_build = isl::ast_build::from_context(impl_->context_);

  if (!impl_->build_options_.is_null())
    ast_build = isl::manage(isl_ast_build_set_options(
        ast_build.release(), impl_->build_options_.release()));

  // Set iterators names for readable code.
  auto iterator_names = impl_->iterator_names_.empty()
                            ? impl_->schedule_group_.dimension_names
                            : impl_->iterator_names_;

  iterator_names = SchedulerBase::WrapIteratorNames(iterator_names);
  isl::id_list ids =
      isl::manage(isl_id_list_alloc(ctx().get(), iterator_names.size()));
  for (int i = 0; i < iterator_names.size(); i++) {
    ids = isl::manage(isl_id_list_add(
        ids.release(),
        isl_id_alloc(ctx().get(), iterator_names[i].c_str(), nullptr)));
  }
  ast_build = isl::manage(
      isl_ast_build_set_iterators(ast_build.release(), ids.release()));

  // collect iterator map
  auto get_domain_by_name = [this](const std::string& name) -> isl::set {
    auto ele_it = std::find_if(
        impl_->stages_.begin(),
        impl_->stages_.end(),
        [&name](const Shared<Stage>& ele) { return ele->id() == name; });
    PADDLE_ENFORCE_EQ(
        ele_it != std::end(impl_->stages_),
        true,
        ::common::errors::InvalidArgument(
            "Stage with name %s not found in the stages vector.", name));
    return (*ele_it)->domain();
  };

  auto collect = [&](isl::ast_node node,
                     isl::ast_build build) -> isl::ast_node {
    auto tuple_name = detail::GetTupleName(node.get());
    auto indice_map = impl_->ExtractIslTransformedIndiceMap(
        get_domain_by_name(tuple_name), build.get());
    impl_->transformed_indice_map_[tuple_name] = indice_map;
    return node;
  };

  ast_build = ast_build.set_at_each_domain(collect);

  isl::union_map transformed_schedule =
      impl_->transform().apply_range(schedule);
  VLOG(4) << "transformed_schedule: " << transformed_schedule;
  isl::union_map schedule_domain =
      transformed_schedule.intersect_domain(impl_->domain());
  VLOG(4) << "domain: " << impl_->domain();
  VLOG(4) << "transform schedule " << impl_->stages()[0]->transform();
  VLOG(4) << "schedule: " << schedule;
  VLOG(4) << "schedule_domain: " << schedule_domain;
  isl::ast_node ast = ast_build.node_from_schedule_map(schedule_domain);
  VLOG(2) << "AST:\n" << isl_ast_node_to_C_str(ast.get());
  return ast;
}

AstGen& AstGen::SetIteratorNames(const std::vector<std::string>& names) {
  impl_->iterator_names_ = names;
  return *this;
}

isl::ast_expr CreateIslAstIndexExpression(isl_ast_build* build,
                                          const isl::map& access);

std::map<std::string, isl::ast_expr>
AstGen::Impl::ExtractIslTransformedIndiceMap(const isl::set& iterator_domain,
                                             isl_ast_build* build) {
  std::map<std::string, isl::ast_expr> iterator_map;
  isl::map identity = isl::manage(isl_set_identity(iterator_domain.copy()));
  isl::map schedule = identity;

  identity = identity.apply_domain(schedule);
  isl::ast_expr idx_expr = CreateIslAstIndexExpression(build, identity);
  isl::space domain_space = iterator_domain.space();

  for (int i = 1; i < isl_ast_expr_get_op_n_arg(idx_expr.get()); i++) {
    if (isl_space_has_dim_name(domain_space.get(), isl_dim_set, i - 1)) {
      std::string original_idx_name =
          isl_space_get_dim_name(domain_space.get(), isl_dim_set, i - 1);
      isl::ast_expr transformed_index =
          isl::manage(isl_ast_expr_get_op_arg(idx_expr.get(), i));
      VLOG(4) << "axis-" << i - 1 << " named " << original_idx_name << ", is "
              << isl_ast_expr_to_C_str(transformed_index.get());
      iterator_map.emplace(original_idx_name, transformed_index);
      iterator_map.emplace(std::to_string(i - 1), transformed_index);
    }
  }

  return iterator_map;
}

const std::map<std::string, isl::ast_expr>& AstGen::axis2ast(
    const std::string& tuple_name) const {
  auto it = impl_->transformed_indice_map_.find(tuple_name);
  PADDLE_ENFORCE_EQ(it != impl_->transformed_indice_map_.end(),
                    true,
                    ::common::errors::InvalidArgument(
                        "No id named %s, please check.", tuple_name));
  return it->second;
}

const std::map<std::string, Expr> AstGen::axis2expr(
    const std::string& tuple_name) const {
  const auto& axis_to_ast = axis2ast(tuple_name);
  std::map<std::string, Expr> res;
  for (auto item : axis_to_ast) {
    Expr expr;
    IslAstExprToCinnExpr(item.second, &expr);
    res[item.first] = expr;
  }
  return res;
}

isl::ast_expr CreateIslAstIndexExpression(isl_ast_build* build,
                                          const isl::map& access) {
  PADDLE_ENFORCE_NOT_NULL(
      build,
      ::common::errors::InvalidArgument(
          "The isl_ast_build pointer is null in CreateIslAstIndexExpression."));
  isl::map schedule =
      isl::manage(isl_map_from_union_map(isl_ast_build_get_schedule(build)));

  // get identity access from schedule.
  auto statement = isl_map_get_statement_repr(schedule.get(), isl_dim_in);
  auto statement_set = isl::manage(isl_set_read_from_str(
      isl_map_get_ctx(schedule.get()),
      utils::StringFormat("{ %s : }", statement.c_str()).c_str()));
  auto identity_access = isl::manage(isl_set_identity(statement_set.release()));
  isl::map map = isl::manage(isl_map_reverse(schedule.copy()));

  isl::pw_multi_aff iterator_map =
      isl::manage(isl_pw_multi_aff_from_map(map.copy()));
  isl::pw_multi_aff index_aff =
      isl::manage(isl_pw_multi_aff_from_map(identity_access.copy()));

  isl::space model2 = iterator_map.space();
  index_aff = isl::manage(
      isl_pw_multi_aff_align_params(index_aff.copy(), model2.copy()));
  isl::space model = index_aff.space();
  iterator_map = isl::manage(
      isl_pw_multi_aff_align_params(iterator_map.copy(), model.copy()));
  iterator_map = isl::manage(isl_pw_multi_aff_pullback_pw_multi_aff(
      index_aff.copy(), iterator_map.copy()));
  isl::ast_expr index_expr = isl::manage(
      isl_ast_build_access_from_pw_multi_aff(build, iterator_map.copy()));

  return index_expr;
}

isl::union_map AstGen::Impl::transform() {
  std::vector<isl::map> transforms;
  for (auto& stage : stages()) {
    transforms.push_back(stage->transform());
  }
  return isl_maps_to_union_map(transforms);
}

namespace detail {

std::string GetTupleName(isl_ast_node* node) {
  auto expr = isl::manage(isl_ast_node_user_get_expr(node));
  auto arg = isl::manage(isl_ast_expr_get_op_arg(expr.get(), 0));
  auto name = isl_id_get_name(isl_ast_expr_get_id(arg.get()));
  return name;
}

}  // namespace detail

//! Eat an isl block node.
void EatBlock(const isl::ast_node& node, ir::Expr* expr);
//! Eat an isl user node.
void EatUser(const isl::ast_node& node, ir::Expr* expr);
//! Eat an isl for node.
void EatFor(const isl::ast_node& node, ir::Expr* expr);
//! Eat an isl `if` node.
void EatIf(const isl::ast_node& node, ir::Expr* expr);
//! Eat an isl mark node.
void EatMark(const isl::ast_node& node, ir::Expr* expr);

void IslAstNodeToCinnExpr(const isl::ast_node& node, ir::Expr* expr) {
  PADDLE_ENFORCE_EQ(!node.is_null(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The isl::ast_node is null in IslAstNodeToCinnExpr."));
  PADDLE_ENFORCE_NOT_NULL(
      expr,
      ::common::errors::InvalidArgument(
          "The ir::Expr pointer is null in IslAstNodeToCinnExpr."));

  switch (isl_ast_node_get_type(node.get())) {
    case isl_ast_node_block: {
      VLOG(6) << "get isl block node";
      EatBlock(node, expr);
    } break;
    case isl_ast_node_for: {
      VLOG(6) << "get isl for node";
      EatFor(node, expr);
    } break;
    case isl_ast_node_if: {
      VLOG(6) << "get isl if node";
      EatIf(node, expr);
    } break;
    case isl_ast_node_user: {
      VLOG(6) << "get isl user node";
      EatUser(node, expr);
    } break;
    case isl_ast_node_mark: {
      VLOG(6) << "get isl mark";
      // EatMark(node, expr);
    } break;
    default:
      std::stringstream ss;
      ss << "Unexpected ISL node type " << isl_ast_node_get_type(node.get());
      PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
      break;
  }
}

// Eat an isl block node.
void EatBlock(const isl::ast_node& node, ir::Expr* expr) {
  VLOG(2) << "get isl ast body node";
  PADDLE_ENFORCE_EQ(!node.is_null(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The isl::ast_node is null in EatBlock."));
  PADDLE_ENFORCE_NOT_NULL(expr,
                          ::common::errors::InvalidArgument(
                              "The ir::Expr pointer is null in EatBlock."));
  PADDLE_ENFORCE_EQ(isl_ast_node_get_type(node.get()),
                    isl_ast_node_block,
                    ::common::errors::InvalidArgument(
                        "The node type should be isl_ast_node_block"));
  isl::ast_node_list list =
      isl::manage(isl_ast_node_block_get_children(node.get()));
  std::vector<ir::Expr> exprs;
  for (int i = 0; i < isl_ast_node_list_n_ast_node(list.get()); i++) {
    isl::ast_node child =
        isl::manage(isl_ast_node_list_get_ast_node(list.get(), i));
    // visit child
    ir::Expr child_expr;
    IslAstNodeToCinnExpr(child, &child_expr);
    exprs.push_back(child_expr);
  }
  *expr = ir::Block::Make(std::move(exprs));
}
// Eat an isl user node.
void EatUser(const isl::ast_node& node, ir::Expr* expr) {
  PADDLE_ENFORCE_EQ(isl_ast_node_get_type(node.get()),
                    isl_ast_node_user,
                    ::common::errors::InvalidArgument(
                        "The node type should be isl_ast_node_user"));
  isl::ast_expr isl_expr = isl::manage(isl_ast_node_user_get_expr(node.get()));
  IslAstExprToCinnExpr(isl_expr, expr);
}
// Eat an isl `for` node.
void EatFor(const isl::ast_node& node, ir::Expr* expr) {
  PADDLE_ENFORCE_EQ(isl_ast_node_get_type(node.get()),
                    isl_ast_node_for,
                    ::common::errors::InvalidArgument(
                        "The node type should be isl_ast_node_for"));

  // iter name
  isl::ast_expr iter = isl::manage(isl_ast_node_for_get_iterator(node.get()));
  isl::id iter_id = isl::manage(isl_ast_expr_get_id(iter.get()));
  std::string iter_name = iter_id.name();

  // get condition
  isl::ast_expr condition = isl::manage(isl_ast_node_for_get_cond(node.get()));
  isl::ast_expr incrementor = isl::manage(isl_ast_node_for_get_inc(node.get()));
  isl::ast_expr initializer =
      isl::manage(isl_ast_node_for_get_init(node.get()));
  isl::ast_node body = isl::manage(isl_ast_node_for_get_body(node.get()));

  ir::Expr ir_body;
  IslAstNodeToCinnExpr(body, &ir_body);
  ir_body = ir::Block::Make({ir_body});

  ir::Expr ir_initializer;
  IslAstExprToCinnExpr(initializer, &ir_initializer);

  ir::Expr ir_condition;
  IslAstExprToCinnExpr(condition, &ir_condition);
  ir::Expr tmp;

  isl::ast_expr arg = isl::manage(isl_ast_expr_get_op_arg(condition.get(), 1));
  IslAstExprToCinnExpr(arg, &tmp);

  ir::Expr ir_inc;
  IslAstExprToCinnExpr(incrementor, &ir_inc);

  ir::Var ir_iter(iter_name);

  *expr = ir::PolyFor::Make(ir::Var(iter_name),
                            ir_initializer,
                            ir_condition,
                            ir_inc,
                            ir::ForType::Serial,
                            ir::DeviceAPI ::Host,
                            ir_body);
}

void EatIf(const isl::ast_node& node, ir::Expr* expr) {
  PADDLE_ENFORCE_EQ(isl_ast_node_get_type(node.get()),
                    isl_ast_node_if,
                    ::common::errors::InvalidArgument(
                        "The node type should be isl_ast_node_if."));
  isl::ast_node then_body = isl::manage(isl_ast_node_if_get_then(node.get()));
  isl::ast_expr condition = isl::manage(isl_ast_node_if_get_cond(node.get()));

  ir::Expr ir_then_body;
  IslAstNodeToCinnExpr(then_body, &ir_then_body);

  ir::Expr ir_else_body;
  if (isl_bool_true == isl_ast_node_if_has_else(node.get())) {
    isl::ast_node else_body = isl::manage(isl_ast_node_if_get_else(node.get()));
    IslAstNodeToCinnExpr(else_body, &ir_else_body);
  }

  ir::Expr ir_condition;
  IslAstExprToCinnExpr(condition, &ir_condition);

  if (ir_else_body.defined()) {
    *expr = ir::IfThenElse::Make(ir_condition, ir_then_body, ir_else_body);
  } else {
    *expr = ir::IfThenElse::Make(ir_condition, ir_then_body, ir::Expr());
  }
}

void IslAstExprToCinnExpr(const isl::ast_expr& node, ir::Expr* expr) {
  switch (isl_ast_expr_get_type(node.get())) {
    case isl_ast_expr_int: {
      isl::val val = isl::manage(isl_ast_expr_get_val(node.get()));
      *expr = ir::Expr(static_cast<int>(isl_val_get_num_si(val.get())));
    } break;
    case isl_ast_expr_id: {
      isl::id id = isl::manage(isl_ast_expr_get_id(node.get()));
      *expr = ir::Var(id.name());
    } break;
    case isl_ast_expr_op: {
      std::vector<ir::Expr> ops;
      const int n_args = isl_ast_expr_get_op_n_arg(node.get());

      for (int i = 0; i < n_args; i++) {
        ir::Expr op;
        isl::ast_expr expr0 =
            isl::manage(isl_ast_expr_get_op_arg(node.get(), i));
        IslAstExprToCinnExpr(expr0, &op);
        ops.push_back(op);
      }

      auto set_ops_ptype = [&](ir::Type type) {
        for (auto& op : ops) {
          op->set_type(type);
        }
      };

      // set ops as int32 by default.
      set_ops_ptype(Int(32));

      isl_ast_op_type op_type = isl_ast_expr_get_op_type(node.get());
      switch (op_type) {
        case isl_ast_op_and: {
          set_ops_ptype(Bool());
          *expr = ir::And::Make(ops[0], ops[1]);
        } break;
        case isl_ast_op_or:
          *expr = ir::Or::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_min:
          *expr = ir::Min::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_max:
          *expr = ir::Max::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_minus:
          *expr = ir::Minus::Make(ops[0]);
          break;
        case isl_ast_op_add:
          *expr = ir::Add::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_sub:
          *expr = ir::Sub::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_mul:
          *expr = ir::Mul::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_div:
          *expr = ir::Div::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_le:
          *expr = ir::LE::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_lt:
          *expr = ir::LT::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_ge:
          *expr = ir::GE::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_gt:
          *expr = ir::GT::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_eq:
          *expr = ir::EQ::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_pdiv_q:
          *expr = ir::Div::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_zdiv_r:
        case isl_ast_op_pdiv_r:
          *expr = ir::Mod::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_call: {
          ir::Expr caller_expr = ops.front();
          // TODO(Superjomn) make it an string
          PADDLE_ENFORCE_EQ(
              caller_expr.node_type() == ir::IrNodeTy::_Var_,
              true,
              ::common::errors::InvalidArgument(
                  "Expected caller_expr to be of type _Var_, but got %s.",
                  caller_expr.node_type()));
          std::string caller = caller_expr.As<ir::_Var_>()->name;
          ops.erase(ops.begin());
          // NOTE the type here is not important.
          *expr = ir::Call::Make(Float(32),
                                 caller,
                                 ops,
                                 {},
                                 ir::CallType::ISL,
                                 ir::FunctionRef(),
                                 0);
        } break;
        case isl_ast_op_fdiv_q:
          *expr = ir::Div::Make(ops[0], ops[1]);
          break;
        case isl_ast_op_select:
          PADDLE_ENFORCE_EQ(ops.size(),
                            3UL,
                            ::common::errors::InvalidArgument(
                                "In ir::Select, the ops size should be 3"));
          ops[0]->set_type(Bool());
          *expr = ir::Select::Make(ops[0], ops[1], ops[2]);
          break;
        default:
          std::stringstream ss;
          ss << "unsupported op " << op_type;
          PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
      }
    } break;
    default:
      break;
  }
}

void AddUnitLoopOfDomain(const isl::ast_node& node,
                         const isl::set& domain,
                         ir::Expr* expr) {
  std::vector<std::string> dim_names = isl_get_dim_names(domain);
  std::vector<std::tuple<int, int, int>> dim_min_max;
  for (int i = 0; i < dim_names.size(); ++i) {
    auto minv_maxv = isl_set_get_axis_range(domain.get(), i);
    int min_iv = std::get<0>(minv_maxv).get_num_si();
    int max_iv = std::get<1>(minv_maxv).get_num_si();
    dim_min_max.emplace_back(i, min_iv, max_iv);
  }

  DomainAddUnitLoopMutator mutator(dim_names, dim_min_max);
  mutator(expr);
}

void IslAstNodeToCinnExpr(const isl::ast_node& node,
                          const isl::union_set& domain,
                          ir::Expr* expr) {
  IslAstNodeToCinnExpr(node, expr);

  isl_set_list* set_list = isl_union_set_get_set_list(domain.get());
  VLOG(6) << "After convert to CinnExpr, n = " << isl_set_list_n_set(set_list);
  for (int i = 0; i < isl_set_list_n_set(set_list); i++) {
    isl::set s = isl::manage(isl_set_list_get_set(set_list, i));
    AddUnitLoopOfDomain(node, s, expr);
  }
}

void AstGen::Impl::InitIslAstConfig() {
  isl_options_set_ast_build_detect_min_max(ctx().get(), 1);
  isl_options_set_ast_build_exploit_nested_bounds(ctx().get(), 1);
  isl_options_set_ast_build_scale_strides(ctx().get(), 1);
  isl_options_set_ast_build_allow_else(ctx().get(), 1);
}

AstGen::AstGen(const isl::set& context,
               const std::vector<Stage*>& stages,
               const poly::ScheduleGroup& group)
    : impl_(new Impl(context, group)) {
  for (auto* x : stages) impl_->stages_.emplace_back(x);
  impl_->InitIslAstConfig();
}
void AstGen::SetBuildOptions(const isl::union_map& options) {
  impl_->build_options_ = options;
}
bool AstGen::ContainsStatement(const std::string& name) const {
  return impl_->transformed_indice_map_.count(name);
}

AstGen::~AstGen() {}

}  // namespace poly
}  // namespace cinn
