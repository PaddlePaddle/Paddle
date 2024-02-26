// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/poly/domain_add_unit_loop_mutator.h"

#include <glog/logging.h>

#include <tuple>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace poly {

DomainAddUnitLoopMutator::DomainAddUnitLoopMutator(
    const std::vector<std::string>& dim_names,
    const std::vector<std::tuple<int, int, int>>& dim_min_max)
    : dim_names_(dim_names), dim_min_max_(dim_min_max) {}

void DomainAddUnitLoopMutator::operator()(ir::Expr* expr) {
  ir::IRMutator<>::Visit(expr, expr);

  // If the loop with length 1 is the most inner loop, Visit cannot find it
  // in deleted-length-1-loop expr. So we should check after visit
  MutateAfterVisit(expr);
}

void DomainAddUnitLoopMutator::Visit(const ir::For* op, Expr* expr) {
  VLOG(6) << "DomainAddUnitLoopMutator Visit For";
  ir::For* node = expr->As<ir::For>();
  bool add_unit_loop = false;
  if (parent_for_.size() < dim_names_.size()) {
    std::string check_name = dim_names_[parent_for_.size()];
    std::tuple<int, int, int> t = dim_min_max_[parent_for_.size()];
    if (!utils::StartsWith(node->loop_var->name, check_name) &&
        (std::get<2>(t) - std::get<1>(t) == 0)) {
      ir::Expr unit_loop = ir::For::Make(ir::Var(check_name),
                                         ir::Expr(0),
                                         ir::Expr(1),
                                         ir::ForType::Serial,
                                         node->device_api,
                                         ir::Block::Make({*expr}));
      if (parent_for_.empty()) {
        *expr = unit_loop;
        parent_for_.push_back(unit_loop.As<ir::For>());
        longest_loop_.push_back(unit_loop);
        add_unit_loop = true;
      } else if (parent_for_.back()->body.As<ir::For>() &&
                 parent_for_.back()->body == *expr) {
        parent_for_.back()->body = ir::Block::Make({unit_loop});
        parent_for_.push_back(unit_loop.As<ir::For>());
        longest_loop_.push_back(unit_loop);
        add_unit_loop = true;
      } else if (parent_for_.back()->body.As<ir::Block>()) {
        ir::Block* body = parent_for_.back()->body.As<ir::Block>();
        if (body->stmts.size() == 1 && body->stmts[0] == *expr) {
          parent_for_.back()->body = ir::Block::Make({unit_loop});
          parent_for_.push_back(unit_loop.As<ir::For>());
          longest_loop_.push_back(unit_loop);
          add_unit_loop = true;
        }
      }
    }
  }

  if (add_unit_loop) {
    ir::IRMutator<>::Visit(&(parent_for_.back()->body),
                           &(parent_for_.back()->body));
    parent_for_.pop_back();
  } else {
    parent_for_.push_back(node);
    longest_loop_.push_back(*expr);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    parent_for_.pop_back();
  }
}

void DomainAddUnitLoopMutator::Visit(const ir::PolyFor* op, Expr* expr) {
  VLOG(6) << "DomainAddUnitLoopMutator Visit PolyFor";
  ir::PolyFor* node = expr->As<ir::PolyFor>();
  bool add_unit_loop = false;
  if (parent_poly_for_.size() < dim_names_.size()) {
    std::string check_name = dim_names_[parent_poly_for_.size()];
    std::tuple<int, int, int> t = dim_min_max_[parent_poly_for_.size()];
    if (!utils::StartsWith(node->iterator->name, check_name) &&
        (std::get<2>(t) - std::get<1>(t) == 0)) {
      ir::Expr unit_loop =
          ir::PolyFor::Make(ir::Var(check_name),
                            ir::Expr(0),
                            ir::LE::Make(ir::Var(check_name), ir::Expr(0)),
                            ir::Expr(1),
                            ir::ForType::Serial,
                            node->device_api,
                            ir::Block::Make({*expr}));

      if (parent_poly_for_.empty()) {
        *expr = unit_loop;
        parent_poly_for_.push_back(unit_loop.As<ir::PolyFor>());
        longest_loop_.push_back(unit_loop);
        add_unit_loop = true;
      } else if (parent_poly_for_.back()->body.As<ir::PolyFor>() &&
                 parent_poly_for_.back()->body == *expr) {
        parent_poly_for_.back()->body = ir::Block::Make({unit_loop});
        parent_poly_for_.push_back(unit_loop.As<ir::PolyFor>());
        longest_loop_.push_back(unit_loop);
        add_unit_loop = true;
      } else if (parent_poly_for_.back()->body.As<ir::Block>()) {
        ir::Block* body = parent_poly_for_.back()->body.As<ir::Block>();
        if (body->stmts.size() == 1 && body->stmts[0] == *expr) {
          parent_poly_for_.back()->body = ir::Block::Make({unit_loop});
          parent_poly_for_.push_back(unit_loop.As<ir::PolyFor>());
          longest_loop_.push_back(unit_loop);
          add_unit_loop = true;
        }
      }
    }
  }

  if (add_unit_loop) {
    ir::IRMutator<>::Visit(&(parent_poly_for_.back()->body),
                           &(parent_poly_for_.back()->body));
    parent_poly_for_.pop_back();
  } else {
    parent_poly_for_.push_back(node);
    longest_loop_.push_back(*expr);
    ir::IRMutator<>::Visit(&node->body, &node->body);
    parent_poly_for_.pop_back();
  }
}

void DomainAddUnitLoopMutator::MutateAfterVisit(ir::Expr* expr) {
  VLOG(6) << "DomainAddUnitLoopMutator::MutateAfterVisit";
  if (longest_loop_.size() >= dim_min_max_.size()) {
    // No loops to add
    return;
  }
  int loop_match_len = 0;
  for (int i = 0; i < longest_loop_.size(); ++i) {
    std::tuple<int, int, int> t = dim_min_max_[i];
    if (longest_loop_[i].As<ir::For>()) {
      const ir::For* node = longest_loop_[i].As<ir::For>();
      if (utils::StartsWith(node->loop_var->name, dim_names_[i]) &&
          node->min.is_constant() && node->min.as_int32() == std::get<1>(t) &&
          node->extent.is_constant() &&
          node->extent.as_int32() == std::get<2>(t)) {
        ++loop_match_len;
      } else {
        loop_match_len = -1;
        break;
      }
    } else if (longest_loop_[i].As<ir::PolyFor>()) {
      const ir::PolyFor* node = longest_loop_[i].As<ir::PolyFor>();
      if (utils::StartsWith(node->iterator->name, dim_names_[i]) &&
          node->init.is_constant() && node->init.as_int32() == std::get<1>(t) &&
          node->condition ==
              ir::LE::Make(ir::Var(dim_names_[i]), ir::Expr(std::get<2>(t)))) {
        ++loop_match_len;
      } else {
        loop_match_len = -1;
        break;
      }
    } else {
      loop_match_len = -1;
      break;
    }
  }

  if (loop_match_len == -1 || loop_match_len >= dim_min_max_.size()) {
    // Not matched loops, shouldn't change anything
    return;
  }
  for (int i = loop_match_len; i < dim_min_max_.size(); ++i) {
    std::tuple<int, int, int> t = dim_min_max_[i];
    if (std::get<2>(t) != std::get<1>(t)) {
      // Not all remaining loops are length 1, just return
      return;
    }
  }

  if (longest_loop_.empty() || longest_loop_.back().As<ir::PolyFor>()) {
    ir::Expr body = longest_loop_.empty()
                        ? *expr
                        : longest_loop_.back().As<ir::PolyFor>()->body;
    for (int i = dim_min_max_.size() - 1; i >= loop_match_len; --i) {
      if (!body.As<ir::Block>()) {
        body = ir::Block::Make({body});
      }
      body = ir::PolyFor::Make(
          ir::Var(dim_names_[i]),
          ir::Expr(0),
          ir::LE::Make(ir::Var(dim_names_[i]), ir::Expr(0)),
          ir::Expr(1),
          ir::ForType::Serial,
          longest_loop_.empty()
              ? ir::DeviceAPI::UNK
              : longest_loop_.back().As<ir::PolyFor>()->device_api,
          body);
    }
    if (longest_loop_.empty()) {
      *expr = body;
    } else {
      longest_loop_.back().As<ir::PolyFor>()->body = ir::Block::Make({body});
    }
  } else if (longest_loop_.back().As<ir::For>()) {
    ir::For* node = longest_loop_.back().As<ir::For>();
    ir::Expr body = node->body;
    for (int i = dim_min_max_.size() - 1; i >= loop_match_len; --i) {
      ir::Expr unit_loop = ir::For::Make(ir::Var(dim_names_[i]),
                                         ir::Expr(0),
                                         ir::Expr(1),
                                         ir::ForType::Serial,
                                         node->device_api,
                                         body);
      body = ir::Block::Make({unit_loop});
    }
    node->body = body;
  }
}

}  // namespace poly
}  // namespace cinn
