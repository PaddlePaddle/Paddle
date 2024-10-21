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

#include "paddle/cinn/optim/for_cse_optimize.h"

#include <stack>

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/optim/ir_simplify.h"

namespace cinn {
namespace optim {

namespace {

using namespace ir;  // NOLINT

enum class IndexRelated : int { UnRelated = 0, Related = 1 };

std::ostream &operator<<(std::ostream &os, const IndexRelated &x) {
  os << static_cast<int>(x);
  return os;
}

struct InsertNode {
  ir::Expr parent_expr;
  ir::Expr base_expr;
  std::vector<ir::Expr> new_insert_expr;

  InsertNode(const ir::Expr &parent_expr_t,
             const ir::Expr &base_expr_t,
             const std::vector<ir::Expr> &vec_insert_expr)
      : parent_expr(parent_expr_t),
        base_expr(base_expr_t),
        new_insert_expr(vec_insert_expr) {}
};

template <typename T = Expr *>
class TestMutator : public ir::IRMutator<> {
 public:
  // explicit TestMutator( const std::vector<Var> iter_var);

  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void print_map() {
    std::cerr << "!!!!!!" << std::endl;
    for (auto it = expr_map_.begin(); it != expr_map_.end(); ++it) {
      std::cerr << "expr !! " << it->first << std::endl;

      for (auto &op : it->second) {
        std::cerr << "used by " << *op << std::endl;
      }
    }
  }

  void process_call(Expr *op, Expr orig, Expr new_value) {
    auto call_op = op->As<ir::Call>();
  }

  void process_binary(std::vector<Expr> *opreands,
                      Expr orig,
                      Expr new_value) {  // NOLINT
    if ((*opreands)[0].get() == orig.get()) {
      (*opreands)[0] = new_value;
    } else if ((*opreands)[1].get() == orig.get()) {
      (*opreands)[1] = new_value;
    } else {
      // PADDLE_THORW(phi::errors::PermissionDenied("binary error"));
      std::cerr << "cant find match input\n";
      throw std::runtime_error("cant find matc input");
    }
  }

  void replace_all_user(Expr orig, Expr new_value) {
    PADDLE_ENFORCE_EQ(expr_map_.count(orig),
                      true,
                      phi::errors::PreconditionNotMet("Expr must in expr map"));

    auto replace_op_list = expr_map_.at(orig);

    for (auto *op : replace_op_list) {
      if (auto call_op = op->As<Call>()) {
        call_op->read_args[0] = new_value;
      } else if (auto binary_op = op->As<Add>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<Sub>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<Mul>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<Div>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<Mod>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<EQ>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<NE>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<LT>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<LE>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<GT>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<GE>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<And>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<Or>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<Min>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<Max>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto binary_op = op->As<GE>()) {
        process_binary(&binary_op->operands(), orig, new_value);
      } else if (auto minus_op = op->As<Minus>()) {
        minus_op->v() = new_value;
      } else if (auto not_op = op->As<Not>()) {
        not_op->v() = new_value;
      } else if (auto store_op = op->As<Store>()) {
        store_op->value = new_value;
      } else {
        std::cerr << "not support op " << op->node_type() << std::endl;
        throw std::runtime_error("not support op");
        // PADDLE_THROW(::common::err::Unimplemented("not support op"));
      }
    }
  }

  std::vector<Expr> get_un_related_expr() {
    std::cerr << "path list size " << path_list_.size() << std::endl;
    int index = 0;
    for (auto &v : path_list_) {
      std::cerr << index++ << "\t" << v.first << "\t" << v.second << std::endl;
    }

    size_t i = 0;
    std::unordered_set<Expr> visited_set;
    std::vector<Expr> output_expr;

    while (visited_set.size() * 2 < path_list_.size() &&
           i + 1 < path_list_.size()) {
      std::cerr << "i  " << i << "\t" << visited_set.size() << std::endl;
      if (visited_set.count(path_list_[i].first)) {
        ++i;
        continue;
      }
      bool related = false;
      // bool related = path_list_[i].second == IndexRelated::Related;
      // if(  related )
      // {
      //   ++i;
      //   visited_set.insert(path_list_[i].first);
      //   continue;
      // }
      size_t j = i;
      bool meet_end = false;
      for (; j < path_list_.size(); ++j) {
        if (path_list_[j].second == IndexRelated::Related) {
          related = true;
          break;
        }

        if (j != i && path_list_[j].first.get() == path_list_[i].first.get()) {
          meet_end = true;
          break;
        }
      }

      if (related) {
        ++i;
        visited_set.insert(path_list_[i].first);
      } else if (meet_end) {
        for (size_t k = i; k <= j; ++k) {
          visited_set.insert(path_list_[k].first);
        }

        std::cerr << "i " << i << " j " << j << std::endl;
        std::cerr << "un relate " << path_list_[i].first << std::endl;

        output_expr.push_back(path_list_[i].first);

        i = j + 1;
      } else {
        ++i;
        visited_set.insert(path_list_[i].first);
      }

      // std::cerr << "i  "  <<  i << std::endl;
      // break;
    }

    return output_expr;
  }

 private:
#define __(op__) void Visit(const op__ *op, T expr) override;
  NODETY_OP_FOR_EACH(__)
#undef __

  void Visit(const Call *op, Expr *expr) override {
    std::cerr << "call op " << op->name << std::endl;
    path_list_.push_back(std::make_pair(*expr, IndexRelated::UnRelated));

    std::cerr << "input" << op->read_args[0] << std::endl;
    expr_map_[op->read_args[0]].push_back(expr);
    ir::IRMutator<>::Visit(op, expr);

    path_list_.push_back(std::make_pair(*expr, IndexRelated::UnRelated));
  }

  void Visit(const Store *op, Expr *expr) override {
    // path_list_.push_back( std::make_pair(*expr, IndexRelated::UnRelated));

    auto store_node = expr->As<Store>();
    expr_map_[store_node->value].push_back(expr);
    IRVisitorRequireReImpl<void, T>::Visit(&store_node->value,
                                           &store_node->value);

    // path_list_.push_back( std::make_pair(*expr, IndexRelated::UnRelated));
  }

  void Visit(const Load *op, Expr *expr) override {
    // path_list_.push_back( std::make_pair(*expr, IndexRelated::UnRelated));

    // IRVisitorRequireReImpl<void, T>::Visit( &op->value, &op->value);

    // path_list_.push_back( std::make_pair(*expr, IndexRelated::UnRelated));
    auto indices = op->indices;

    auto is_related = IndexRelated::UnRelated;

    for (auto &idx : indices) {
      if (!idx.is_constant()) {
        is_related = IndexRelated::Related;
      }
    }

    path_list_.push_back(std::make_pair(*expr, is_related));
    path_list_.push_back(std::make_pair(*expr, is_related));
  }

  void Visit(const ScheduleBlockRealize *op, Expr *expr) override {
    auto *node = expr->As<ScheduleBlockRealize>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "Node is null. Ensure that the node is "
                                "properly initialized and not null."));

    IRVisitorRequireReImpl<void, T>::Visit(&node->schedule_block,
                                           &node->schedule_block);
  }

  std::unordered_map<Expr, std::vector<Expr *>> expr_map_;

  std::vector<std::pair<Expr, IndexRelated>> path_list_;
};

#define UNARY_OP_IMPL(op__)                                             \
  template <typename T>                                                 \
  void TestMutator<T>::Visit(const op__ *expr, T op) {                  \
    auto *node = op->template As<op__>();                               \
    std::cerr << "unaray !" << std::endl;                               \
    std::cerr << node->v() << std::endl;                                \
    expr_map_[node->v()].push_back(op);                                 \
    path_list_.push_back(std::make_pair(*op, IndexRelated::UnRelated)); \
    IRVisitorRequireReImpl<void, T>::Visit(&node->v(), &node->v());     \
    path_list_.push_back(std::make_pair(*op, IndexRelated::UnRelated)); \
  }

#define BINARY_OP_IMPL(op__)                                              \
  template <typename T>                                                   \
  void TestMutator<T>::Visit(const op__ *expr, T op) {                    \
    auto *node = op->template As<op__>();                                 \
    std::cerr << "binary  " << std::endl;                                 \
    std::cerr << "input " << node->a() << "\t" << node->b() << std::endl; \
    expr_map_[node->a()].push_back(op);                                   \
    expr_map_[node->b()].push_back(op);                                   \
    path_list_.push_back(std::make_pair(*op, IndexRelated::UnRelated));   \
    IRVisitorRequireReImpl<void, T>::Visit(&node->a(), &node->a());       \
    IRVisitorRequireReImpl<void, T>::Visit(&node->b(), &node->b());       \
    path_list_.push_back(std::make_pair(*op, IndexRelated::UnRelated));   \
  }

NODETY_UNARY_OP_FOR_EACH(UNARY_OP_IMPL)
NODETY_BINARY_OP_FOR_EACH(BINARY_OP_IMPL)

#undef UNARY_OP_IMPL
#undef BINARY_OP_IMPL

class ForLoopCSETest : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void insert_new_node() {
    std::cerr << "begin to insert\n";
    std::cerr << "insert size " << insert_node_list.size() << std::endl;
    for (size_t i = 0; i < insert_node_list.size(); ++i) {
      if (auto block_op = insert_node_list[i].parent_expr.As<ir::Block>()) {
        auto &op_list = block_op->stmts;
        std::cerr << "op list size  " << op_list.size() << std::endl;

        int match_index = -1;
        for (size_t k = 0; k < op_list.size(); ++k) {
          if (op_list[k].get() == insert_node_list[i].base_expr.get()) {
            match_index = k;
            break;
          }
        }
        if (match_index == -1) {
          std::cerr << "can not match expr  " << insert_node_list[i].base_expr
                    << std::endl;
          throw std::runtime_error("can not match base expr");
        }

        for (auto &e : insert_node_list[i].new_insert_expr) {
          std::cerr << "begin to insert !! " << e << std::endl;
          op_list.insert(op_list.begin() + match_index, e);
        }
      } else {
        std::cerr << "not support " << insert_node_list[i].parent_expr
                  << std::endl;
        throw std::runtime_error("only support id block");
      }
    }
  }

 private:
  void Visit(const ir::IfThenElse *op, Expr *expr) override {
    // std::cerr << "for body " << op->body << std::endl;

    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Block *op, Expr *expr) override {
    std::cerr << "block op \n";
    for (auto e : op->stmts) {
      std::cerr << "!!!!!!!! ==================\n" << e << std::endl;
    }

    inner_stack.push(*expr);
    ir::IRMutator<>::Visit(op, expr);
    inner_stack.pop();
  }

  void Visit(const ir::For *op, Expr *expr) override {
    // std::cerr << "for body " << op->body << std::endl;

    auto for_node = expr->As<For>();
    auto body_block = for_node->body.As<ir::Block>();

    for (size_t i = 0; i < body_block->stmts.size(); ++i) {
      std::cerr << "for inner \n\ni " << i << "\t" << body_block->stmts[i]
                << std::endl;

      TestMutator test_mutator_1;
      test_mutator_1(&(body_block->stmts[i]));

      auto unrelated_expr_list = test_mutator_1.get_un_related_expr();

      for (size_t i = 0; i < unrelated_expr_list.size(); ++i) {
        Var idx("t_i_" + std::to_string(insert_idx++),
                unrelated_expr_list[i].type());
        std::cerr << "EEEE " << unrelated_expr_list[i] << std::endl;

        auto outer_expr = ir::Let::Make(idx, unrelated_expr_list[i]);

        std::cerr << "outer expr " << outer_expr << std::endl;

        // TODO(phlrain): Add check current For Expr is in inner_stack.top()
        // Expr

        InsertNode insert_node(inner_stack.top(), *expr, {outer_expr});
        insert_node_list.push_back(insert_node);
        test_mutator_1.replace_all_user(unrelated_expr_list[i], idx);
      }

      std::cerr << "insert stack\n\n" << inner_stack.top() << std::endl;
    }
  }

  std::stack<Expr> inner_stack;

  std::vector<InsertNode> insert_node_list;

  int insert_idx{0};
};

}  // namespace

void ForCSEOptimize(Expr *expr) {
  ForLoopCSETest for_loop_ces_test;

  for_loop_ces_test(expr);

  for_loop_ces_test.insert_new_node();

  std::cerr << "!!!!!!!!! ~~~~~~~~~~~~~~~~~~~\n" << *expr << std::endl;
}
}  // namespace optim
}  // namespace cinn
