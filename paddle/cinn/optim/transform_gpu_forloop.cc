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

#include "paddle/cinn/optim/transform_gpu_forloop.h"

#include <algorithm>
#include <map>
#include <stack>
#include <string>
#include <vector>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

/**
 * 1. Determine the grid and block dimensions.
 * It takes the domains like `[0, 20]` or `[0, min(20, M/2)]`, the domain should
 * have a integer right bound.
 *
 * 2. Replace the grid/thread iterators with something like `threadIdx.x`,
 * `threadIdx.y`.
 *
 * 3. Remove the forloops owning the gpu axis.
 *   1. if the extent is an IntImm, just remove this forloop.
 *   2. if the extent is a Min, replace the forloop with an IfThenElse, with
 * forloop's condition, new check will add (if the min of forloop is not zero).
 *
 * @param expr The expression to mutate.
 */
void RemoveGpuForloopsAxis(Expr *expr) {
  struct Mutator : public ir::IRMutator<Expr *> {
    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const ir::For *op, Expr *expr) override {
      switch (op->for_type()) {
        case ir::ForType::GPUBlock:
          if (NeedToReplaceForloopWithIfThenElse(op)) {
            ReplaceForloopWithIfThenElse(expr);
          } else {
            *expr = op->body;
          }
          IRMutator<>::Visit(expr, expr);
          break;
        case ir::ForType::GPUThread:
          if (NeedToReplaceForloopWithIfThenElse(op)) {
            ReplaceForloopWithIfThenElse(expr);
          } else {
            *expr = op->body;
          }
          IRMutator<>::Visit(expr, expr);
          break;
        default:
          auto *node = expr->As<ir::For>();
          IRMutator<>::Visit(&node->body, &node->body);
          break;
      }
    }

    bool NeedToReplaceForloopWithIfThenElse(const ir::For *n) const {
      return true;
    }

    void ReplaceForloopWithIfThenElse(Expr *expr) {
      auto *for_n = expr->As<ir::For>();
      auto *poly_for_n = expr->As<ir::PolyFor>();
      CHECK(for_n || poly_for_n);

      Expr condition;

      auto condition_append = [&](Expr new_cond) {
        if (condition.defined()) {
          condition = ir::And::Make(condition, new_cond);
        } else {
          condition = new_cond;
        }
      };

      if (for_n) {
        // for(i, 2, 100);
        //        ^
        if (for_n->min != common::make_const(0)) {
          condition_append(ir::GE::Make(for_n->loop_var, for_n->min));
        }

        // for(i, 2, min(M/2, 20)
        //            ^
        condition_append(ir::LT::Make(for_n->loop_var, for_n->extent));
      } else {
        if (poly_for_n->init != common::make_const(0)) {
          condition_append(
              ir::GE::Make(poly_for_n->iterator, poly_for_n->init));
        }

        condition_append(poly_for_n->condition);
      }

      CHECK(condition.defined());

      VLOG(3) << "GPU replacing\n" << *expr;
      VLOG(3) << "\nto\n";
      auto if_n = ir::IfThenElse::Make(condition, for_n->body);
      VLOG(3) << if_n;
      *expr = if_n;
    }

    void Visit(const ir::PolyFor *op, Expr *expr) override {
      const auto msg =
          "PolyFor is not allowed for GPU, only For nodes are allowed";
      CHECK(op->for_type() != ir::ForType::GPUBlock) << msg;
      CHECK(op->for_type() != ir::ForType::GPUThread) << msg;
      CHECK(op->for_type() != ir::ForType::GPULane) << msg;
    }
  };

  Mutator mutator;
  mutator(expr);
}

/**
 * The generated __syncthreads call will be wrapped with a `if (xxxx == 0) { }`,
 * this is the problem of isl AST output, drop it to make it run in all the
 * threads.
 */
void CudaSyncThreadsDropIfThenElse(Expr *expr) {
  struct Mutator : public ir::IRMutator<> {
    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::IfThenElse *op, Expr *expr) override {
      blocked_statement_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      blocked_statement_stack.pop_back();
    }

    void Visit(const ir::Call *op, Expr *expr) override {
      if (op->name == runtime::intrinsic::cuda_sync_threads) {
        if (!blocked_statement_stack.empty()) {
          auto *last_for = blocked_statement_stack.back()->As<ir::IfThenElse>();
          if (auto *eq_n = last_for->condition.As<ir::EQ>()) {
            if (eq_n->b() == common::make_const(0)) {
              *blocked_statement_stack.back() = *expr;
            }
          }
        }
      }
    }

    // Collect all the statements with Block(include Block) to the statement.
    std::vector<ir::Expr *> blocked_statement_stack;
  };

  Mutator()(expr);
}

class RestructureVarNodes : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Load *load, Expr *op) override {
    std::vector<ir::Expr> indices_copied;
    for (const ir::Expr &indice : load->indices) {
      indices_copied.push_back(ir::ir_utils::IRCopy(indice));
    }
    op->As<ir::Load>()->indices = indices_copied;

    IRMutator::Visit(load, op);
  }

  void Visit(const ir::Store *store, Expr *op) override {
    std::vector<ir::Expr> indices_copied;
    for (const ir::Expr &indice : store->indices) {
      indices_copied.push_back(ir::ir_utils::IRCopy(indice));
    }
    op->As<ir::Store>()->indices = indices_copied;

    IRMutator::Visit(store, op);
  }
};

class ReplaceIndexToBindExpr : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlockRealize *op, Expr *expr) override {
    ir::ScheduleBlockRealize *schedule_block_realize =
        expr->As<ir::ScheduleBlockRealize>();
    CHECK(schedule_block_realize->schedule_block.As<ir::ScheduleBlock>());
    std::vector<ir::Expr> iter_values = schedule_block_realize->iter_values;
    ir::Expr body =
        schedule_block_realize->schedule_block.As<ir::ScheduleBlock>()->body;
    std::vector<ir::Var> iter_vars =
        schedule_block_realize->schedule_block.As<ir::ScheduleBlock>()
            ->iter_vars;

    CHECK_EQ(iter_values.size(), iter_vars.size());
    for (int idx = 0; idx < iter_values.size(); ++idx) {
      ReplaceVarWithExpr(&body, iter_vars[idx], iter_values[idx]);
    }
    ir::IRMutator<>::Visit(&body, &body);
  }
};

using TENSOR_LOOP = std::pair<ir::Expr, std::vector<ir::Expr>>;
class CollectTensorLoopVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto tensor = op->tensor.as_tensor_ref();
    // if buffer defined and buffer is not Heap.
    if (tensor->buffer.defined() &&
        tensor->buffer->memory_type != ir::MemoryType::Heap) {
      if (buffer_tensor_loop_map_.count(tensor->buffer->name)) {
        buffer_tensor_loop_map_[tensor->buffer->name].push_back(
            std::make_pair(*expr, loops_));
      } else {
        buffer_tensor_loop_map_[tensor->buffer->name] = {
            std::make_pair(*expr, loops_)};
      }
    }

    IRMutator::Visit(op, expr);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    if (op->is_addr_scalar()) {
      return;
    }
    auto tensor = op->tensor.as_tensor_ref();
    // if buffer defined and buffer is not Heap.
    if (tensor->buffer.defined() &&
        tensor->buffer->memory_type != ir::MemoryType::Heap) {
      if (buffer_tensor_loop_map_.count(tensor->buffer->name)) {
        buffer_tensor_loop_map_[tensor->buffer->name].push_back(
            std::make_pair(*expr, loops_));
      } else {
        buffer_tensor_loop_map_[tensor->buffer->name] = {
            std::make_pair(*expr, loops_)};
      }
    }

    IRMutator::Visit(op, expr);
  }

  void Visit(const ir::For *op, Expr *expr) override {
    loops_.push_back(*expr);
    IRMutator::Visit(op, expr);
    loops_.pop_back();
  }

  void Visit(const ir::PolyFor *op, Expr *expr) override {
    LOG(FATAL) << "Unkown PolyFor!";
  }

 public:
  std::vector<ir::Expr> loops_;
  std::unordered_map<std::string, std::vector<TENSOR_LOOP>>
      buffer_tensor_loop_map_;
};

void UpdateBufferAxisPass(ir::Expr *expr) {
  CollectTensorLoopVisitor collect_tensor_loop_visitor;
  collect_tensor_loop_visitor(expr);

  auto buffer_tensor_loop = collect_tensor_loop_visitor.buffer_tensor_loop_map_;

  for (auto &tmp : buffer_tensor_loop) {
    auto tensor_loop_v = tmp.second;

    auto &front = tensor_loop_v.front();
    int count = tensor_loop_v.size() > 1 ? front.second.size() : 0;
    for (int idx = 1; idx < tensor_loop_v.size(); ++idx) {
      auto &other = tensor_loop_v[idx];
      for (int idy = 0;
           idy < std::min(front.second.size(), other.second.size());
           ++idy) {
        if (front.second[idy] != other.second[idy]) {
          count = std::min(count, idy);
          break;
        }
      }
    }

    auto get_thread_bind_var = [](const std::vector<ir::Expr> &loops) {
      // threadidx loop_var,extent.
      using ThreadLoopVarExtentMap =
          std::unordered_map<std::string, std::pair<std::string, int>>;
      ThreadLoopVarExtentMap thread_loop_var_exent_map;
      for (auto loop : loops) {
        auto loop_ir = loop.As<ir::For>();
        CHECK(loop_ir);
        if (loop_ir->is_gpu_thread_binded()) {
          std::string axis = "";
          if (loop_ir->bind_info().offset == 0) {
            axis = "threadIdx.x";
          } else if (loop_ir->bind_info().offset == 1) {
            axis = "threadIdx.y";
          } else {
            axis = "threadIdx.z";
          }
          // insert gpu thread loop var.
          if (thread_loop_var_exent_map.count(axis)) {
            auto &loop_var_extent = thread_loop_var_exent_map[axis];
            if (loop_var_extent.second >= loop_ir->extent.as_int32()) {
              thread_loop_var_exent_map[axis] = std::make_pair(
                  loop_ir->loop_var->name, loop_ir->extent.as_int32());
            }
          } else {
            thread_loop_var_exent_map[axis] = std::make_pair(
                loop_ir->loop_var->name, loop_ir->extent.as_int32());
          }
        }
      }

      std::unordered_set<std::string> loop_var_map;
      for (auto &tmp : thread_loop_var_exent_map) {
        loop_var_map.insert(tmp.second.first);
      }

      return loop_var_map;
    };

    auto load = front.first.As<ir::Load>();
    auto store = front.first.As<ir::Store>();
    auto tensor =
        load ? load->tensor.as_tensor_ref() : store->tensor.as_tensor_ref();
    // find store and load keep loop for shared
    std::vector<std::unordered_set<std::string>> keep_loop_vars;
    if (tensor->buffer->memory_type == ir::MemoryType::GPUShared) {
      for (auto &tensor_loop : tensor_loop_v) {
        keep_loop_vars.push_back(get_thread_bind_var(tensor_loop.second));
      }
      CHECK_EQ(keep_loop_vars.size(), tensor_loop_v.size());
    }

    auto &loops = front.second;
    for (int idx = 0; idx < count; ++idx) {
      auto loop_expr = loops[idx];
      auto loop_ir = loop_expr.As<ir::For>();
      auto loop_var = loop_ir->loop_var;

      for (int idy = 0; idy < tensor_loop_v.size(); ++idy) {
        auto expr = tensor_loop_v[idy].first;
        auto load = expr.As<ir::Load>();
        auto store = expr.As<ir::Store>();
        if (keep_loop_vars.size() == 0 ||
            !keep_loop_vars[idy].count(loop_var->name)) {
          auto &indices = load ? load->indices : store->indices;
          for (auto &indice : indices) {
            optim::ReplaceVarWithExpr(&indice, loop_var, ir::Expr(0));
            indice = common::AutoSimplify(indice);
          }
        }
      }
    }
  }
}

class ReplaceLoopVarToGpu : public ir::IRMutator<> {
 public:
  void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::For *op, Expr *expr) override {
    auto for_ir = expr->As<ir::For>();
    CHECK(for_ir);

    auto bind_info = for_ir->bind_info();

    std::string var_name = "";
    if (bind_info.offset == 0)
      var_name = "x";
    else if (bind_info.offset == 1)
      var_name = "y";
    else if (bind_info.offset == 2)
      var_name = "z";
    if (for_ir->is_gpu_block_binded()) {
      var_name = "blockIdx." + var_name;
      optim::ReplaceVarWithExpr(
          expr, op->loop_var, ir::Expr(ir::Var(var_name)));
    } else if (for_ir->is_gpu_thread_binded()) {
      var_name = "threadIdx." + var_name;
      optim::ReplaceVarWithExpr(
          expr, op->loop_var, ir::Expr(ir::Var(var_name)));
    }

    ir::IRMutator<>::Visit(&for_ir->body, &for_ir->body);
  }
  void Visit(const ir::PolyFor *op, Expr *expr) override {
    LOG(FATAL) << "Unkown PolyFor!";
  }
};

class SharedAxisVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store = expr->As<ir::Store>();
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (store->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPUShared) {
      for (auto &indice : store->indices) {
        for (auto axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&indice, ir::Var(axis), ir::Expr(0));
        }
        indice = common::AutoSimplify(indice);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (load->is_addr_scalar()) {
      return;
    }
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPUShared) {
      for (auto &indice : load->indices) {
        for (auto axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&indice, ir::Var(axis), ir::Expr(0));
        }
        indice = common::AutoSimplify(indice);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  const std::vector<std::string> gpu_axis = {
      "blockIdx.x", "blockIdx.y", "blockIdx.z"};
};

class LocalAxisVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store = expr->As<ir::Store>();
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (store->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      for (auto &indice : store->indices) {
        for (auto axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&indice, ir::Var(axis), ir::Expr(0));
        }
        indice = common::AutoSimplify(indice);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (load->is_addr_scalar()) {
      return;
    }
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      for (auto &indice : load->indices) {
        for (auto axis : gpu_axis) {
          optim::ReplaceVarWithExpr(&indice, ir::Var(axis), ir::Expr(0));
        }
        indice = common::AutoSimplify(indice);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  const std::vector<std::string> gpu_axis = {"blockIdx.x",
                                             "blockIdx.y",
                                             "blockIdx.z",
                                             "threadIdx.x",
                                             "threadIdx.y",
                                             "threadIdx.z"};
};

class ResizeBufferSizeVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store = expr->As<ir::Store>();
    auto store_tensor = store->tensor.as_tensor_ref();

    if (!store_tensor->buffer.defined()) {
      return;
    }
    if (store_tensor->buffer->memory_type == ir::MemoryType::Heap) {
      ir::IRMutator<>::Visit(op, expr);
      return;
    }

    auto &indices = store->indices;
    auto &shape = store_tensor->shape;
    auto &buffer = store_tensor->buffer->shape;

    shape.clear();
    buffer.clear();
    for (int idx = 0; idx < indices.size(); ++idx) {
      shape.push_back(ir::Expr(BufferSize(indices[idx])));
      buffer.push_back(shape.back());
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::Heap) {
      ir::IRMutator<>::Visit(op, expr);
      return;
    }

    load->tensor.as_tensor_ref()->shape =
        load->tensor.as_tensor_ref()->buffer->shape;

    // For the moment, align the load tensor indices with the tensor shape using
    // the trick method. A better way would be to modify the FlattenLoop
    // Schedule.
    int cnt = load->indices.size() - load->tensor.as_tensor_ref()->shape.size();
    for (int i = 0; i < cnt; i++) {
      load->indices.erase(load->indices.begin());
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For *op, Expr *expr) override {
    CHECK(expr->As<ir::For>());
    auto for_ir = expr->As<ir::For>();
    auto var_name = for_ir->loop_var->name;
    auto extent_i = for_ir->extent;

    if (extent_i.is_constant()) loop_2_extent_[var_name] = extent_i.as_int32();
    ir::IRMutator<>::Visit(op, expr);
  }

  int BufferSize(ir::Expr indice) {
    auto copy = ir::ir_utils::IRCopy(indice);
    auto vars = ir::ir_utils::CollectIRNodesInOrder(
        copy, [](const ir::Expr *expr) { return expr->As<ir::_Var_>(); });

    int max_range = 1;
    // using recursion funcitons index range.
    std::function<void(int, ir::Expr)> compute_range = [&](const int deep,
                                                           ir::Expr index) {
      auto var = vars[deep].as_var_ref();
      CHECK(loop_2_extent_.count(var->name)) << var->name;
      auto extent = loop_2_extent_.find(var->name)->second;

      for (int idx = 0; idx < extent; ++idx) {
        auto tmp = ir::ir_utils::IRCopy(index);
        ReplaceVarWithExpr(&tmp, var, Expr(idx));

        if (deep == vars.size() - 1) {
          auto simplify = common::AutoSimplify(tmp);
          auto range = common::AutoSimplify(simplify);
          CHECK(range.is_constant());
          max_range = std::max(max_range, range.as_int32() + 1);
        } else {
          compute_range(deep + 1, tmp);
        }
      }
    };

    if (vars.size()) compute_range(0, copy);
    return max_range;
  }

  std::unordered_map<std::string, int> loop_2_extent_;
};

class ReplaceVarToZero : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store = expr->As<ir::Store>();
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    auto &indices = store->indices;
    for (auto &indice : indices) {
      for (auto var_ : loop_var_) {
        optim::ReplaceVarWithExpr(&indice, ir::Var(var_), ir::Expr(0));
      }
      indice = common::AutoSimplify(indice);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    auto &indices = load->indices;
    for (auto &indice : indices) {
      for (auto var_ : loop_var_) {
        optim::ReplaceVarWithExpr(&indice, ir::Var(var_), ir::Expr(0));
      }
      indice = common::AutoSimplify(indice);
    }

    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::For *op, Expr *expr) override {
    CHECK(expr->As<ir::For>());
    auto for_ir = expr->As<ir::For>();
    auto var_name = for_ir->loop_var->name;
    auto extent_i = for_ir->extent;

    if (extent_i.is_constant() && extent_i.as_int32() == 1)
      loop_var_.insert(var_name);
    ir::IRMutator<>::Visit(op, expr);
    loop_var_.erase(var_name);
  }
  std::unordered_set<std::string> loop_var_;
};

void OptimizeExprGPU(Expr *expr) {
  VLOG(2) << "Before Optimize Expr:\n" << *expr;

  // copy var nodes to prevent one modification leading to multiple changes
  RestructureVarNodes restructure_var_nodes;
  restructure_var_nodes(expr);

  // replace var to bind expr
  ReplaceIndexToBindExpr replace_index_to_bind_expr;
  replace_index_to_bind_expr(expr);

  // resize buffer axis
  UpdateBufferAxisPass(expr);

  // replace var name with block/thread
  ReplaceLoopVarToGpu replace_loop_var_to_gpu;
  replace_loop_var_to_gpu(expr);

  // update shared buffer axis
  SharedAxisVisitor shared_axis_visitor;
  shared_axis_visitor(expr);

  // update local buffer axis
  LocalAxisVisitor local_axis_visitor;
  local_axis_visitor(expr);

  ResizeBufferSizeVisitor resize_buffer_size_visitor;
  resize_buffer_size_visitor(expr);

  ReplaceVarToZero replace_var_to_zero;
  replace_var_to_zero(expr);

  VLOG(2) << "After Optimize Expr: \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
