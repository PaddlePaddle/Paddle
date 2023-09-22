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

#include "paddle/cinn/optim/compute_inline_expand.h"

#include <map>
#include <string>

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

namespace {

/*
 * Replace a tensor(marked as compute_inline) to the expanded expression.
 */
struct TensorInlineExpandMutator : public ir::IRMutator<> {
  const std::string &tensor_name_;
  std::map<std::string, ir::Tensor> *all_tensor_map_;
  poly::StageMap stages_;
  bool inline_code{false};
  bool temp_buffer{false};
  bool memory_local{false};
  std::unordered_map<std::string, std::vector<Expr>> resized_buffer_cache;
  std::vector<std::string> tensor_names;
  std::vector<std::vector<Var>> replace_var;
  std::map<std::string, Expr> var_to_extent;

  TensorInlineExpandMutator(const std::string &tensor_name,
                            std::map<std::string, ir::Tensor> *all_tensor_map,
                            poly::StageMap stages)
      : tensor_name_(tensor_name),
        all_tensor_map_(all_tensor_map),
        stages_(stages) {}

  void operator()(Expr *expr) {
    ir::IRMutator<>::Visit(expr, expr);
    for (int i = 0; i < tensor_names.size(); i++) {
      for (auto &var : replace_var[i]) {
      }
    }
  }

  void Visit(const ir::_Var_ *expr, Expr *op) override {
    if (inline_code && temp_buffer) {
      if (utils::Startswith(expr->name, "blockIdx") ||
          (utils::Startswith(expr->name, "threadIdx") && memory_local)) {
        *op = ir::Expr(0);
      }
    }
  }

  void Visit(const ir::_Tensor_ *op, Expr *expr) override {
    if (inline_code && utils::Endswith(op->name, "_write_cache") &&
        (*all_tensor_map_).at(op->name)->buffer->memory_type ==
            ir::MemoryType::Heap) {
      auto no_cache_name = op->name.substr(0, op->name.size() - 12);
      VLOG(2) << "no_cache_name: " << no_cache_name;
      CHECK(all_tensor_map_->count(no_cache_name));
      *expr = (*all_tensor_map_)[no_cache_name];
    }
  }

  void Visit(const ir::For *op, Expr *expr) override {
    CHECK(op->extent.is_constant());
    int cons_extent = static_cast<int>(op->extent.get_constant());
    var_to_extent[op->loop_var->name] = op->extent;
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::PolyFor *op, Expr *expr) override {
    auto extent = op->ExtractExtent();
    var_to_extent[op->iterator->name] = extent;
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto *node = expr->As<ir::Load>();
    auto *tensor = node->tensor.as_tensor();
    if (tensor && tensor->name == tensor_name_) {
      *expr = tensor->inline_expanded(op->indices);
      inline_code = true;
      ir::IRMutator<>::Visit(expr, expr);
      inline_code = false;
    } else if (inline_code && tensor->buffer.defined()) {
      bool is_heap = (*all_tensor_map_).at(tensor->name)->buffer->memory_type ==
                     ir::MemoryType::Heap;
      if (utils::Endswith(tensor->buffer->name, "_write_cache") && is_heap) {
        // temp fix: cache_write will change the tensor to the cache tensor
        // wrongly
        auto no_cache_name =
            tensor->buffer->name.substr(1, tensor->buffer->name.size() - 13);
        if (all_tensor_map_->count(no_cache_name)) {
          ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
        } else {
          auto *tensor = node->tensor.as_tensor();
          CHECK(tensor);
          // fix computeAt case
          auto shapes = tensor->shape;
          CHECK_EQ(shapes.size(), node->indices.size());
          for (int i = 0; i < shapes.size(); i++) {
            if (common::is_zero(shapes[i] - 1)) {
              node->indices[i] = Expr(0);
            }
          }
        }
      } else if (utils::Endswith(tensor->buffer->name, "_write_cache") ||
                 utils::Endswith(tensor->buffer->name, "_read_cache") ||
                 utils::Endswith(tensor->buffer->name, "_temp_buffer")) {
#ifdef CINN_WITH_CUDA
        auto axis_names = stages_[tensor]->axis_names();
        auto compute_ats = stages_[tensor]->GetComputeAts();
        if (compute_ats.size() == 1) {
          int level_tmp;
          for (auto &i : compute_ats) {
            level_tmp = i.second.level;
          }
          std::vector<Var> replace_vars;
          for (int j = 0; j <= level_tmp; j++) {
            if (var_to_extent.count(axis_names[j]) == 0) continue;
            replace_vars.push_back(
                Var(var_to_extent[axis_names[j]], axis_names[j]));
          }
          replace_var.push_back(replace_vars);
          tensor_names.push_back(tensor->buffer->name);
        }
#endif
        bool keep_buffer = temp_buffer;
        temp_buffer = true;
        bool keep_memory_local = memory_local;
        if ((*all_tensor_map_).at(tensor->name)->buffer->memory_type ==
            ir::MemoryType::GPULocal) {
          memory_local = true;
        }
        ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
        for (int i = 0; i < node->indices.size(); i++) {
          auto temp = ir::ir_utils::IRCopy(node->indices[i]);
          ir::IRMutator<>::Visit(&temp, &temp);
          node->indices[i] = temp;
        }
        temp_buffer = keep_buffer;
        memory_local = keep_memory_local;
      } else {
        ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
        for (int i = 0; i < node->indices.size(); i++) {
          auto temp = ir::ir_utils::IRCopy(node->indices[i]);
          ir::IRMutator<>::Visit(&temp, &temp);
          node->indices[i] = temp;
        }
      }
    } else {
      ir::IRMutator<>::Visit(&node->tensor, &node->tensor);
      for (int i = 0; i < node->indices.size(); i++) {
        auto temp = ir::ir_utils::IRCopy(node->indices[i]);
        ir::IRMutator<>::Visit(&temp, &temp);
        node->indices[i] = temp;
      }
    }
  }
};

struct SSANode : public common::GraphNode {
  std::string id_;

  explicit SSANode(const std::string &id) : id_(id) {}

  std::string id() const override { return id_; }

  const char *type_info() const override { return __type_info__; }

  static constexpr char *__type_info__ = "optim::SSANode";
};

// TODO(Superjomn) the graph here is not a SSA now, it is flattern for the
// ir::CollectIRNodes method collects all the tensors recursively, so it can not
// reserve the level information, fix it.
struct SSABuilder : public ir::IRMutator<> {
  common::Graph graph;

  SSABuilder &operator()(Expr *expr) {
    ir::IRMutator<>::Visit(expr, expr);
    return *this;
  }

  void Visit(const ir::Store *op, Expr *expr) override {
    auto *node = expr->As<ir::Store>();

    auto *cur_graph_node = graph.RetrieveNode(node->tensor.as_tensor()->name);
    if (!cur_graph_node) {
      cur_graph_node =
          graph.RegisterNode(node->tensor.as_tensor()->name,
                             new SSANode(node->tensor.as_tensor()->name));
    }

    auto deps_tensor_names = node->tensor.as_tensor()->GetDependTensorNames();
    for (auto &t : deps_tensor_names) {
      auto *n = graph.RetrieveNode(t);
      if (!n) {
        n = graph.RegisterNode(t, new SSANode(t));
      }
      n->Controls(cur_graph_node);
    }
  }
};

}  // namespace

void ComputeInlineExpand(Expr *expr,
                         poly::StageMap stages,
                         std::map<std::string, ir::Tensor> *all_tensor_map) {
  // the inline tensors contained in the expression.
  auto inline_tensors = ir::ir_utils::CollectIRNodes(*expr, [&](const Expr *x) {
    return x->as_tensor() && stages[x->as_tensor()]->inlined();
  });

  // keep inline expand if any inline tensor exists
  // NOTE This is a naive method to greedily expand the inline tensors until
  // none exists, a better way is to create a SSA graph and expand the inline
  // tensors in the reversed dependency order.
  // TODO(Superjomn) Use the SSA graph to improve this.
  while (!inline_tensors.empty()) {
    for (const auto &t : inline_tensors) {
      auto *tensor = t.as_tensor();
      TensorInlineExpandMutator(tensor->name, all_tensor_map, stages)(expr);
    }

    inline_tensors =
        ir::ir_utils::CollectLoadTensors(*expr, [&](const Expr *x) {
          return x->as_tensor() && stages[x->as_tensor()]->inlined();
        });
  }
}

}  // namespace optim
}  // namespace cinn
