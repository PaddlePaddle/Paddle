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

#include "paddle/cinn/optim/update_buffer_axis_pass.h"

#include <unordered_map>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_replace.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

class AnalyzeBufferAxis : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  // Analyze the buffer access inside store
  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Store* store = expr->As<ir::Store>();
    ir::Tensor tensor = store->tensor.as_tensor_ref();
    AnalyzeTensorAxis(store->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  // Analyze the buffer access inside load
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    AnalyzeTensorAxis(load->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::ScheduleBlockRealize* x, Expr* expr) override {
    const ir::ScheduleBlock* schedule_block =
        x->schedule_block.As<ir::ScheduleBlock>();
    const std::vector<ir::Var>& iter_vars = schedule_block->iter_vars;
    const std::vector<ir::Expr>& iter_values = x->iter_values;
    for (int i = 0; i < iter_vars.size(); ++i) {
      iter_var_to_bind_expr_[iter_vars[i]->name] = iter_values[i];
    }
    ir::IRMutator<>::Visit(x, expr);
  }

 private:
  void AnalyzeTensorAxis(const std::vector<Expr>& indices,
                         const ir::Tensor& tensor) {
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }

    const std::string& buffer_name = tensor->buffer->name;
    if (!buffer_name_access_same_index_expr.count(buffer_name)) {
      for (int i = 0; i < indices.size(); ++i) {
        buffer_name_access_same_index_expr[buffer_name][i] =
            GetIndexBindExpr(indices[i]);
      }
      return;
    }

    std::map<int, ir::Expr>& index_expr =
        buffer_name_access_same_index_expr[buffer_name];
    for (int i = 0; i < indices.size(); ++i) {
      if (index_expr.count(i)) {
        if (index_expr[i] == GetIndexBindExpr(indices[i])) {
          continue;
        } else {
          index_expr.erase(i);
        }
      }
    }
    if (index_expr.empty()) {
      buffer_name_access_same_index_expr.erase(buffer_name);
    }
  }

  ir::Expr GetIndexBindExpr(ir::Expr index) {
    if (index.as_var() && iter_var_to_bind_expr_.count(index.as_var()->name)) {
      return iter_var_to_bind_expr_[index.as_var()->name];
    }
    return index;
  }

 public:
  // Stores the buffer names, and its indice where always using same Expr to
  // access For example:
  //   _A[i * 3][j] = ...
  //   ... = _A[k][j]
  // The buffer name _A will map to {1 : j}, where 1 is the indice
  // having same expr j.
  std::unordered_map<std::string, std::map<int, ir::Expr>>
      buffer_name_access_same_index_expr;

 private:
  std::unordered_map<std::string, ir::Expr> iter_var_to_bind_expr_;
};

class ReplaceSameAxisToZero : public ir::IRMutator<> {
 public:
  ReplaceSameAxisToZero(
      const std::unordered_map<std::string, std::map<int, ir::Expr>>&
          buffer_name_access_same_index_expr)
      : buffer_name_access_same_index_expr_(
            buffer_name_access_same_index_expr) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  // Analyze the buffer access inside store
  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Store* store = expr->As<ir::Store>();
    ir::Tensor tensor = store->tensor.as_tensor_ref();
    std::set<int> replace_zero_indice = GetReplaceIndice(tensor);
    for (int r : replace_zero_indice) {
      ir::ir_utils::IrReplace(
          &(store->indices[r]), store->indices[r], ir::Expr(0));
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  // Analyze the buffer access inside load
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    std::set<int> replace_zero_indice = GetReplaceIndice(tensor);
    for (int r : replace_zero_indice) {
      ir::ir_utils::IrReplace(
          &(load->indices[r]), load->indices[r], ir::Expr(0));
    }
    ir::IRMutator<>::Visit(op, expr);
  }

 private:
  std::set<int> GetReplaceIndice(const ir::Tensor& tensor) {
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return {};
    }
    const std::string& buffer_name = tensor->buffer->name;
    std::set<int> ret;
    if (buffer_name_access_same_index_expr_.count(buffer_name)) {
      for (auto p : buffer_name_access_same_index_expr_.at(buffer_name)) {
        ret.insert(p.first);
      }
    }
    return ret;
  }

  const std::unordered_map<std::string, std::map<int, ir::Expr>>&
      buffer_name_access_same_index_expr_;
};

void UpdateBufferAxisPass(ir::Expr* expr) {
  VLOG(6) << "Before UpdateBufferAxisPass, Expr = \n" << *expr;
  AnalyzeBufferAxis analyzer;
  analyzer(expr);
  for (auto p : analyzer.buffer_name_access_same_index_expr) {
    VLOG(6) << "Buffer name: " << p.first;
    for (auto q : p.second) {
      VLOG(6) << "  Index: " << q.first << " Expr: " << q.second;
    }
  }
  ReplaceSameAxisToZero replacer(analyzer.buffer_name_access_same_index_expr);
  replacer(expr);
  VLOG(6) << "After UpdateBufferAxisPass, Expr = \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
