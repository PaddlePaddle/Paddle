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
#include "paddle/cinn/optim/resize_buffer.h"

#include <unordered_map>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_mod_to_max.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

class AnalyzeLoopVarRange : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::IfThenElse* op, Expr* expr) override {
    PADDLE_ENFORCE_NOT_NULL(
        expr->As<ir::IfThenElse>(),
        ::common::errors::InvalidArgument(
            "The expression could not be cast to ir::IfThenElse. Please check "
            "the expression type."));

    const ir::IfThenElse* if_ir = expr->As<ir::IfThenElse>();
    const ir::LT* less_than_ir = if_ir->condition.As<ir::LT>();
    if (less_than_ir != nullptr) {
      std::stringstream oss;
      oss << less_than_ir->a();
      std::string var_name = oss.str();
      if (utils::StartsWith(var_name, "blockIdx") ||
          utils::StartsWith(var_name, "threadIdx")) {
        var_name_to_extent_[var_name] = less_than_ir->b();
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  // Visit for and collect extent
  void Visit(const ir::For* op, Expr* expr) override {
    PADDLE_ENFORCE_NOT_NULL(expr->As<ir::For>(),
                            ::common::errors::InvalidArgument(
                                "The expression could not be cast to ir::For. "
                                "Please check the expression type."));
    ir::For* for_ir = expr->As<ir::For>();
    std::string var_name = for_ir->loop_var->name;
    Expr extent = for_ir->extent;
    var_name_to_extent_[var_name] = extent;
    if (for_ir->is_binded()) {
      const ir::BindInfo& bind_info = for_ir->bind_info();
      if (bind_info.valid()) {
        std::string bind_var_str = static_cast<std::string>(bind_info);
        var_name_to_extent_[bind_var_str] = extent;
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  // Analyze the buffer access inside store
  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Store* store = expr->As<ir::Store>();
    ir::Tensor tensor = store->tensor.as_tensor_ref();
    AnalyzeTensorRange(store->indices, tensor);
    AnalyzeBufferSize(store->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  // Analyze the buffer access inside load
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    AnalyzeTensorRange(load->indices, tensor);
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::ScheduleBlockRealize* x, Expr* expr) override {
    const ir::ScheduleBlock* schedule_block =
        x->schedule_block.As<ir::ScheduleBlock>();
    const std::vector<ir::Var>& iter_vars = schedule_block->iter_vars;
    const std::vector<ir::Expr>& iter_values = x->iter_values;
    for (int i = 0; i < iter_vars.size(); ++i) {
      const std::string& var_name = iter_vars[i]->name;
      VLOG(6) << "Analyzing var_name = " << var_name
              << ", expression = " << iter_values[i];
      Expr bind_value = MaxIndexRange(iter_values[i]);

      VLOG(6) << "Get extent of " << var_name
              << ", bind_value = " << bind_value;
      var_name_to_extent_[var_name] = bind_value;
    }
    ir::IRMutator<>::Visit(x, expr);
  }

 private:
  void AnalyzeTensorRange(const std::vector<Expr>& indices,
                          const ir::Tensor& tensor) {
    if (!tensor->buffer.defined()) return;
    if (tensor->buffer->memory_type == ir::MemoryType::Heap) return;

    std::vector<ir::Expr> indice_extent;
    for (int i = 0; i < indices.size(); ++i) {
      Expr simplified_idx_extent = MaxIndexRange(indices[i]);
      indice_extent.push_back(simplified_idx_extent);
    }

    std::string buffer_name = tensor->buffer->name;
    if (buffer_name_to_indice_extent.count(buffer_name)) {
      std::vector<ir::Expr>& stored_indice_extent =
          buffer_name_to_indice_extent[buffer_name];
      if (indice_extent.size() > stored_indice_extent.size()) {
        // multi-dimension access vs single index access, we treat
        // multi-dimension access as better buffer size computation.
        buffer_name_to_indice_extent[buffer_name] = indice_extent;
      } else if (indice_extent.size() == stored_indice_extent.size()) {
        for (int i = 0; i < indice_extent.size(); ++i) {
          if (stored_indice_extent[i].is_constant() &&
              indice_extent[i].is_constant()) {
            int64_t stored_extent = stored_indice_extent[i].as_int64();
            int64_t cur_extent = indice_extent[i].as_int64();
            if (cur_extent > stored_extent) {
              stored_indice_extent[i] = ir::Expr(cur_extent);
              stored_indice_extent[i]->set_type(indice_extent[i].type());
            }
          }
          // if there indice extent is not constant, which means dynamic shape
          // we don't change the value now.
        }
      }
    } else {
      buffer_name_to_indice_extent[buffer_name] = indice_extent;
    }
    VLOG(6) << "buffer_name = " << buffer_name << ", indice_extent = "
            << buffer_name_to_indice_extent[buffer_name];
  }

  void AnalyzeBufferSize(const std::vector<Expr>& indices,
                         const ir::Tensor& tensor) {
    if (!tensor->buffer.defined()) return;
    if (tensor->buffer->memory_type == ir::MemoryType::Heap) return;

    const std::string& buffer_name = tensor->buffer->name;
    buffer_name_to_size[buffer_name] = AnalyzeBufferSize(indices);
    VLOG(6) << "buffer_name = " << buffer_name
            << ", size = " << buffer_name_to_size[buffer_name];
  }

  ir::Expr AnalyzeBufferSize(const std::vector<ir::Expr>& indices) {
    const auto GetIterVarNames =
        [](const std::vector<ir::Expr>& indices) -> std::set<std::string> {
      std::set<std::string> iter_var_names;
      for (const ir::Expr& e : indices) {
        ir::ir_utils::CollectIRNodes(e, [&](const ir::Expr* x) {
          if (x->as_var() && !x->as_var()->is_symbolic_constant) {
            iter_var_names.insert(x->as_var()->name);
          }
          return false;
        });
      }
      return iter_var_names;
    };

    std::set<std::string> iter_var_names = GetIterVarNames(indices);
    ir::Expr size(1);
    for (const std::string& var_name : iter_var_names) {
      PADDLE_ENFORCE_GT(var_name_to_extent_.count(var_name),
                        0,
                        ::common::errors::PreconditionNotMet(
                            "Cannot find the extent of var %s", var_name));
      size = optim::ArithSimplify(size * var_name_to_extent_.at(var_name));
    }

    return size;
  }

  // A recursion function to calculate the max index range
  // The index may contain some vars like index = 8 * i / j, where we know the
  // range of i, j, we search all values to get the max index range
  Expr MaxIndexRange(const ir::Expr& index) {
    ir::Expr copy = ir::ir_utils::IRCopy(index);
    std::vector<ir::Expr> vars = ir::ir_utils::CollectIRNodesInOrder(
        copy, [](const ir::Expr* expr) { return expr->As<ir::_Var_>(); });

    // We only use the maximal of var, maximal of Mod operation,
    // which may not be the maximal of index
    // mathematically, but it works for current CINN.
    //
    // We may add better computation of MaxIndexRange if we need
    for (int i = 0; i < vars.size(); ++i) {
      for (auto kv : var_name_to_extent_) {
        auto var_name = vars[i].as_var_ref()->name;
        if (var_name_to_extent_.count(var_name) != 0) {
          Expr max_var_value = ir::Sub::Make(
              var_name_to_extent_.at(vars[i].as_var_ref()->name), ir::Expr(1));
          ReplaceModToMax(&copy);
          ReplaceVarWithExpr(&copy, vars[i], max_var_value);
        }
      }
    }
    ir::Expr tmp = ir::Add::Make(copy, ir::Expr(1));
    ir::Expr simplified = optim::ArithSimplify(tmp);
    if (simplified.As<ir::Min>()) {
      ir::Expr lhs = simplified.As<ir::Min>()->a();
      ir::Expr rhs = simplified.As<ir::Min>()->b();
      common::cas_intervals_t var_intervals =
          common::CollectVarIntervalsOfExprs({lhs, rhs});
      common::SymbolicExprAnalyzer analyzer(var_intervals);
      if (analyzer.ProveLE(lhs, rhs)) {
        return lhs;
      } else if (analyzer.ProveGE(lhs, rhs)) {
        return rhs;
      }
    }
    return simplified;
  }

 public:
  std::unordered_map<std::string, std::vector<ir::Expr>>
      buffer_name_to_indice_extent;
  std::unordered_map<std::string, ir::Expr> buffer_name_to_size;

 private:
  std::unordered_map<std::string, ir::Expr> var_name_to_extent_;
};

class ResizeBufferFromAnalyzedRange : public ir::IRMutator<> {
 public:
  ResizeBufferFromAnalyzedRange(
      const std::unordered_map<std::string, std::vector<ir::Expr>>&
          buffer_name_to_shape,
      const std::unordered_map<std::string, ir::Expr>& buffer_name_to_size)
      : buffer_name_to_shape_(buffer_name_to_shape),
        buffer_name_to_size_(buffer_name_to_size) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Store* store = expr->As<ir::Store>();
    ir::Tensor tensor = store->tensor.as_tensor_ref();
    ResizeTensor(&tensor);
    ReplaceTensorIndices<ir::Store>(store);
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Load* op, Expr* expr) override {
    auto load = expr->As<ir::Load>();
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::Heap) {
      ir::IRMutator<>::Visit(op, expr);
      return;
    }

    ir::Tensor tensor = load->tensor.as_tensor_ref();
    ResizeTensor(&tensor);

    // For the moment, align the load tensor indices with the tensor shape using
    // the trick method. A better way would be to modify the FlattenLoop
    // Schedule.
    int cnt = load->indices.size() - load->tensor.as_tensor_ref()->shape.size();
    for (int i = 0; i < cnt; i++) {
      load->indices.erase(load->indices.begin());
    }
    ReplaceTensorIndices<ir::Load>(load);
    ir::IRMutator<>::Visit(op, expr);
  }

 private:
  void ResizeTensor(ir::Tensor* tensor_ptr) {
    ir::Buffer buffer = (*tensor_ptr)->buffer;
    if (!buffer.defined()) return;
    if (buffer->memory_type == ir::MemoryType::Heap) return;

    const std::string& buffer_name = buffer->name;
    if (buffer_name_to_shape_.count(buffer_name)) {
      const std::vector<ir::Expr>& analyzed_shape =
          buffer_name_to_shape_.at(buffer_name);
      VLOG(6) << "Replacing shape of tensor " << (*tensor_ptr)->name
              << " with shape " << analyzed_shape;
      (*tensor_ptr)->shape = analyzed_shape;
      buffer->shape = analyzed_shape;
    }
    if (buffer_name_to_size_.count(buffer_name) > 0) {
      const ir::Expr& analyzed_size = buffer_name_to_size_.at(buffer_name);
      VLOG(6) << "Replacing shape of buffer " << buffer->name << " with shape "
              << analyzed_size;
      buffer->shape = {analyzed_size};
    }
  }

  template <typename T>
  void ReplaceTensorIndices(T* op) {
    ir::Tensor tensor = op->tensor.as_tensor_ref();
    ir::Buffer buffer = tensor->buffer;
    if (!buffer.defined()) return;
    if (buffer->memory_type != ir::MemoryType::GPULocal) return;

    VLOG(4) << "replacing index of tensor: " << tensor->name;
    ir::Expr index_expr = op->index();
    std::unordered_map<std::string, ir::Expr> var_name_to_expr;
    ir::ir_utils::CollectIRNodes(index_expr, [&](const ir::Expr* x) {
      const ir::_Var_* var = x->as_var();
      if (var) {
        var_name_to_expr[var->name] = var->Copy();
      }
      return false;
    });
    if (var_name_to_expr.size() != 1) {
      return;
    }

    ir::Expr single_var = var_name_to_expr.begin()->second;
    VLOG(4) << "found single var: " << single_var;
    for (size_t i = 0; i + 1 < op->indices.size(); i++) {
      op->indices[i] = ir::Expr(0);
    }
    op->indices.back() = single_var;
  }

 private:
  const std::unordered_map<std::string, std::vector<ir::Expr>>&
      buffer_name_to_shape_;
  const std::unordered_map<std::string, ir::Expr>& buffer_name_to_size_;
};

void ResizeBufferToMaxVarRange(ir::Expr* expr) {
  VLOG(6) << "Before ResizeBufferToMaxVarRange, Expr = \n" << *expr;
  AnalyzeLoopVarRange analyze_functor;
  analyze_functor(expr);
  ResizeBufferFromAnalyzedRange resize_functor(
      analyze_functor.buffer_name_to_indice_extent,
      analyze_functor.buffer_name_to_size);
  resize_functor(expr);
  VLOG(6) << "After ResizeBufferToMaxVarRange, Expr = \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
