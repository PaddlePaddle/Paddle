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
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
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
    CHECK(expr->As<ir::IfThenElse>());

    const ir::IfThenElse* if_ir = expr->As<ir::IfThenElse>();
    const ir::LT* less_than_ir = if_ir->condition.As<ir::LT>();
    if (less_than_ir != nullptr) {
      std::stringstream oss;
      oss << less_than_ir->a();
      std::string var_name = oss.str();
      if (utils::Startswith(var_name, "blockIdx") ||
          utils::Startswith(var_name, "threadIdx")) {
        var_name_to_extent_[var_name] = less_than_ir->b();
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  // Visit for and collect extent
  void Visit(const ir::For* op, Expr* expr) override {
    CHECK(expr->As<ir::For>());
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
    if (!tensor->buffer.defined() ||
        tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }

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
            int stored_extent = stored_indice_extent[i].as_int32();
            int cur_extent = indice_extent[i].as_int32();
            if (cur_extent > stored_extent) {
              stored_indice_extent[i] = ir::Expr(cur_extent);
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

  // A recursion function to calculate the max index range
  // The index may contain some vars like index = 8 * i / j, where we know the
  // range of i, j, we search all values to get the max index range
  Expr MaxIndexRange(const ir::Expr& index) {
    ir::Expr copy = ir::ir_utils::IRCopy(index);
    std::vector<ir::Expr> vars = ir::ir_utils::CollectIRNodesInOrder(
        copy, [](const ir::Expr* expr) { return expr->As<ir::_Var_>(); });

    // We only use the maximal of var, maximal of Mod operation,
    // which may not be the maximal of index
    // mathmetically, but it works for current CINN.
    //
    // We may add better computation of MaxIndexRange if we need
    for (int i = 0; i < vars.size(); ++i) {
      Expr max_var_value = ir::Sub::Make(
          var_name_to_extent_.at(vars[i].as_var_ref()->name), ir::Expr(1));
      ReplaceModToMax(&copy);
      ReplaceVarWithExpr(&copy, vars[i], max_var_value);
    }
    ir::Expr tmp = ir::Add::Make(copy, ir::Expr(1));
    ir::Expr simplify = common::AutoSimplify(tmp);
    return simplify;
  }

 public:
  std::unordered_map<std::string, std::vector<ir::Expr>>
      buffer_name_to_indice_extent;

 private:
  std::unordered_map<std::string, ir::Expr> var_name_to_extent_;
};

class ResizeBufferFromAnalyzedRange : public ir::IRMutator<> {
 public:
  ResizeBufferFromAnalyzedRange(
      const std::unordered_map<std::string, std::vector<ir::Expr>>&
          buffer_name_to_shape)
      : buffer_name_to_shape_(buffer_name_to_shape) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Store* store = expr->As<ir::Store>();
    ir::Tensor tensor = store->tensor.as_tensor_ref();
    ResizeTensor(&tensor);
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

 private:
  void ResizeTensor(ir::Tensor* tensor_ptr) {
    ir::Buffer buffer = (*tensor_ptr)->buffer;
    if (!buffer.defined() || buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }
    const std::string& buffer_name = buffer->name;
    if (buffer_name_to_shape_.count(buffer_name)) {
      const std::vector<ir::Expr>& analyzed_shape =
          buffer_name_to_shape_.at(buffer_name);
      VLOG(6) << "Replacing shape of tensor " << (*tensor_ptr)->name
              << ", buffer " << buffer->name << ", with shape "
              << analyzed_shape;

      (*tensor_ptr)->shape = analyzed_shape;
      buffer->shape = analyzed_shape;
    }
  }

 private:
  const std::unordered_map<std::string, std::vector<ir::Expr>>&
      buffer_name_to_shape_;
};

void ResizeBufferToMaxVarRange(ir::Expr* expr) {
  VLOG(6) << "Before ResizeBufferToMaxVarRange, Expr = \n" << *expr;
  AnalyzeLoopVarRange analyze_functor;
  analyze_functor(expr);
  ResizeBufferFromAnalyzedRange resize_functor(
      analyze_functor.buffer_name_to_indice_extent);
  resize_functor(expr);
  VLOG(6) << "After ResizeBufferToMaxVarRange, Expr = \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
