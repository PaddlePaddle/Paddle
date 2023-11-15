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
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

class AnalyzeLoopVarRange : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  // Visit for and collect extent
  void Visit(const ir::For* op, Expr* expr) override {
    CHECK(expr->As<ir::For>());
    ir::For* for_ir = expr->As<ir::For>();
    std::string var_name = for_ir->loop_var->name;
    Expr extent = for_ir->extent;
    if (extent.is_constant()) {
      var_name_to_extent_[var_name] = extent.as_int32();
      if (for_ir->is_binded()) {
        const ir::BindInfo& bind_info = for_ir->bind_info();
        if (bind_info.valid()) {
          std::string bind_var_str = static_cast<std::string>(bind_info);
          var_name_to_extent_[bind_var_str] = extent.as_int32();
        }
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
      int bind_value = MaxIndexRange(iter_values[i]);

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
      int simplified_idx_extent = MaxIndexRange(indices[i]);
      indice_extent.push_back(ir::Expr(simplified_idx_extent));
    }

    std::string buffer_name = tensor->buffer->name;
    if (buffer_name_to_indice_extent.count(buffer_name)) {
      std::vector<ir::Expr>& stored_indice_extent =
          buffer_name_to_indice_extent[buffer_name];
      if (indice_extent.size() > stored_indice_extent.size()) {
        buffer_name_to_indice_extent[buffer_name] = indice_extent;
      } else if (indice_extent.size() == stored_indice_extent.size()) {
        for (int i = 0; i < indice_extent.size(); ++i) {
          int stored_extent = stored_indice_extent[i].as_int32();
          int cur_extent = indice_extent[i].as_int32();
          if (cur_extent > stored_extent) {
            stored_indice_extent[i] = ir::Expr(cur_extent);
          }
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
  int MaxIndexRange(const ir::Expr& index) {
    ir::Expr copy = ir::ir_utils::IRCopy(index);
    std::vector<ir::Expr> vars = ir::ir_utils::CollectIRNodesInOrder(
        copy, [](const ir::Expr* expr) { return expr->As<ir::_Var_>(); });

    int max_range = 1;

    // using recursion funcitons index range.
    std::function<void(ir::Expr, int)> compute_range =
        [&](ir::Expr index, int num_replaced_var) {
          ir::Var var = vars[num_replaced_var].as_var_ref();
          CHECK(var_name_to_extent_.count(var->name))
              << "Index used a loop var " << var->name << " not in loop";
          int extent = var_name_to_extent_.at(var->name);

          for (int idx = extent - 1; idx < extent; ++idx) {
            ir::Expr tmp = ir::ir_utils::IRCopy(index);
            ReplaceVarWithExpr(&tmp, var, Expr(idx));
            ++num_replaced_var;
            if (num_replaced_var >= vars.size()) {
              ir::Expr simplify = common::AutoSimplify(tmp);
              ir::Expr range = common::AutoSimplify(simplify);
              // TODO(zhhsplendid): consider dynamic shape case
              CHECK(range.is_constant())
                  << "Range is not constant when AnalyzeTensorRange";
              max_range = std::max(max_range, range.as_int32() + 1);
            } else {
              compute_range(tmp, num_replaced_var);
            }
            --num_replaced_var;
          }
        };

    if (vars.size()) {
      compute_range(copy, 0);
    }
    return max_range;
  }

 public:
  std::unordered_map<std::string, std::vector<ir::Expr>>
      buffer_name_to_indice_extent;

 private:
  std::unordered_map<std::string, int> var_name_to_extent_;
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

  /*
  void Visit(const ir::Load* op, Expr* expr) override {
    ir::Load* load = expr->As<ir::Load>();
    ir::Tensor tensor = load->tensor.as_tensor_ref();
    ResizeTensor(&tensor);
    ir::IRMutator<>::Visit(op, expr);
  }
  */

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
    std::string buffer_name = buffer->name;
    if (buffer_name_to_shape_.count(buffer->name)) {
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
}

}  // namespace optim
}  // namespace cinn
