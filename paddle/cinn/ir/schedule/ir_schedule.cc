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

#include "paddle/cinn/ir/schedule/ir_schedule.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule_error.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_visitor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/utils/string.h"

PD_DECLARE_int32(cinn_error_message_level);

namespace cinn {
namespace ir {

/**
 * A struct helps to implement Schedule primitives.
 */
class ScheduleImpl {
 public:
  ScheduleImpl() = default;
  explicit ScheduleImpl(const ModuleExpr& module_expr,
                        bool debug_flag = false,
                        utils::ErrorMessageLevel err_msg_level =
                            utils::ErrorMessageLevel::kGeneral)
      : module_expr_(module_expr), debug_flag_(debug_flag) {
    err_msg_level_ = static_cast<utils::ErrorMessageLevel>(
        FLAGS_cinn_error_message_level || static_cast<int>(err_msg_level));
  }
  explicit ScheduleImpl(ModuleExpr&& module_expr)
      : module_expr_(std::move(module_expr)) {}

  //! Set the debug flag.
  void SetDebugFlag(bool debug_flag) { debug_flag_ = debug_flag; }

  //! Get the ModuleExpr stored in ScheduleImpl.
  const ModuleExpr& GetModule() const { return module_expr_; }

  void MergeExprs();

  void SetExprs(const std::vector<Expr>& exprs) {
    module_expr_.SetExprs(exprs);
  }

  bool HasBlock(const std::string& block_name) const;

  std::vector<Expr> GetLoops(const Expr& block) const;
  std::vector<Expr> GetLoops(const std::string& block_name) const;
  std::vector<Expr> GetAllBlocks() const;
  std::vector<Expr> GetChildBlocks(const Expr& expr) const;
  Expr GetBlock(const std::string& block_name) const;
  std::vector<Expr> Split(const Expr& loop, const std::vector<int>& factors);
  std::vector<Expr> SamplePerfectTile(
      utils::LinearRandomEngine::StateType* rand_seed,
      const Expr& loop,
      int n,
      int max_innermost_factor);
  Expr Fuse(const std::vector<Expr>& loops);
  Expr Fuse(const std::string& block_name, const std::vector<int>& loops_index);
  Expr Fuse(const Expr& block, const std::vector<int>& loops_index);
  void ComputeAt(const Expr& block, const Expr& loop, bool keep_unit_loops);
  void SimpleComputeAt(const Expr& block, const Expr& loop);
  void ReverseComputeAt(const Expr& block,
                        const Expr& loop,
                        bool keep_unit_loops);
  Expr GetRootBlock(const Expr& expr) const;
  Expr CacheRead(const Expr& block,
                 int read_buffer_index,
                 const std::string& memory_type);
  Expr CacheWrite(const Expr& block,
                  int write_buffer_index,
                  const std::string& memory_type);
  void SyncThreads(const Expr& ir_node, bool after_node = true);
  void SetBuffer(Expr& block,  // NOLINT
                 const std::string& memory_type,
                 bool fixed = false);
  Expr Reorder(const std::vector<Expr>& loops);
  Expr Reorder(const std::string& block_name,
               const std::vector<int>& loops_index);
  Expr Reorder(const Expr& block, const std::vector<int>& loops_index);
  DeviceAPI GetDeviceAPI() const;
  void MutateForType(const Expr& loop, ForType for_type, int factor = -1);
  void Parallel(const Expr& loop);
  void Vectorize(const Expr& loop, int factor);
  void Unroll(const Expr& loop);
  void ComputeInline(const Expr& schedule_block);
  void ReverseComputeInline(const Expr& schedule_block);
  void Bind(const Expr& loop, const std::string& thread_axis);
  Expr Rfactor(const Expr& rf_loop, int rf_axis);
  Expr AddUnitLoop(const Expr& block) const;
  void Annotate(const Expr& block, const std::string& key, const attr_t& value);
  void Unannotate(Expr& block, const std::string& key);  // NOLINT
  void FlattenLoops(const std::vector<Expr>& loops,
                    const bool force_flat = false);
  void CopyTransformAndLoopInfo(const Expr& block, const Expr& block_target);
  void CopyTransformAndLoopInfo(const std::string& block_name,
                                const std::string& block_target_name);
  Expr SampleCategorical(utils::LinearRandomEngine::StateType* rand_seed,
                         const std::vector<int>& candidates,
                         const std::vector<float>& probs);

 private:
  void Replace(const Expr& src_sref, const Expr& tgt_stmt);

  ModuleExpr module_expr_;
  bool debug_flag_{false};
  utils::ErrorMessageLevel err_msg_level_ = utils::ErrorMessageLevel::kGeneral;
};

/** \brief A macro that guards the beginning of each implementation of schedule
 */
#define CINN_IR_SCHEDULE_BEGIN() try {
/**
 * \brief A macro that pairs with `CINN_IR_SCHEDULE_BEGIN`, handling potential
 * errors and error message printing.
 * @param primitive A string representing the kind of schedule primitive.
 * @param err_msg_level A ScheduleErrorMessageLevel enum, level of error message
 * printing
 */
#define CINN_IR_SCHEDULE_END(err_msg_level)                    \
  }                                                            \
  catch (const utils::ErrorHandler& err_hanlder) {             \
    CINN_THROW(err_hanlder.FormatErrorMessage(err_msg_level)); \
  }

std::vector<Expr> ScheduleImpl::Split(const Expr& loop,
                                      const std::vector<int>& factors) {
  CHECK(loop.As<ir::For>())
      << "Expr param of Split must be For node! Please check.";
  auto* for_node = loop.As<ir::For>();
  CHECK(common::is_zero(for_node->min))
      << "The For node must start with 0! Please check.";
  CHECK(for_node->extent.is_constant())
      << "The For node's extent must be constant! Please check.";
  int tot_extent = for_node->extent.get_constant();

  VLOG(3) << "Try Split loop from (" << for_node->loop_var->name << ", 0, "
          << tot_extent << ") to (" << cinn::utils::Join(factors, ", ")
          << ") at loop:\n"
          << loop;

  std::vector<int> processed_factors;
  CINN_IR_SCHEDULE_BEGIN();
  processed_factors = ValidateFactors(factors, tot_extent, this->module_expr_);
  CINN_IR_SCHEDULE_END(this->err_msg_level_);
  int prod_size = std::accumulate(processed_factors.begin(),
                                  processed_factors.end(),
                                  1,
                                  std::multiplies<int>());
  std::vector<Var> new_loop_vars;
  Expr substitute_value(0);
  for (int i = 0; i < processed_factors.size(); ++i) {
    Var temp_var(common::UniqName(for_node->loop_var->name));
    substitute_value =
        Expr(temp_var) + substitute_value * Expr(processed_factors[i]);
    new_loop_vars.push_back(temp_var);
  }
  substitute_value = common::AutoSimplify(substitute_value);
  Expr new_node = ir::ir_utils::IRCopy(for_node->body);
  ReplaceExpr(&new_node, {for_node->loop_var}, {substitute_value});
  std::vector<Expr> splited_loops;
  splited_loops.resize(processed_factors.size());
  if (tot_extent < prod_size) {
    new_node = IfThenElse::Make(LT::Make(substitute_value, for_node->extent),
                                new_node);
  }
  for (int i = processed_factors.size() - 1; i >= 0; i--) {
    if (!new_node.As<ir::Block>()) new_node = Block::Make({new_node});
    new_node = For::Make(new_loop_vars[i],
                         Expr(0),
                         Expr(processed_factors[i]),
                         for_node->for_type(),
                         for_node->device_api,
                         new_node);
    splited_loops[i] = new_node;
  }

  this->Replace(loop, new_node);
  VLOG(3) << "After Split, ir is:\n" << splited_loops.at(0);
  return splited_loops;
}

Expr ScheduleImpl::Fuse(const std::vector<Expr>& loops) {
  VLOG(3) << "Tring to fuse:\n" << cinn::utils::Join(loops, "\n");
  std::vector<const ir::For*> for_nodes;
  std::vector<Var> loop_vars;
  CHECK(!loops.empty())
      << "The loops param of Fuse should not be empty! Please check.";

  for (const Expr& it_loop : loops) {
    CHECK(it_loop.As<ir::For>())
        << "Expr param of Fuse must be For node! Please check.";
    if (!for_nodes.empty()) {
      CHECK(for_nodes.back()->body.As<ir::Block>())
          << "The body of for node is not Block!";
      CHECK_EQ(for_nodes.back()->body.As<ir::Block>()->stmts.size(), 1U)
          << "The Block'size of for node is not 1!";
      CHECK_EQ(for_nodes.back()->body.As<ir::Block>()->stmts[0], it_loop)
          << "The For nodes in loops param of Fuse must be adjacent! Please "
             "check.";
    }
    for_nodes.push_back(it_loop.As<ir::For>());
    loop_vars.push_back(it_loop.As<ir::For>()->loop_var);
  }
  std::string suffix;
  suffix = for_nodes[0]->loop_var->name;
  int loops_number = for_nodes.size();
  for (int i = 1; i < loops_number; ++i) {
    suffix += "_" + for_nodes[i]->loop_var->name;
  }
  suffix += "_fused";
  Var fused_var(suffix);
  std::vector<Expr> substitute_value;
  substitute_value.resize(loops_number);
  Expr fused_expr(fused_var);
  for (int i = loops_number - 1; i > 0; i--) {
    substitute_value[i] = Mod::Make(fused_expr, for_nodes[i]->extent);
    fused_expr = Div::Make(fused_expr, for_nodes[i]->extent);
  }
  substitute_value[0] = fused_expr;

  Expr fused_body = ir::ir_utils::IRCopy(for_nodes.back()->body);
  ReplaceExpr(&fused_body, loop_vars, substitute_value);
  optim::Simplify(&fused_body);
  Expr fused_extent(1);
  for (int i = 0; i < loops_number; ++i) {
    fused_extent = fused_extent * for_nodes[i]->extent;
  }
  fused_extent = common::AutoSimplify(fused_extent);

  if (!fused_body.As<ir::Block>()) fused_body = Block::Make({fused_body});
  Expr new_stmt = For::Make(fused_var,
                            Expr(0),
                            fused_extent,
                            for_nodes[0]->for_type(),
                            for_nodes[0]->device_api,
                            fused_body);
  this->Replace(loops[0], new_stmt);

  VLOG(3) << "After fuse, ir is:\n" << new_stmt;
  return new_stmt;
}

Expr ScheduleImpl::Fuse(const std::string& block_name,
                        const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0)
      CHECK_EQ(loops_index[i - 1] + 1, loops_index[i])
          << "Loops index in Fuse shoule be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

Expr ScheduleImpl::Fuse(const Expr& block,
                        const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i = 0; i < loops_index.size(); ++i) {
    if (i > 0)
      CHECK_EQ(loops_index[i - 1] + 1, loops_index[i])
          << "Loops index in Fuse shoule be continuous!";
  }
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Fuse should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Fuse should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Fuse(loops_expr);
}

void ScheduleImpl::MutateForType(const Expr& loop,
                                 ForType for_type,
                                 int factor) {
  auto* for_node = loop.As<ir::For>();
  CHECK(for_node) << "loop param must be For node! Please check.";
  CHECK(for_node->is_serial())
      << "loop is not serial, current forloop type is "
      << static_cast<int>(for_node->for_type()) << ", and it cannot become "
      << static_cast<int>(for_type);
  auto loop_copy = ir::ir_utils::IRCopy(loop);
  auto* new_for_node = loop_copy.As<ir::For>();
  CHECK(new_for_node);
  new_for_node->set_for_type(for_type);
  if (new_for_node->is_vectorized()) {
    VectorizeInfo vec_info(0, factor);
    new_for_node->set_vectorize_info(vec_info);
  } else if (new_for_node->is_binded()) {
    BindInfo bind_info(for_type, factor, DeviceAPI::GPU);
    new_for_node->set_bind_info(bind_info);
  }
  this->Replace(loop, loop_copy);
}

void ScheduleImpl::Parallel(const Expr& loop) {
  MutateForType(loop, ForType::Parallel);
}

void ScheduleImpl::Vectorize(const Expr& loop, int factor) {
  CHECK_GT(factor, 0) << "vectorize factor should be more than 0";
  MutateForType(loop, ForType::Vectorized, factor);
}

void ScheduleImpl::Unroll(const Expr& loop) {
  MutateForType(loop, ForType::Unrolled);
}

void ScheduleImpl::Bind(const Expr& loop, const std::string& thread_axis) {
  static std::set<std::string> thread_axes = {"blockIdx.x",
                                              "blockIdx.y",
                                              "blockIdx.z",
                                              "threadIdx.x",
                                              "threadIdx.y",
                                              "threadIdx.z"};
  CHECK(thread_axes.count(thread_axis))
      << "thread_axis " << thread_axis << " is not supported";
  int offset = thread_axis.back() - 'x';
  if (thread_axis[0] == 'b') {
    MutateForType(loop, ForType::GPUBlock, offset);
  } else {
    MutateForType(loop, ForType::GPUThread, offset);
  }
}

// The struct used to mutate new rfactor forloop and its' schedule block.
struct RfMutator : public ir::IRMutator<> {
 public:
  RfMutator(const Expr& rf_loop, const int& rf_axis)
      : rf_loop_(rf_loop), rf_axis_(rf_axis) {}
  void operator()(Expr* expr) {
    auto* rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    old_rf_loop_var_ = rf_for->loop_var;
    new_rf_loop_var_ = Var("rf_" + old_rf_loop_var_->name);
    IRMutator::Visit(expr, expr);
  }

  Tensor GetNewRfTensor() { return new_rf_tensor_; }

  void Visit(const ScheduleBlockRealize* op, Expr* expr) override {
    // modify iter_vars and iter_values
    auto* node = expr->As<ScheduleBlockRealize>();
    CHECK(node);
    auto* schedule_block = node->schedule_block.As<ScheduleBlock>();
    CHECK(schedule_block);
    old_output_name_ = schedule_block->name;
    find_tensor_ = false;
    auto& block_vars = schedule_block->iter_vars;
    auto& iter_values = node->iter_values;
    CHECK(old_rf_loop_var_.defined());
    CHECK(new_rf_loop_var_.defined());
    CHECK_EQ(iter_values.size(), block_vars.size());
    int rf_index = -1;
    for (int i = 0; i < iter_values.size(); ++i) {
      // substitute the old rfactor loop var to new rfactor loop var
      if (ContainVar({iter_values[i]}, old_rf_loop_var_->name)) {
        CHECK_EQ(rf_index, -1)
            << "only one block var can bind the rfactor loop var";
        CHECK(iter_values[i].As<_Var_>())
            << "rfactor loop var not support composite bindings";
        rf_index = i;
        optim::ReplaceVarWithExpr(
            &iter_values[i], old_rf_loop_var_, new_rf_loop_var_);
        new_rf_itervar_ = block_vars[i];
      }
    }
    // create new rfactor block var if not exist
    if (rf_index == -1) {
      new_rf_itervar_ =
          Var(cinn::UniqName("i" + std::to_string(block_vars.size())));
      iter_values.push_back(new_rf_loop_var_);
      block_vars.push_back(new_rf_itervar_);
    }
    IRMutator::Visit(&node->schedule_block, &node->schedule_block);
    CHECK(find_tensor_)
        << "not find the store tensor with the schedule block name "
        << old_output_name_;
    schedule_block->name = "rf_" + old_output_name_;
  }

  void Visit(const Load* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Load>();
    CHECK(node);
    auto* tensor = node->tensor.As<_Tensor_>();
    CHECK(tensor);
    if (tensor->name == "rf_" + old_output_name_) {
      int size = node->indices.size();
      CHECK_LE(rf_axis_, size)
          << "rf_axis should not be greater than indice size " << size;
      CHECK(new_rf_itervar_.defined());
      CHECK(!ContainVar(node->indices, new_rf_itervar_->name))
          << "original output tensor " << old_output_name_
          << " should not have the new rfactor index " << new_rf_itervar_;
      node->indices.insert(node->indices.begin() + rf_axis_, new_rf_itervar_);
    }
  }

  void Visit(const Store* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Store>();
    CHECK(node);
    auto* tensor = node->tensor.As<_Tensor_>();
    CHECK(tensor);
    if (tensor->name == old_output_name_) {
      find_tensor_ = true;
      tensor->name = "rf_" + tensor->name;
      int size = node->indices.size();
      CHECK_LE(rf_axis_, size)
          << "rf_axis should not be greater than indice size " << size;
      CHECK(!ContainVar(node->indices, new_rf_itervar_->name))
          << "original output tensor " << old_output_name_
          << " should not have the new rfactor index " << new_rf_itervar_;
      node->indices.insert(node->indices.begin() + rf_axis_, new_rf_itervar_);
      auto* rf_for = rf_loop_.As<For>();
      CHECK(rf_for);
      CHECK(is_zero(rf_for->min)) << "rfactor loop's min should be zero";
      auto extent = common::AutoSimplify(rf_for->extent);
      auto& shape = tensor->shape;
      auto& domain = tensor->domain;
      CHECK_LE(rf_axis_, shape.size())
          << "rf_axis should not be greater than tensor shape size "
          << shape.size();
      CHECK_LE(rf_axis_, domain.size())
          << "rf_axis should not be greater than tensor domain size "
          << domain.size();
      shape.insert(shape.begin() + rf_axis_, extent);
      domain.insert(domain.begin() + rf_axis_, extent);
      if (tensor->buffer.defined()) {
        if (tensor->buffer->name.find_first_of("rf") == std::string::npos) {
          tensor->buffer->name = "rf_" + tensor->buffer->name;
          tensor->buffer->shape = shape;
        }
      }
      new_rf_tensor_ = Tensor(tensor);
    }
    IRMutator::Visit(&node->value, &node->value);
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<For>();
    CHECK(node);
    depth++;
    auto* rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    // erase the original rfactor forloop
    if (node->loop_var->name == old_rf_loop_var_->name) {
      auto body = node->body.As<Block>();
      if (body && body->stmts.size() == 1) {
        *expr = body->stmts[0];
      } else {
        *expr = node->body;
      }
      IRMutator::Visit(expr, expr);
    } else {
      IRMutator::Visit(&node->body, &node->body);
    }
    if (rf_axis_ == 0 && depth == rf_axis_) {
      // insert new rfactor forloop in the rf_axis as serial loop
      *expr = For::Make(new_rf_loop_var_,
                        rf_for->min,
                        rf_for->extent,
                        ForType::Serial,
                        rf_for->device_api,
                        Block::Make({*expr}));
    } else if (depth == rf_axis_ - 1) {
      // insert new rfactor forloop in the rf_axis as serial loop
      node->body = Block::Make({For::Make(new_rf_loop_var_,
                                          rf_for->min,
                                          rf_for->extent,
                                          ForType::Serial,
                                          rf_for->device_api,
                                          node->body)});
    }
    depth--;
  }

 private:
  Expr rf_loop_;
  Var old_rf_loop_var_;
  Var new_rf_loop_var_;
  int rf_axis_;
  int depth = -1;
  bool find_tensor_ = false;
  std::string old_output_name_;
  Var new_rf_itervar_;
  Tensor new_rf_tensor_;
};

// The struct used to mutate final write-back forloop and schedule block.
struct FinalMutator : public ir::IRMutator<> {
 public:
  FinalMutator(const Expr& rf_loop,
               const int& rf_axis,
               const Tensor& new_rf_tensor)
      : rf_loop_(rf_loop), rf_axis_(rf_axis), new_rf_tensor_(new_rf_tensor) {}
  void operator()(Expr* expr) {
    auto* rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    old_rf_loop_var_ = rf_for->loop_var;
    IRMutator::Visit(expr, expr);
  }

  void Visit(const ScheduleBlockRealize* op, Expr* expr) override {
    auto* node = expr->As<ScheduleBlockRealize>();
    CHECK(node);
    auto* schedule_block = node->schedule_block.As<ScheduleBlock>();
    CHECK(schedule_block);
    auto& iter_vars = schedule_block->iter_vars;
    auto& iter_values = node->iter_values;
    output_name_ = schedule_block->name;
    visit_init_block_ = output_name_.rfind("_init") != std::string::npos;
    if (!visit_init_block_) {
      for (int i = 0; i < iter_values.size(); ++i) {
        if (ContainVar({iter_values[i]}, old_rf_loop_var_->name)) {
          // record the rfactor loop var's block var
          CHECK(iter_values[i].As<_Var_>())
              << "not support complex reduce bindings: " << iter_values[i];
          old_rf_iter_var_ = iter_vars[i];
          break;
        }
      }
    }
    IRMutator::Visit(&node->schedule_block, &node->schedule_block);
    // modify iter_vars and iter_values, erase other reduce block vars and
    // values
    for (auto it = iter_values.begin(); it != iter_values.end(); ++it) {
      for (auto erase_var : erase_reduce_loopvars_) {
        if (ContainVar({*it}, erase_var)) {
          CHECK((*it).As<_Var_>())
              << "not support complex reduce bindings: " << *it;
          iter_vars.erase(it - iter_values.begin() + iter_vars.begin());
          iter_values.erase(it);
          --it;
          break;
        }
      }
    }
  }

  // currently only support reduce_sum, reduce_mul, reduce_min and reduce_max
  void Visit(const Add* op, Expr* expr) override {
    auto* node = expr->As<Add>();
    CHECK(node);
    auto& oper_b = node->b();
    oper_b = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Mul* op, Expr* expr) override {
    auto* node = expr->As<Mul>();
    CHECK(node);
    auto& oper_b = node->b();
    oper_b = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Min* op, Expr* expr) override {
    auto* node = expr->As<Min>();
    CHECK(node);
    auto& oper_b = node->b();
    oper_b = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Max* op, Expr* expr) override {
    auto* node = expr->As<Max>();
    CHECK(node);
    auto& oper_b = node->b();
    oper_b = Load::Make(new_rf_tensor_, new_rf_indice_);
  }

  void Visit(const Store* op, Expr* expr) override {
    // insert the new rfactor indice if not exist
    auto* node = expr->As<Store>();
    CHECK(node);
    auto* tensor = node->tensor.As<_Tensor_>();
    CHECK(tensor);
    CHECK_EQ(tensor->name, output_name_)
        << "store name should be same with the schedule block name";
    if (!visit_init_block_) {
      new_rf_indice_ = node->indices;
      CHECK_LE(rf_axis_, new_rf_indice_.size())
          << "rf_axis_ should not be greater than tensor indice size "
          << new_rf_indice_.size();
      CHECK(old_rf_iter_var_.defined());
      new_rf_indice_.insert(new_rf_indice_.begin() + rf_axis_,
                            old_rf_iter_var_);
      IRMutator::Visit(&node->value, &node->value);
    }
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<For>();
    CHECK(node);
    auto* rf_for = rf_loop_.As<For>();
    // erase the reduce forloops after the init block except the rfactor loop
    if (visit_init_block_ && node->loop_var->name != old_rf_loop_var_->name) {
      erase_reduce_loopvars_.insert(node->loop_var->name);
      auto body = node->body.As<Block>();
      if (body && body->stmts.size() == 1) {
        *expr = body->stmts[0];
      } else {
        *expr = node->body;
      }
      IRMutator::Visit(expr, expr);
    } else {
      IRMutator::Visit(&node->body, &node->body);
    }
  }

 private:
  Expr rf_loop_;
  int rf_axis_;
  Var old_rf_loop_var_;
  Var old_rf_iter_var_;
  std::string output_name_;
  // collect reduce loop vars except rfactor loop var
  std::set<std::string> erase_reduce_loopvars_;
  bool visit_init_block_ = false;
  Tensor new_rf_tensor_;
  std::vector<Expr> new_rf_indice_;
};

// The struct used to create all stmts after rfactor transformation.
struct RfCreater : public ir::IRMutator<> {
 public:
  RfCreater(const Expr& root, const Expr& rf_loop, const int& rf_axis)
      : root_(root), rf_loop_(rf_loop), rf_axis_(rf_axis) {}
  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  Expr CreateRfAllStmts() {
    auto root_realize = root_.As<ScheduleBlockRealize>();
    CHECK(root_realize);
    auto root_block = root_realize->schedule_block.As<ScheduleBlock>();
    CHECK(root_block);
    Expr root_loop = ir::ir_utils::IRCopy(root_block->body);
    if (auto block = root_loop.As<Block>()) {
      CHECK_EQ(block->stmts.size(), 1U)
          << "rfactor root should only have one block stmt";
      root_loop = block->stmts[0];
    }
    auto* root_for = root_loop.As<For>();
    CHECK(root_for);
    auto rf_for = rf_loop_.As<For>();
    CHECK(rf_for);
    // create new rfactor forloops
    Expr new_rf_forloop = ir::ir_utils::IRCopy(root_loop);
    RfMutator rf_mutator(rf_loop_, rf_axis_);
    rf_mutator(&new_rf_forloop);
    VLOG(3) << "After RfMutator, new rf_forloop is\n" << new_rf_forloop;
    auto new_rf_tensor = rf_mutator.GetNewRfTensor();
    // create final write-back forloops
    Expr final_forloop = ir::ir_utils::IRCopy(root_loop);
    FinalMutator final_mutator(rf_loop_, rf_axis_, new_rf_tensor);
    final_mutator(&final_forloop);
    VLOG(3) << "After FinalMuator, final write-back forloop is\n"
            << final_forloop;
    // combine the new created rfactor forloops with the final write-back
    // forloops and replace
    root_block->body = Block::Make({new_rf_forloop, final_forloop});
    return new_rf_tensor;
  }

  Expr root_;
  Expr rf_loop_;
  int rf_axis_;
};

Expr ScheduleImpl::Rfactor(const Expr& rf_loop, int rf_axis) {
  CHECKRfactorValidation(rf_loop, rf_axis);
  // get root ScheduleBlockRealize
  Expr root = GetRootBlock(rf_loop);
  // create all stmts after rfactor transformation
  RfCreater rf_create(root, rf_loop, rf_axis);
  // return new created rfactor tensor
  return rf_create.CreateRfAllStmts();
}

struct CacheReadRewriter : public ir::IRMutator<> {
 public:
  static Expr Rewrite(const Expr& root, CacheBlockInfo* info) {
    CacheReadRewriter rewriter(root, info);
    Expr new_root = ir::ir_utils::IRCopy(root);
    rewriter(&new_root);
    return new_root;
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit CacheReadRewriter(const Expr& root, CacheBlockInfo* info)
      : root_(root), info_(info) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    if (*op == info_->loc_block) {
      IRMutator::Visit(expr, op);
      op->As<Block>()->stmts.insert(
          op->As<Block>()->stmts.begin() + info_->loc_pos, info_->cache_block);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    if (expr->tensor == Expr(info_->read_tensor)) {
      IRMutator::Visit(expr, op);
      op->As<Load>()->tensor = Expr(info_->write_tensor);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

 private:
  /*! \brief The parent scope of the insertion */
  const Expr& root_;
  /*! \brief The info for inserting cache stage */
  CacheBlockInfo* info_;
};

struct CacheWriteRewriter : public ir::IRMutator<> {
 public:
  static Expr Rewrite(const Expr& root, CacheBlockInfo* info) {
    CacheWriteRewriter rewriter(root, info);
    Expr new_root = ir::ir_utils::IRCopy(root);
    rewriter.mutate_cache_block = true;
    rewriter(&info->cache_block);
    rewriter.mutate_cache_block = false;
    rewriter(&new_root);
    auto find_tensor = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_root,
        [&](const Expr* x) {
          return x->As<Store>() &&
                 (x->As<Store>()->tensor == Expr(info->read_tensor));
        },
        true);
    if (!find_tensor.empty()) {
      auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
          (*find_tensor.begin()), [&](const Expr* x) {
            return x->As<Load>() &&
                   (x->As<Load>()->tensor == Expr(info->write_tensor));
          });
      for (auto load_ir : find_store) {
        load_ir.As<Load>()->tensor = Expr(info->read_tensor);
      }
    }
    return new_root;
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit CacheWriteRewriter(const Expr& root, CacheBlockInfo* info)
      : root_(root), info_(info) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    if (*op == info_->loc_block) {
      IRMutator::Visit(expr, op);
      op->As<Block>()->stmts.insert(
          op->As<Block>()->stmts.begin() + info_->loc_pos, info_->cache_block);
    } else {
      IRMutator::Visit(expr, op);
    }
  }

  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (op->As<ScheduleBlock>()->name == info_->write_tensor->name) {
      op->As<ScheduleBlock>()->name = info_->read_tensor->name;
    } else if (op->As<ScheduleBlock>()->name == info_->read_tensor->name) {
      op->As<ScheduleBlock>()->name = info_->write_tensor->name;
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
    if (op->As<Load>()->tensor == Expr(info_->write_tensor) &&
        mutate_cache_block) {
      op->As<Load>()->tensor = Expr(info_->read_tensor);
    } else if (op->As<Load>()->tensor == Expr(info_->read_tensor) &&
               mutate_cache_block) {
      op->As<Load>()->tensor = Expr(info_->write_tensor);
    }
  }

  void Visit(const ir::Store* expr, Expr* op) override {
    IRMutator::Visit(expr, op);
    if (op->As<Store>()->tensor == Expr(info_->write_tensor)) {
      op->As<Store>()->tensor = Expr(info_->read_tensor);
    } else if (op->As<Store>()->tensor == Expr(info_->read_tensor) &&
               mutate_cache_block) {
      op->As<Store>()->tensor = Expr(info_->write_tensor);
    }
  }

 private:
  /*! \brief The parent scope of the insertion */
  const Expr& root_;
  /*! \brief The info for inserting cache stage */
  CacheBlockInfo* info_;
  /*! \brief Are we mutating the cache tensor's block */
  bool mutate_cache_block{true};
};

//! Visit all ScheduleBlock and change its body to ir::Block if it is not.
struct ChangeBodyToBlock : public ir::IRMutator<> {
 public:
  static void Change(Expr* expr) {
    ChangeBodyToBlock mutator;
    mutator(expr);
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (!op->As<ScheduleBlock>()->body.As<Block>()) {
      op->As<ScheduleBlock>()->body =
          Block::Make({op->As<ScheduleBlock>()->body});
    }
    IRMutator::Visit(expr, op);
  }
};

DeviceAPI ScheduleImpl::GetDeviceAPI() const {
  auto exprs = this->GetModule().GetExprs();
  auto find_for_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
      exprs.front(), [&](const Expr* x) { return x->As<ir::For>(); }, true);
  CHECK(!find_for_nodes.empty());
  return (*find_for_nodes.begin()).As<ir::For>()->device_api;
}

Expr ScheduleImpl::CacheRead(const Expr& block,
                             int read_tensor_index,
                             const std::string& memory_type) {
  CHECK(block.As<ScheduleBlockRealize>());
  auto root = GetRootBlock(block);
  ChangeBodyToBlock::Change(&root);
  Expr read_expr = GetNthAccessExpr(block, read_tensor_index, false);
  CHECK(read_expr.As<ir::Load>());
  auto tensor_indices = read_expr.As<ir::Load>()->indices;
  CacheBlockInfo info;
  info.read_tensor = read_expr.As<ir::Load>()->tensor.as_tensor_ref();
  info.write_tensor = MakeCacheTensor(info.read_tensor, memory_type);
  info.alloc = info.write_tensor;

  auto read_ranges =
      CalculateTensorRegions(block, tensor_indices, info.read_tensor, root);
  auto new_block =
      MakeCacheBlock(read_ranges, &info, memory_type, this->GetDeviceAPI());
  FindInsertionPoint(root, &info, false);
  auto new_root = CacheReadRewriter::Rewrite(root, &info);
  this->Replace(
      root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body,
      new_root.As<ScheduleBlockRealize>()
          ->schedule_block.As<ScheduleBlock>()
          ->body);
  return new_block;
}

Expr ScheduleImpl::CacheWrite(const Expr& block,
                              int write_buffer_index,
                              const std::string& memory_type) {
  CHECK(block.As<ScheduleBlockRealize>());
  auto root = GetRootBlock(block);
  ChangeBodyToBlock::Change(&root);
  Expr write_expr = GetNthAccessExpr(block, write_buffer_index, true);
  CHECK(write_expr.As<ir::Store>());
  Tensor write_tensor = write_expr.As<ir::Store>()->tensor.as_tensor_ref();
  auto tensor_indices = write_expr.As<ir::Store>()->indices;
  CacheBlockInfo info;
  info.read_tensor = MakeCacheTensor(write_tensor, memory_type);
  info.write_tensor = write_tensor;
  info.alloc = info.read_tensor;
  auto write_ranges =
      CalculateTensorRegions(block, tensor_indices, info.write_tensor, root);
  auto new_block =
      MakeCacheBlock(write_ranges, &info, memory_type, this->GetDeviceAPI());
  FindInsertionPoint(root, &info, true);

  auto new_root = CacheWriteRewriter::Rewrite(root, &info);
  this->Replace(
      root.As<ScheduleBlockRealize>()->schedule_block.As<ScheduleBlock>()->body,
      new_root.As<ScheduleBlockRealize>()
          ->schedule_block.As<ScheduleBlock>()
          ->body);

  auto find_cache_block = ir::ir_utils::CollectIRNodesWithoutTensor(
      root,
      [&](const Expr* x) {
        return x->As<ir::ScheduleBlockRealize>() &&
               !x->As<ir::ScheduleBlockRealize>()->iter_values.empty() &&
               GetTensor(*x)->name == info.read_tensor->name;
      },
      true);

  CHECK(info.write_tensor->buffer.defined());

  // Replace buffer
  auto all_tensors =
      ir::ir_utils::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
        return x->as_tensor() && x->as_tensor()->buffer.defined();
      });

  for (auto i : all_tensors) {
    if (i.as_tensor()->name != info.write_tensor->name &&
        i.as_tensor()->buffer.defined() &&
        i.as_tensor()->buffer->name == info.write_tensor->buffer->name) {
      i.as_tensor()->Bind(info.read_tensor->buffer);
    }
  }

  CHECK_EQ(find_cache_block.size(), 1U);

  return *find_cache_block.begin();
}

struct InsertExpr : public ir::IRMutator<> {
 public:
  static void Insert(const Expr& ir_node,
                     const Expr& insert_node,
                     bool after_node,
                     Expr* expr) {
    InsertExpr mutator(ir_node, insert_node, after_node);
    mutator(expr);
  }

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  explicit InsertExpr(const Expr& ir_node,
                      const Expr& insert_node,
                      bool after_node)
      : ir_node_(ir_node), insert_node_(insert_node), after_node_(after_node) {}

  void Visit(const ir::Block* expr, Expr* op) override {
    for (int i = 0; i < expr->stmts.size(); i++) {
      if (expr->stmts[i] == ir_node_) {
        if (after_node_) {
          op->As<ir::Block>()->stmts.insert(
              op->As<ir::Block>()->stmts.begin() + i + 1, insert_node_);
        } else {
          op->As<ir::Block>()->stmts.insert(
              op->As<ir::Block>()->stmts.begin() + i, insert_node_);
        }
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    if (expr->body == ir_node_) {
      if (after_node_)
        op->As<ir::For>()->body =
            ir::Block::Make({op->As<ir::For>()->body, insert_node_});
      else
        op->As<ir::For>()->body =
            ir::Block::Make({insert_node_, op->As<ir::For>()->body});
      return;
    }
    IRMutator::Visit(expr, op);
  }

 private:
  const Expr& ir_node_;
  const Expr& insert_node_;
  bool after_node_;
};

void ScheduleImpl::SyncThreads(const Expr& ir_node, bool after_node) {
  CHECK(ir_node.As<ScheduleBlockRealize>() || ir_node.As<ir::For>());
  auto root = GetRootBlock(ir_node);
  ChangeBodyToBlock::Change(&root);
  Expr sync_threads = runtime::IntrinsicCall(Void(), "__syncthreads", {});
  InsertExpr::Insert(ir_node, sync_threads, after_node, &root);
  return;
}

/**
 * Replace a For node to another For node.
 * @param src_sref The For node to be changed.
 * @param tgt_stmt The For node we want.
 */
void ScheduleImpl::Replace(const Expr& src_sref, const Expr& tgt_stmt) {
  CHECK(src_sref.As<ir::For>() || src_sref.As<ir::Block>() ||
        src_sref.As<ir::ScheduleBlockRealize>());
  CHECK(tgt_stmt.As<ir::For>() || tgt_stmt.As<ir::Block>() ||
        tgt_stmt.As<ir::ScheduleBlockRealize>());
  if (src_sref == tgt_stmt) {
    return;
  }
  struct ForLoopMutator : public ir::IRMutator<> {
    ForLoopMutator(const Expr& source, const Expr& target)
        : source_(source), target_(target) {}

    void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::For* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::ScheduleBlockRealize* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    void Visit(const ir::Block* op, Expr* expr) override {
      if (*expr == source_) {
        *expr = target_;
        return;
      }
      ir::IRMutator<>::Visit(op, expr);
    }

    const Expr& source_;
    const Expr& target_;
  };
  auto exprs = module_expr_.GetExprs();
  ForLoopMutator mutator(src_sref, tgt_stmt);
  for (auto& i : exprs) {
    mutator(&i);
  }
}

Expr ScheduleImpl::Reorder(const std::vector<Expr>& loops) {
  if (loops.size() <= 1) {
    return Expr{nullptr};
  }
  VLOG(4) << "Before Reorder, ir is:\n" << loops[0];

  std::set<Expr, CompExpr> loop_set = CollectLoopsToSet(loops);
  auto boundary = GetBoundaryOfReorderRange(loop_set);
  Expr top = boundary.first;
  Expr bottom = boundary.second;
  std::vector<Expr> chain = GetLoopsInRange(top, bottom);
  std::vector<Expr> if_nodes = GetIfThenElseInRange(top, bottom);
  Expr new_loop = ConstructNewLoopChain(chain, loops, loop_set, if_nodes);
  this->Replace(top, new_loop);

  VLOG(4) << "After Reorder, ir is:\n" << new_loop;
  return new_loop;
}

Expr ScheduleImpl::Reorder(const std::string& block_name,
                           const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Reorder should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Reorder should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Reorder(loops_expr);
}

Expr ScheduleImpl::Reorder(const Expr& block,
                           const std::vector<int>& loops_index) {
  std::vector<Expr> all_loops = this->GetLoops(block);
  std::vector<Expr> loops_expr;
  loops_expr.reserve(loops_index.size());
  for (int i : loops_index) {
    CHECK_LT(i, (int)all_loops.size())
        << "The loop index in Reorder should be less than total loop's number.";
    CHECK_GE(i, 0) << "The loop index in Reorder should be >= 0.";
    loops_expr.emplace_back(all_loops[i]);
  }
  return this->Reorder(loops_expr);
}

Expr ScheduleImpl::GetRootBlock(const Expr& expr) const {
  auto exprs = this->GetModule().GetExprs();
  for (auto& it_expr : exprs) {
    auto find_expr = ir::ir_utils::CollectIRNodesWithoutTensor(
        it_expr,
        [&](const Expr* x) {
          return x->node_type() == expr.node_type() && *x == expr;
        },
        true);
    if (!find_expr.empty()) {
      CHECK(it_expr.As<ir::Block>());
      CHECK_EQ(it_expr.As<ir::Block>()->stmts.size(), 1U);
      CHECK(it_expr.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>());
      return it_expr.As<ir::Block>()->stmts[0];
    }
  }
  LOG(FATAL) << "Didn't find expr \n"
             << expr << "in ScheduleImpl:\n"
             << exprs[0];
}

// The struct used to reconstruct the new For node to replace the old For node.
struct LoopReconstructor : public ir::IRMutator<> {
 public:
  explicit LoopReconstructor(const Expr& root,
                             const Expr& block,
                             const Expr& loop)
      : root_(root), block_(block), loop_(loop) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

  /* \param inserted_pos The position index of the new_loop_ body `stmts` to be
   * inserted:
   *        - `index = -1` means inserted into the tail
   *        - otherwise, it should be a index between [0, stmts size)
   */
  std::string MakeNewLoop(const std::vector<IterRange>& iter_ranges,
                          bool keep_unit_loops,
                          int inserted_pos = -1) {
    int n_iters = iter_ranges.size();
    std::vector<Var> loop_vars;
    std::vector<Expr> loop_extents;
    std::vector<Expr> iter_values;
    loop_vars.reserve(n_iters);
    loop_extents.reserve(n_iters);
    iter_values.reserve(n_iters);
    std::vector<std::string> new_var_names;
    for (int i = 0; i < n_iters; ++i) {
      const auto& range = iter_ranges[i];
      if (keep_unit_loops || range.extent != Expr(1)) {
        std::string var_name =
            common::UniqName("ax" + std::to_string(loop_vars.size()));
        new_var_names.push_back(var_name);
        Var var(var_name, Int(32));
        loop_vars.push_back(var);
        loop_extents.push_back(range.extent);
        iter_values.push_back(common::AutoSimplify(range.min) + var);
      } else {
        iter_values.push_back(common::AutoSimplify(range.min));
      }
    }
    auto schedule_block_node =
        block_.As<ir::ScheduleBlockRealize>()->schedule_block;
    new_block_ = ScheduleBlockRealize::Make(std::move(iter_values),
                                            std::move(schedule_block_node));
    Expr loop_body = new_block_;
    for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
      auto loop_var = loop_vars[i];
      auto loop_extent = loop_extents[i];
      if (!loop_body.As<ir::Block>()) loop_body = Block::Make({loop_body});
      loop_body = For::Make(loop_var,
                            Expr(0),
                            loop_extent,
                            ForType::Serial,
                            loop_.As<ir::For>()->device_api,
                            std::move(loop_body));
    }
    new_loop_ = ir::ir_utils::IRCopy(loop_);

    // Replace the copied Tensor object with the original Tensor object,
    // to ensure that the same Tensor in a AST is the same object.
    std::unordered_map<std::string, ir::Expr> tensors_map;
    ir::ir_utils::CollectIRNodesWithoutTensor(
        loop_, [&tensors_map](const Expr* x) {
          if (x->as_tensor()) {
            tensors_map.insert({x->as_tensor()->name, *x});
            return true;
          }
          return false;
        });
    auto find_store = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop_, [](const Expr* x) { return x->As<ir::Store>(); });
    for (auto store : find_store) {
      store.As<ir::Store>()->tensor =
          tensors_map.at(store.As<ir::Store>()->tensor.as_tensor()->name);
    }
    auto find_load = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop_, [](const Expr* x) { return x->As<ir::Load>(); });
    for (auto load : find_load) {
      load.As<ir::Load>()->tensor =
          tensors_map.at(load.As<ir::Load>()->tensor.as_tensor()->name);
    }

    InsertBlock(new_loop_, loop_body, inserted_pos);
    return utils::Join(new_var_names, ",");
  }

 public:
  /*! \brief The root block */
  Expr root_;
  /*! \brief The given block to be moved */
  Expr block_;
  /*! \brief The given loop the block and its loop nest to be put under */
  Expr loop_;
  /*! \brief The new loop to replace the original loop */
  Expr new_loop_{nullptr};
  /*! \brief The new block realize to the moved block */
  Expr new_block_{nullptr};
  /*! \brief The plan to remove the given block by replacing this loop/block in
   * the AST */
  Expr source_expr{nullptr};
  /*! \brief The plan to remove the given block by replacing to this loop/block
   * in the AST */
  Expr target_expr{nullptr};
};

struct FixLocalBufferSize : public ir::IRMutator<> {
 public:
  explicit FixLocalBufferSize(const std::string& tensor_name)
      : tensor_name_(tensor_name) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Store* expr, Expr* op) override {
    if (op->As<Store>()->tensor.As<_Tensor_>()->name == tensor_name_) {
      op->As<Store>()->tensor.As<_Tensor_>()->shape = {Expr(1)};
      op->As<Store>()->tensor.As<_Tensor_>()->domain = {Expr(1)};
      op->As<Store>()->tensor.As<_Tensor_>()->buffer->shape = {Expr(1)};
      op->As<Store>()->indices = {Expr(0)};
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::Load* expr, Expr* op) override {
    if (op->As<Load>()->tensor.As<_Tensor_>()->name == tensor_name_) {
      op->As<Load>()->tensor.As<_Tensor_>()->shape = {Expr(1)};
      op->As<Load>()->tensor.As<_Tensor_>()->domain = {Expr(1)};
      op->As<Load>()->tensor.As<_Tensor_>()->buffer->shape = {Expr(1)};
      op->As<Load>()->indices = {Expr(0)};
    }
    IRMutator::Visit(expr, op);
  }
  std::string tensor_name_;
};

void ScheduleImpl::SetBuffer(Expr& block,
                             const std::string& memory_type,
                             bool fixed) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  auto find_tensor = ir::ir_utils::CollectIRNodesWithoutTensor(
      block, [&](const Expr* x) { return x->As<ir::Store>(); }, true);
  CHECK_EQ(find_tensor.size(), 1U)
      << "One block should only have one Store node!(except for root block)";
  auto& tensor = (*find_tensor.begin()).As<ir::Store>()->tensor;
  tensor.as_tensor_ref()->WithBuffer(
      memory_type, "_" + tensor.as_tensor_ref()->name + "_temp_buffer");

  auto exprs = this->GetModule().GetExprs();
  for (auto& it_expr : exprs) {
    auto find_tensor =
        ir::ir_utils::CollectIRNodesWithoutTensor(it_expr, [&](const Expr* x) {
          return x->as_tensor() &&
                 (x->as_tensor()->name == tensor.as_tensor_ref()->name ||
                  x->as_tensor()->name ==
                      tensor.as_tensor_ref()->name + "__reduce_init");
        });
    for (auto& t : find_tensor) {
      CHECK(t.as_tensor());
      t.as_tensor_ref()->Bind(tensor.as_tensor_ref()->buffer);
    }
  }

  // if buffer type == "local"
  if (memory_type == "local" && fixed) {
    FixLocalBufferSize mutator(block.As<ir::ScheduleBlockRealize>()
                                   ->schedule_block.As<ir::ScheduleBlock>()
                                   ->name);
    auto root = GetRootBlock(block);
    mutator(&root);
  }
}

void ScheduleImpl::MergeExprs() {
  auto exprs = this->GetModule().GetExprs();
  if (exprs.size() == 1U) return;
  CHECK(exprs[0].As<ir::Block>());
  CHECK_EQ(exprs[0].As<ir::Block>()->stmts.size(), 1U);
  CHECK(exprs[0].As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>());
  CHECK(exprs[0]
            .As<ir::Block>()
            ->stmts[0]
            .As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  std::vector<Expr> merged_block;
  merged_block.push_back(exprs[0]
                             .As<ir::Block>()
                             ->stmts[0]
                             .As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>()
                             ->body);
  VLOG(3) << "Before merging, exprs[0] is : " << exprs[0];
  for (int i = 1; i < exprs.size(); ++i) {
    auto root_block = ir::ir_utils::CollectIRNodesWithoutTensor(
        exprs[i],
        [&](const Expr* x) {
          return x->As<ir::ScheduleBlockRealize>() &&
                 x->As<ir::ScheduleBlockRealize>()->iter_values.empty();
        },
        true);
    CHECK_EQ(root_block.size(), 1U);
    for (auto& it_block : root_block) {
      auto& block_body = it_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>()
                             ->body;
      merged_block.push_back(block_body);
    }
  }
  for (auto& block : merged_block) {
    VLOG(3) << "in merged_block, it has " << block;
  }
  auto merged_expr = ir::Block::Make(merged_block);
  exprs[0]
      .As<ir::Block>()
      ->stmts[0]
      .As<ir::ScheduleBlockRealize>()
      ->schedule_block.As<ir::ScheduleBlock>()
      ->body = merged_expr;
  VLOG(3) << "After merging, exprs[0] is : " << exprs[0];
  exprs.erase(exprs.begin() + 1, exprs.end());
  this->SetExprs(exprs);
}

void ScheduleImpl::ComputeAt(const Expr& block,
                             const Expr& loop,
                             bool keep_unit_loops) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(loop.As<ir::For>());
  Expr root = this->GetRootBlock(block);

  VLOG(3) << "Begin ComputeAt of loop:\n" << loop << "\nat block:\n" << root;

  auto producers = GetProducers(block, root);
  auto consumers = GetConsumers(block, root);
  CheckComputeAtValidation(block, loop, root);
  LoopReconstructor reconstructor(root, block, loop);
  LeafBlockRemovalPlan remove_plan(
      block, &reconstructor.source_expr, &reconstructor.target_expr);
  remove_plan(&root);
  auto iter_ranges = CalculateRequiredRegions(block, loop, root, consumers);
  std::string new_var_names =
      reconstructor.MakeNewLoop(iter_ranges, keep_unit_loops, 0);
  auto sch_block_expr = block.As<ir::ScheduleBlockRealize>()->schedule_block;
  sch_block_expr.As<ir::ScheduleBlock>()->attrs.emplace(
      ir::attr::compute_at_extra_var, new_var_names);
  this->Replace(reconstructor.source_expr, reconstructor.target_expr);
  this->Replace(reconstructor.loop_, reconstructor.new_loop_);

  VLOG(3) << "After SimpleComputeAt, ir is:\n" << reconstructor.new_loop_;
}

void ScheduleImpl::SimpleComputeAt(const Expr& block, const Expr& loop) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(loop.As<ir::For>());
  std::vector<Expr> block_loops = this->GetLoops(block);
  Expr root = this->GetRootBlock(block);
  auto loops = GetLoopsOfExpr(loop, root);

  VLOG(3) << "Begin SimpleComputeAt of loop:\n"
          << loop << "\nat block:\n"
          << root;

  auto this_loop = loop;
  auto block_name = GetTensor(block)->name;
  auto this_block = block;
  if (GetLoopExtent(loops[0]) == 1 && GetLoopExtent(block_loops[0]) != 1) {
    this->Split(block_loops[0], {1, -1});
    this_block = this->GetBlock(block_name);
  } else if (GetLoopExtent(loops[0]) != 1 &&
             GetLoopExtent(block_loops[0]) == 1) {
    auto splited = this->Split(loops[0], {1, -1});
    this_loop = splited[1];
  }

  block_loops = this->GetLoops(this_block);
  root = this->GetRootBlock(this_block);
  loops = GetLoopsOfExpr(this_loop, root);

  CHECK_LE(loops.size(), block_loops.size());

  std::vector<Var> replaced_var;
  std::vector<Expr> substitute_expr;
  for (int i = 0; i < loops.size(); ++i) {
    CHECK_EQ(GetLoopExtent(loops[i]), GetLoopExtent(block_loops[i]));
    if (block_loops[i].As<ir::For>()->bind_info().valid() &&
        !loops[i].As<ir::For>()->bind_info().valid()) {
      loops[i].As<ir::For>()->set_bind_info(
          block_loops[i].As<ir::For>()->bind_info());
    }
    replaced_var.push_back(block_loops[i].As<ir::For>()->loop_var);
    substitute_expr.push_back(Expr(loops[i].As<ir::For>()->loop_var));
  }

  Expr result = loops.size() < block_loops.size()
                    ? ir::ir_utils::IRCopy(block_loops[loops.size()])
                    : ir::ir_utils::IRCopy(this_block);
  Expr new_loop = ir::ir_utils::IRCopy(this_loop);

  // Get the body of block_loop under the same loops
  auto body = block_loops.at(loops.size() - 1).As<ir::For>()->body;
  // collect if
  auto if_checker = [](const Expr* x) { return x->As<ir::IfThenElse>(); };
  auto if_set = ir::ir_utils::CollectIRNodesWithoutTensor(body, if_checker);
  for (auto if_expr : if_set) {
    auto checker = [block_name](const Expr* x) {
      return x->As<ir::ScheduleBlockRealize>() &&
             x->As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ScheduleBlock>()
                     ->name == block_name;
    };
    if (ir::ir_utils::CollectIRNodesWithoutTensor(if_expr, checker, true)
            .size() > 0) {
      result =
          IfThenElse::Make(if_expr.As<ir::IfThenElse>()->condition, result);
      break;
    }
  }

  ReplaceExpr(&result, replaced_var, substitute_expr);
  // When there are two identical IfThenElse
  if (new_loop.As<ir::For>() && new_loop.As<ir::For>()->body.As<ir::Block>() &&
      new_loop.As<ir::For>()
          ->body.As<ir::Block>()
          ->stmts[0]
          .As<ir::IfThenElse>()) {
    auto if_then_else = new_loop.As<ir::For>()->body.As<ir::Block>()->stmts[0];
    if (result.As<ir::IfThenElse>() &&
        if_then_else.As<ir::IfThenElse>()->condition ==
            result.As<ir::IfThenElse>()->condition) {
      new_loop.As<ir::For>()
          ->body.As<ir::Block>()
          ->stmts[0]
          .As<ir::IfThenElse>()
          ->true_case = ir::Block::Make({result.As<ir::IfThenElse>()->true_case,
                                         new_loop.As<ir::For>()
                                             ->body.As<ir::Block>()
                                             ->stmts[0]
                                             .As<ir::IfThenElse>()
                                             ->true_case});
    } else {
      std::vector<ir::Expr>::iterator pos =
          new_loop.As<ir::For>()->body.As<ir::Block>()->stmts.begin();
      new_loop.As<ir::For>()->body.As<ir::Block>()->stmts.insert(pos, result);
    }
  } else {
    new_loop.As<ir::For>()->body =
        ir::Block::Make({result, new_loop.As<ir::For>()->body});
  }

  Expr source_expr{nullptr};
  Expr target_expr{nullptr};

  LeafBlockRemovalPlan remove_plan(
      result.As<ir::For>() ? block_loops[loops.size()] : this_block,
      &source_expr,
      &target_expr);
  remove_plan(&root);

  this->Replace(source_expr, target_expr);
  this->Replace(this_loop, new_loop);

  VLOG(3) << "After SimpleComputeAt, ir is:\n" << new_loop;
}

void ScheduleImpl::ReverseComputeAt(const Expr& block,
                                    const Expr& loop,
                                    bool keep_unit_loops) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(loop.As<ir::For>());
  Expr root = this->GetRootBlock(block);
  auto producers = GetProducers(block, root);
  auto consumers = GetConsumers(block, root);
  CheckComputeAtValidation(block, loop, root);
  LoopReconstructor reconstructor(root, block, loop);
  LeafBlockRemovalPlan remove_plan(
      block, &reconstructor.source_expr, &reconstructor.target_expr);
  remove_plan(&root);
  auto iter_ranges =
      CalculateRequiredRegions(block, loop, root, producers, false);
  std::string new_var_names =
      reconstructor.MakeNewLoop(iter_ranges, keep_unit_loops, -1);
  auto sch_block_expr = block.As<ir::ScheduleBlockRealize>()->schedule_block;
  sch_block_expr.As<ir::ScheduleBlock>()->attrs.emplace(
      ir::attr::reverse_compute_at_extra_var, new_var_names);
  this->Replace(reconstructor.source_expr, reconstructor.target_expr);
  this->Replace(reconstructor.loop_, reconstructor.new_loop_);
  return;
}

void BaseInliner::operator()(Expr* expr) {
  IRMutator::Visit(&tgt_stmt, &tgt_stmt);
  IRMutator::Visit(expr, expr);
}

void BaseInliner::Visit(const ir::Block* expr, Expr* op) {
  if (*op == src_stmt) {
    *op = tgt_stmt;
    return;
  }
  IRMutator::Visit(expr, op);
}

bool BaseInliner::UpdateAndCheckIndexVars(const std::vector<Expr>& indices,
                                          int expected_ndim) {
  int n = indices.size();
  if (n != expected_ndim) {
    return false;
  }
  std::vector<Var> result;
  result.reserve(n);
  for (auto& i : indices) {
    if (i.as_var()) {
      result.push_back(i.as_var_ref());
    } else {
      return false;
    }
  }
  int n_distinct = std::set<Var, CompVar>(result.begin(), result.end()).size();
  if (n != n_distinct) {
    return false;
  }
  if (idx_vars_.empty()) {
    idx_vars_ = std::move(result);
  } else {
    if (idx_vars_.size() != result.size()) return false;
    for (int i = 0; i < result.size(); ++i) {
      if (Expr(idx_vars_[i]) != Expr(result[i])) return false;
    }
  }
  return true;
}

void BaseInliner::SetIndexSubstitution(const std::vector<Expr>& indices) {
  CHECK_EQ(indices.size(), idx_vars_.size());
  int n = idx_vars_.size();
  idx_sub_var_.reserve(n);
  idx_sub_expr_.reserve(n);
  for (int i = 0; i < n; ++i) {
    idx_sub_var_.push_back(idx_vars_[i]);
    idx_sub_expr_.push_back(indices[i]);
  }
}

bool ComputeInliner::BodyPatternAllowInline() {
  if (!inlined_store_.defined()) {
    return false;
  }
  CHECK(inlined_store_.As<Store>());
  auto find_vars = ir::ir_utils::CollectIRNodesWithoutTensor(
      inlined_store_, [&](const Expr* x) { return x->as_var(); });
  std::set<Var, CompVar> vars_set;
  for (auto& i : find_vars) vars_set.insert(i.as_var_ref());
  int n_vars = vars_set.size();
  if (!UpdateAndCheckIndexVars(inlined_store_.As<Store>()->indices, n_vars)) {
    return false;
  }
  return true;
}

void ComputeInliner::Visit(const ir::Load* expr, Expr* op) {
  if ((expr->tensor).as_tensor_ref()->name == inlined_tensor_->name) {
    *op = ReplaceInlinedTensor(op);
    return;
  }
  IRMutator::Visit(expr, op);
}

//! Replace the 'Load' node on the tensor to 'Load' node of its producers.
Expr ComputeInliner::ReplaceInlinedTensor(Expr* load) {
  CHECK(load->As<ir::Load>());
  SetIndexSubstitution(load->As<ir::Load>()->indices);
  Expr value_copy = ir::ir_utils::IRCopy(inlined_store_.As<Store>()->value);
  ReplaceExpr(&value_copy, idx_sub_var_, idx_sub_expr_);
  return value_copy;
}

void ScheduleImpl::ComputeInline(const Expr& schedule_block) {
  CHECK(schedule_block.As<ir::ScheduleBlockRealize>());
  Expr root = this->GetRootBlock(schedule_block);
  Expr store = CheckComputeInlineValidationAndGetStore(schedule_block, root);
  ComputeInliner inliner(store.As<ir::Store>()->tensor.as_tensor_ref(), store);
  CHECK(inliner.BodyPatternAllowInline());
  // Create a plan that removes the block to be inlined
  LeafBlockRemovalPlan remove_plan(
      schedule_block, &inliner.src_stmt, &inliner.tgt_stmt);
  remove_plan(&root);
  inliner(&root);
  return;
}

bool ComputeInlineChecker::Check() {
  Expr root = ir_schedule_.GetRootBlock(block_);
  store_ = CheckComputeInlineValidationAndGetStore(block_, root);
  IRMutator::Visit(&root, &root);
  return !should_skip_;
}

void ComputeInlineChecker::BuildDataDependency() {
  ir_schedule_.SetBuffer(block_, "shared", true);
  auto loops = ir_schedule_.GetLoops(block_);
  ir_schedule_.SyncThreads(loops.back(), true);
}

bool ReverseComputeInliner::BodyPatternAllowInline() {
  if (!inlined_store_.defined()) {
    return false;
  }
  if (!inlined_load_.defined()) {
    return false;
  }
  if (!target_store_.defined()) {
    return false;
  }
  CHECK(inlined_store_.As<Store>());
  CHECK(inlined_load_.As<Load>());
  CHECK(target_store_.As<Store>());
  auto find_vars = ir::ir_utils::CollectIRNodesWithoutTensor(
      inlined_store_, [&](const Expr* x) { return x->as_var(); });
  std::set<Var, CompVar> vars_set;
  for (auto& i : find_vars) vars_set.insert(i.as_var_ref());
  int n_vars = vars_set.size();
  if (!UpdateAndCheckIndexVars(inlined_store_.As<Store>()->indices, n_vars)) {
    return false;
  }
  return true;
}

void ReverseComputeInliner::Visit(const ir::Load* expr, Expr* op) {
  if ((expr->tensor).as_tensor_ref()->name == inlined_tensor_->name) {
    *op = inlined_store_.As<Store>()->value;
    return;
  }
  IRMutator::Visit(expr, op);
}

void ReverseComputeInliner::Visit(const ir::Store* expr, Expr* op) {
  if ((expr->tensor).as_tensor_ref()->name == inlined_tensor_->name) {
    *op = ReplaceTargetTensor(op);
    return;
  }
  IRMutator::Visit(expr, op);
}

//! Replace the 'Load' node on the tensor to 'Load' node of its producers.
Expr ReverseComputeInliner::ReplaceInlinedTensor(Expr* load) {
  CHECK(load->As<ir::Load>());
  SetIndexSubstitution(load->As<ir::Load>()->indices);
  Expr value_copy = ir::ir_utils::IRCopy(inlined_store_.As<Store>()->value);
  return value_copy;
}

Expr ReverseComputeInliner::ReplaceTargetTensor(Expr* store) {
  auto indices = inlined_load_.As<ir::Load>()->indices;
  CHECK_EQ(indices.size(), idx_vars_.size());
  size_t n = idx_vars_.size();
  idx_sub_var_.reserve(n);
  idx_sub_expr_.reserve(n);
  for (int i = 0; i < n; ++i) {
    idx_sub_var_.emplace_back(indices[i].as_var_ref());
    idx_sub_expr_.emplace_back(idx_vars_[i]);
  }

  Expr value_copy = ir::ir_utils::IRCopy(target_store_);
  ReplaceExpr(&value_copy, idx_sub_var_, idx_sub_expr_);
  return value_copy;
}

void ScheduleImpl::ReverseComputeInline(const Expr& schedule_block) {
  Expr root = this->GetRootBlock(schedule_block);
  auto exprs =
      CheckReverseComputeInlineValidationAndGetExprs(schedule_block, root);
  Expr inlined_load = std::get<0>(exprs);
  Expr inlined_store = std::get<1>(exprs);
  Expr target_store = std::get<2>(exprs);
  ReverseComputeInliner inliner(
      inlined_store.As<ir::Store>()->tensor.as_tensor_ref(),
      inlined_store,
      inlined_load,
      target_store);
  CHECK(inliner.BodyPatternAllowInline());
  // Create a plan that removes the block to be inlined
  LeafBlockRemovalPlan remove_plan(
      schedule_block, &inliner.src_stmt, &inliner.tgt_stmt);
  remove_plan(&root);
  inliner(&root);
  inliner(&root);
}

struct FindBlockParent : public ir::IRMutator<> {
 public:
  explicit FindBlockParent(const std::string& block_name)
      : block_name_(block_name) {}

  void operator()(Expr* expr) { IRMutator::Visit(expr, expr); }

 private:
  void Visit(const ir::Block* expr, Expr* op) override {
    if (target_) return;
    for (auto& stmt : expr->stmts) {
      if (stmt.As<ir::ScheduleBlockRealize>()) {
        if (stmt.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name == block_name_) {
          target_ = op;
          return;
        }
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::For* expr, Expr* op) override {
    if (target_) return;
    if (expr->body.As<ir::ScheduleBlockRealize>()) {
      if (expr->body.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == block_name_) {
        target_ = op;
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  void Visit(const ir::ScheduleBlock* expr, Expr* op) override {
    if (target_) return;
    if (expr->body.As<ir::ScheduleBlockRealize>()) {
      if (expr->body.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name == block_name_) {
        target_ = op;
        return;
      }
    }
    IRMutator::Visit(expr, op);
  }

  std::string block_name_;

 public:
  ir::Expr* target_{nullptr};
};

Expr ScheduleImpl::AddUnitLoop(const Expr& block) const {
  auto exprs = module_expr_.GetExprs();
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  std::string block_name = block.As<ir::ScheduleBlockRealize>()
                               ->schedule_block.As<ir::ScheduleBlock>()
                               ->name;

  FindBlockParent visitor(block_name);
  for (auto expr : exprs) {
    visitor(&expr);
    if (visitor.target_) {
      break;
    }
  }

  CHECK(visitor.target_) << ", block name : " << block_name << "\n" << exprs;
  if (visitor.target_->As<ir::Block>()) {
    for (auto& stmt : visitor.target_->As<ir::Block>()->stmts) {
      if (stmt.As<ir::ScheduleBlockRealize>()) {
        if (stmt.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name == block_name) {
          auto block = ir::Block::Make({GetBlock(block_name)});
          auto loop = ir::For::Make(ir::Var(common::UniqName("ix")),
                                    ir::Expr(0),
                                    ir::Expr(1),
                                    ir::ForType::Serial,
                                    ir::DeviceAPI::UNK,
                                    block);
          stmt = loop;
          return loop;
        }
      }
    }
  } else if (visitor.target_->As<ir::For>()) {
    auto block = ir::Block::Make({visitor.target_->As<ir::For>()->body});
    auto loop = ir::For::Make(ir::Var(common::UniqName("ix")),
                              ir::Expr(0),
                              ir::Expr(1),
                              ir::ForType::Serial,
                              ir::DeviceAPI::UNK,
                              block);
    visitor.target_->As<ir::For>()->body = loop;
    return loop;
  } else if (visitor.target_->As<ir::ScheduleBlock>()) {
    auto block =
        ir::Block::Make({visitor.target_->As<ir::ScheduleBlock>()->body});
    auto loop = ir::For::Make(ir::Var(common::UniqName("ix")),
                              ir::Expr(0),
                              ir::Expr(1),
                              ir::ForType::Serial,
                              ir::DeviceAPI::UNK,
                              block);
    visitor.target_->As<ir::ScheduleBlock>()->body = loop;
    return loop;
  } else {
    LOG(FATAL) << "Can't find block's parent!";
  }
  LOG(FATAL) << "Shouldn't reach code here in AddUnitLoop";
  return Expr{nullptr};
}

std::vector<Expr> ScheduleImpl::GetLoops(const Expr& block) const {
  std::vector<Expr> result;
  auto exprs = module_expr_.GetExprs();
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  std::string block_name = block.As<ir::ScheduleBlockRealize>()
                               ->schedule_block.As<ir::ScheduleBlock>()
                               ->name;

  for (auto& it_expr : exprs) {
    ir::FindLoopsVisitor visitor(block);
    auto find_loops = visitor(&it_expr);
    if (!find_loops.empty()) {
      if (!result.empty())
        LOG(FATAL) << "Find block with name: \n"
                   << block_name << " appeared in more than one AST!";
      result = find_loops;
    }
  }

  if (result.empty()) {
    result.push_back(AddUnitLoop(block));
  }
  return result;
}

std::vector<Expr> ScheduleImpl::GetLoops(const std::string& block_name) const {
  Expr block = this->GetBlock(block_name);
  std::vector<Expr> result = this->GetLoops(block);
  return result;
}

std::vector<Expr> ScheduleImpl::GetAllBlocks() const {
  std::vector<Expr> result;
  auto exprs = module_expr_.GetExprs();
  for (auto& it_expr : exprs) {
    ir::FindBlocksVisitor visitor;
    auto find_blocks = visitor(&it_expr);
    result.insert(result.end(), find_blocks.begin(), find_blocks.end());
  }
  for (auto& it_expr : exprs) {
    VLOG(3) << "it_expr is : " << it_expr;
  }
  CHECK(!result.empty()) << "Didn't find blocks in expr.";
  return result;
}

std::vector<Expr> ScheduleImpl::GetChildBlocks(const Expr& expr) const {
  CHECK(expr.As<ir::ScheduleBlockRealize>() || expr.As<ir::For>());
  ir::FindBlocksVisitor visitor;
  std::vector<Expr> result = visitor(&expr);
  return result;
}

bool ScheduleImpl::HasBlock(const std::string& block_name) const {
  auto exprs = module_expr_.GetExprs();
  for (auto& it_expr : exprs) {
    ir::FindBlocksVisitor visitor(block_name);
    auto find_blocks = visitor(&it_expr);
    if (!find_blocks.empty()) {
      CHECK_EQ(find_blocks.size(), 1U)
          << "There should not be more than 1 block with identical name!";
      return true;
    }
  }
  return false;
}

Expr ScheduleImpl::GetBlock(const std::string& block_name) const {
  Expr result;
  auto exprs = module_expr_.GetExprs();
  for (auto& it_expr : exprs) {
    ir::FindBlocksVisitor visitor(block_name);
    auto find_blocks = visitor(&it_expr);
    if (!find_blocks.empty()) {
      CHECK_EQ(find_blocks.size(), 1U)
          << "There should not be more than 1 block with identical name!";
      result = find_blocks[0];
      return result;
    }
  }
  LOG(FATAL) << "Didn't find a block with name " << block_name
             << " in this ModuleExpr!";
}

void ScheduleImpl::Annotate(const Expr& block,
                            const std::string& key,
                            const attr_t& value) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  auto copied_block = ir::ir_utils::IRCopy(block);
  auto* schedule_block = copied_block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  schedule_block->attrs.emplace(key, value);
  this->Replace(block, copied_block);
}

void ScheduleImpl::Unannotate(Expr& block, const std::string& ann_key) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  auto* schedule_block = block.As<ir::ScheduleBlockRealize>()
                             ->schedule_block.As<ir::ScheduleBlock>();
  if (schedule_block->attrs.count(ann_key)) {
    schedule_block->attrs.erase(ann_key);
  } else {
    LOG(WARNING) << "Can't find annotation with key: " << ann_key;
    return;
  }
}

void ScheduleImpl::FlattenLoops(const std::vector<Expr>& loops,
                                const bool flat_tensor) {
  CHECK_GT(loops.size(), 0) << "Loops can't be empty!";
  VLOG(4) << "Before FlattenLoops, ir is:\n" << loops[0];
  // compute loop
  int extent = 1;
  std::vector<int> strides;
  std::vector<ir::Var> loop_vars(loops.size());
  for (int idx = loops.size() - 1; idx >= 0; --idx) {
    strides.insert(strides.begin(), extent);
    extent *= loops[idx].As<ir::For>()->extent.as_int32();
    loop_vars[idx] = loops[idx].As<ir::For>()->loop_var;
  }
  CHECK_EQ(loops.size(), strides.size());

  // create new loop.
  auto last = loops.back().As<ir::For>();
  auto var = ir::Var("flat_i");
  auto _var = ir::Var("_flat_i");
  auto loop = ir::For::Make(var,
                            ir::Expr(0),
                            ir::Expr(extent),
                            last->for_type(),
                            last->device_api,
                            last->body);

  // map loop var to old loop var.
  auto _iter = ir::Expr(_var);
  std::unordered_map<std::string, ir::Expr> loops_to_flat_var_map;
  for (int idx = 0; idx < strides.size(); ++idx) {
    if (strides[idx] == 1) {
      // flat_i_to_loop_var.push_back(_iter);
      loops_to_flat_var_map[loops[idx].As<ir::For>()->loop_var->name] = _iter;
    } else {
      // flat_i_to_loop_var.push_back(_iter / Expr(strides[idx]));
      loops_to_flat_var_map[loops[idx].As<ir::For>()->loop_var->name] =
          _iter / Expr(strides[idx]);
      _iter = _iter % Expr(strides[idx]);
    }
  }

  ir::FindBlocksVisitor visitor;
  auto blocks = visitor(&last->body);
  auto can_do_flat = [](const std::vector<Expr>& indexs,
                        const std::vector<Var>& loop_vars) {
    if (indexs.size() != loop_vars.size()) {
      return false;
    }

    for (int idx = 0; idx < indexs.size(); ++idx) {
      if (!indexs[idx].as_var()) {
        return false;
      } else {
        auto var = indexs[idx].as_var_ref();
        if (var->name != loop_vars[idx]->name) {
          return false;
        }
      }
    }
    return true;
  };

  // change blocks iter value/iter var
  for (auto& block : blocks) {
    auto block_realize = block.As<ir::ScheduleBlockRealize>();
    auto schedule_block = block_realize->schedule_block.As<ir::ScheduleBlock>();

    // checkout loops in orders.
    std::vector<std::string> var_names = {};
    CHECK_GE(block_realize->iter_values.size(), loop_vars.size())
        << "the number of iter bind values must be more than loop vars!";
    for (int idx = 0; idx < block_realize->iter_values.size(); ++idx) {
      auto& iter = block_realize->iter_values[idx];
      if (iter.is_var()) {
        CHECK_EQ(iter.as_var_ref()->name, loop_vars[idx]->name)
            << "loops is not the same order with tensor!";
      } else {
        CHECK(iter.As<IntImm>());
        CHECK_EQ(iter.as_int32(), 0);
      }
    }

    auto exprs = ir::ir_utils::CollectIRNodesInOrder(
        schedule_block->body,
        [&](const Expr* x) { return x->As<ir::Store>() || x->As<ir::Load>(); });
    // reverse exprs from last to first.
    std::reverse(std::begin(exprs), std::end(exprs));

    std::vector<ir::Var> var_to_replace;
    std::vector<ir::Expr> flat_i_to_loop_var;
    // if iter var is more than flat i to loop, there exist dim = 1.
    for (int idx = 0; idx < block_realize->iter_values.size(); ++idx) {
      if (block_realize->iter_values[idx].is_var()) {
        var_to_replace.push_back(schedule_block->iter_vars[idx]);
        auto var_name = block_realize->iter_values[idx].as_var_ref()->name;
        CHECK(loops_to_flat_var_map.count(var_name))
            << "Can't find var name : " << var_name;
        flat_i_to_loop_var.push_back(loops_to_flat_var_map[var_name]);
      } else {
        CHECK_EQ(block_realize->iter_values[idx].as_int32(), 0);
        // insert var -> 0, to replace var to 0.
        var_to_replace.push_back(schedule_block->iter_vars[idx]);
        flat_i_to_loop_var.push_back(Expr(0));
      }
    }
    CHECK_EQ(var_to_replace.size(), flat_i_to_loop_var.size());

    for (auto expr : exprs) {
      if (expr.As<ir::Store>()) {
        auto store = expr.As<ir::Store>();
        if (store->is_addr_tensor()) {
          auto t = store->tensor.as_tensor_ref();
          CHECK(!t->reduce_axis.size());
          auto tsize = std::accumulate(t->shape.begin(),
                                       t->shape.end(),
                                       1,
                                       [](const int sum, const Expr& expr) {
                                         return sum * expr.as_int32();
                                       });
          if ((!flat_tensor &&
               !can_do_flat(store->indices, schedule_block->iter_vars)) ||
              extent != tsize) {
            // just replace indexs
            for (auto& indice : store->indices) {
              if (!indice.is_var()) {
                continue;
              }
              ReplaceExpr(&indice, var_to_replace, flat_i_to_loop_var);
            }
            // compute index and flat tensor.
            store->indices = {store->index()};
            continue;
          }
          // update var and shape
          store->indices = {Expr(_var)};
        }
      } else {
        auto load = expr.As<ir::Load>();
        if (load->is_addr_tensor()) {
          auto t = load->tensor.as_tensor_ref();
          CHECK(!t->reduce_axis.size());
          auto tsize = std::accumulate(t->shape.begin(),
                                       t->shape.end(),
                                       1,
                                       [](const int sum, const Expr& expr) {
                                         return sum * expr.as_int32();
                                       });
          if ((!flat_tensor &&
               !can_do_flat(load->indices, schedule_block->iter_vars)) ||
              extent != tsize) {
            // just replace indexs
            for (auto& indice : load->indices) {
              if (!indice.is_var()) {
                continue;
              }
              ReplaceExpr(&indice, var_to_replace, flat_i_to_loop_var);
            }
            // compute index and flat tensor.
            load->indices = {load->index()};
            continue;
          }
          // update var and shape
          load->indices = {Expr(_var)};
        }
      }
    }
    ReplaceExpr(&schedule_block->body, var_to_replace, flat_i_to_loop_var);

    // update iter values
    auto iter = ir::Expr(var);
    block_realize->iter_values = {iter};

    // update iter_vars
    schedule_block->iter_vars = {_var};
    CHECK_EQ(block_realize->iter_values.size(),
             schedule_block->iter_vars.size());
  }

  this->Replace(loops[0], loop);
  VLOG(4) << "After FlattenLoops, ir is:\n" << loop;
}

void ScheduleImpl::CopyTransformAndLoopInfo(
    const std::string& block_name, const std::string& block_target_name) {
  auto block = this->GetBlock(block_name);
  auto block_target = this->GetBlock(block_target_name);
  this->CopyTransformAndLoopInfo(block, block_target);
}

void ScheduleImpl::CopyTransformAndLoopInfo(const Expr& block,
                                            const Expr& block_target) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block_target.As<ir::ScheduleBlockRealize>());
  auto exprs = this->GetModule().GetExprs();
  CHECK_EQ(exprs.size(), 1U);
  auto expr = exprs[0];
  auto vars = block.As<ir::ScheduleBlockRealize>()
                  ->schedule_block.As<ir::ScheduleBlock>()
                  ->iter_vars;
  auto vars_target = block_target.As<ir::ScheduleBlockRealize>()
                         ->schedule_block.As<ir::ScheduleBlock>()
                         ->iter_vars;
  auto old_iter_values = block.As<ir::ScheduleBlockRealize>()->iter_values;
  auto iter_values_target =
      block_target.As<ir::ScheduleBlockRealize>()->iter_values;
  std::vector<Expr> new_iter_values;
  for (int i = 0; i < vars.size() && i < vars_target.size(); ++i) {
    CHECK(vars[i]->upper_bound.defined() &&
          vars_target[i]->upper_bound.defined());
    if (vars[i]->upper_bound.is_constant() &&
        vars_target[i]->upper_bound.is_constant() &&
        vars[i]->upper_bound.get_constant() ==
            vars_target[i]->upper_bound.get_constant() &&
        !vars[i]->is_reduce_axis && !vars_target[i]->is_reduce_axis) {
      new_iter_values.push_back(iter_values_target[i]);
      VLOG(3) << "new_iter_values.push_back " << iter_values_target[i];
    } else {
      break;
    }
  }

  if (new_iter_values.empty())
    LOG(FATAL) << "Cannot CopyTransformAndLoopInfo since shape[0] of source "
                  "and target is not equal! "
               << vars[0]->upper_bound << " v.s "
               << vars_target[0]->upper_bound;

  int changed_loop_num = new_iter_values.size();
  std::set<std::string> used_target_loop_vars;
  for (auto& iter_val : new_iter_values) {
    auto find_partial_loop =
        ir::ir_utils::CollectIRNodesWithoutTensor(iter_val, [&](const Expr* x) {
          if (x->as_var()) used_target_loop_vars.insert(x->as_var_ref()->name);
          return x->as_var();
        });
  }
  CHECK(!used_target_loop_vars.empty());
  std::vector<Expr> used_target_loops;
  auto expr_copy = ir::ir_utils::IRCopy(expr);
  for (auto& var : used_target_loop_vars) {
    auto find_loop_var = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr_copy,
        [&](const Expr* x) {
          return x->As<ir::For>() && x->As<ir::For>()->loop_var->name == var &&
                 Contains(*x, block_target);
        },
        true);
    CHECK_EQ(find_loop_var.size(), 1U);
    used_target_loops.push_back(*find_loop_var.begin());
    VLOG(3) << "used_target_loops push_back " << used_target_loops.back();
  }
  std::sort(
      used_target_loops.begin(), used_target_loops.end(), [&](Expr i, Expr j) {
        return (utils::GetStreamCnt(i).size() > utils::GetStreamCnt(j).size());
      });
  for (int i = new_iter_values.size(); i < old_iter_values.size(); ++i) {
    CHECK(old_iter_values[i].as_var());
    new_iter_values.push_back(old_iter_values[i]);
  }
  Expr new_loop;
  VLOG(3) << "changed_loop_num is : " << changed_loop_num;
  VLOG(3) << "old_iter_values.size() is : " << old_iter_values.size();
  if (changed_loop_num >= static_cast<int>(old_iter_values.size())) {
    new_loop = ir::ir_utils::IRCopy(block);
    new_loop.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  } else {
    CHECK(old_iter_values[changed_loop_num].as_var());
    auto old_var = old_iter_values[changed_loop_num].as_var_ref();
    auto find_partial_loop = ir::ir_utils::CollectIRNodesWithoutTensor(
        expr,
        [&](const Expr* x) {
          return x->As<ir::For>() &&
                 x->As<ir::For>()->loop_var->name == old_var->name &&
                 Contains(*x, block);
        },
        true);
    CHECK_EQ(find_partial_loop.size(), 1U);
    new_loop = ir::ir_utils::IRCopy(*find_partial_loop.begin());
    auto find_schedule_block = ir::ir_utils::CollectIRNodesWithoutTensor(
        new_loop,
        [&](const Expr* x) { return x->As<ir::ScheduleBlockRealize>(); },
        true);
    CHECK_EQ(find_schedule_block.size(), 1U);
    Expr sch_block = (*find_schedule_block.begin());
    sch_block.As<ir::ScheduleBlockRealize>()->iter_values = new_iter_values;
  }
  VLOG(3) << "new_loop is : " << new_loop;
  CHECK(!used_target_loops.empty());
  Expr res;
  if (used_target_loops.size() == 1) {
    auto for_loop = used_target_loops[0].As<ir::For>();
    res = For::Make(for_loop->loop_var,
                    for_loop->min,
                    for_loop->extent,
                    for_loop->for_type(),
                    for_loop->device_api,
                    new_loop,
                    for_loop->vectorize_info(),
                    for_loop->bind_info());
  } else {
    Expr outer_loop = used_target_loops.front();
    Expr inner_loop = used_target_loops.back();
    inner_loop.As<ir::For>()->body = Block::Make({new_loop});
    res = outer_loop;
  }
  VLOG(3) << "res is : " << res;
  std::vector<Expr> all_loops = this->GetLoops(block);
  CHECK(!all_loops.empty());
  this->Replace(all_loops[0], res);
}

std::vector<Expr> ScheduleImpl::SamplePerfectTile(
    utils::LinearRandomEngine::StateType* rand_seed,
    const Expr& loop,
    int n,
    int max_innermost_factor) {
  CHECK(loop.As<ir::For>())
      << "Expr param of SamplePerfectTile should be a For loop";
  CHECK_GE(n, 2) << "The number of tile factors should be at least 2";
  CHECK_GE(max_innermost_factor, 1)
      << "The max innermost factor should be at least 1";
  CHECK(common::is_zero(loop.As<ir::For>()->min))
      << "The For loop should start from 0";
  int loop_extent = GetLoopExtent(loop);
  std::vector<int> innermost_factors;
  for (int i = max_innermost_factor; i >= 1; --i) {
    if (loop_extent % i == 0) {
      innermost_factors.push_back(i);
    }
  }
  CHECK(!innermost_factors.empty()) << "No innermost factor found";
  int innermost_factor = innermost_factors[utils::SampleUniformInt(
      0, innermost_factors.size(), rand_seed)];
  auto result = SampleTile(rand_seed, n - 1, loop_extent / innermost_factor);
  std::vector<Expr> result_expr;
  for (auto& factor : result) {
    result_expr.push_back(Expr(factor));
  }
  result_expr.push_back(Expr(innermost_factor));
  return result_expr;
}

Expr ScheduleImpl::SampleCategorical(
    utils::LinearRandomEngine::StateType* rand_seed,
    const std::vector<int>& candidates,
    const std::vector<float>& probs) {
  // check two sizes
  CHECK_EQ(candidates.size(), probs.size())
      << "candidates and probs must have same size.";
  int seed_idx = utils::SampleDiscreteFromDistribution(probs, rand_seed);
  auto result = candidates[seed_idx];
  Expr result_expr(result);
  return result_expr;
}

IRSchedule::IRSchedule() {}

IRSchedule::IRSchedule(const ModuleExpr& module_expr,
                       utils::LinearRandomEngine::StateType rand_seed,
                       bool debug_flag,
                       utils::ErrorMessageLevel err_msg_level) {
  impl_ =
      std::make_unique<ScheduleImpl>(module_expr, debug_flag, err_msg_level);
  this->InitSeed(rand_seed);
}

IRSchedule::IRSchedule(ir::ModuleExpr&& mod_expr,
                       ScheduleDesc&& trace,
                       utils::LinearRandomEngine::StateType rand_seed)
    : impl_(std::make_unique<ScheduleImpl>(std::move(mod_expr))),
      trace_(std::move(trace)) {
  this->InitSeed(rand_seed);
}

IRSchedule::IRSchedule(const IRSchedule& other)
    : impl_(std::make_unique<ScheduleImpl>(
          ir::ir_utils::IRCopy(other.GetModule()))),
      trace_(other.trace_) {
  this->InitSeed(other.ForkSeed());
}

IRSchedule& IRSchedule::operator=(const IRSchedule& src) {
  impl_ = std::make_unique<ScheduleImpl>(ir::ir_utils::IRCopy(src.GetModule()));
  trace_ = src.trace_;
  this->InitSeed(src.ForkSeed());
  return *this;
}

IRSchedule::IRSchedule(IRSchedule&& other)
    : impl_(std::move(other.impl_)), trace_(std::move(other.trace_)) {
  this->InitSeed(other.ForkSeed());
}

IRSchedule& IRSchedule::operator=(IRSchedule&& src) {
  impl_ = std::move(src.impl_);
  trace_ = std::move(src.trace_);
  this->InitSeed(src.ForkSeed());
  return *this;
}

IRSchedule::~IRSchedule() {}

void IRSchedule::InitSeed(utils::LinearRandomEngine::StateType rand_seed) {
  this->rand_seed_ = utils::LinearRandomEngine::NormalizeState(rand_seed);
}

utils::LinearRandomEngine::StateType IRSchedule::ForkSeed() const {
  return utils::ForkRandomState(&rand_seed_);
}

void IRSchedule::SetExprs(const std::vector<Expr>& exprs) {
  return impl_->SetExprs(exprs);
  // no need to trace
}

const ModuleExpr& IRSchedule::GetModule() const {
  return impl_->GetModule();
  // no need to trace
}

bool IRSchedule::HasBlock(const std::string& block_name) const {
  return impl_->HasBlock(block_name);
  // no need to trace
}

void IRSchedule::MergeExprs() {
  impl_->MergeExprs();
  trace_.Append(ScheduleDesc::Step("MergeExprs", {}, {}, {}));
}

std::vector<Expr> IRSchedule::GetLoops(const Expr& block) const {
  auto results = impl_->GetLoops(block);
  trace_.Append(ScheduleDesc::Step(
      "GetLoops", {{"block", std::vector<Expr>({block})}}, {}, results));
  return results;
}

std::vector<Expr> IRSchedule::GetLoops(const std::string& block_name) const {
  auto results = impl_->GetLoops(block_name);
  trace_.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", block_name}}, results));
  return results;
}

std::vector<Expr> IRSchedule::GetAllBlocks() const {
  auto results = impl_->GetAllBlocks();
  trace_.Append(ScheduleDesc::Step("GetAllBlocks", {}, {}, results));
  return results;
}

std::vector<Expr> IRSchedule::GetChildBlocks(const Expr& expr) const {
  auto results = impl_->GetChildBlocks(expr);
  trace_.Append(ScheduleDesc::Step(
      "GetChildBlocks", {{"expr", std::vector<Expr>({expr})}}, {}, results));
  return results;
}

Expr IRSchedule::GetBlock(const std::string& block_name) const {
  auto result = impl_->GetBlock(block_name);
  trace_.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", block_name}}, {result}));
  return result;
}

std::vector<Expr> IRSchedule::Split(const Expr& loop,
                                    const std::vector<int>& factors) {
  std::vector<Expr> decision = SamplePerfectTile(
      loop, factors.size(), loop.As<ir::For>()->extent.as_int32(), factors);
  auto results = Split(loop, decision);
  return results;
}

std::vector<Expr> IRSchedule::Split(const std::string& block_name,
                                    int loop_index,
                                    const std::vector<int>& factors) {
  std::vector<Expr> all_loops = this->GetLoops(block_name);
  Expr loop_expr;
  CHECK_LT(loop_index, (int)all_loops.size())
      << "The loop index in Split should be less than total loop's number.";
  CHECK_GE(loop_index, 0) << "The loop index in Split should be >= 0.";
  loop_expr = all_loops[loop_index];

  return this->Split(loop_expr, factors);
}

std::vector<Expr> IRSchedule::Split(const Expr& loop,
                                    const std::vector<Expr>& factors) {
  std::vector<int> int_factors;
  std::transform(factors.begin(),
                 factors.end(),
                 std::back_inserter(int_factors),
                 [](Expr x) { return x.as_int32(); });
  auto results = impl_->Split(loop, int_factors);
  trace_.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({loop})}, {"factors", factors}},
      {},
      results));
  return results;
}

Expr IRSchedule::Fuse(const std::vector<Expr>& loops) {
  auto result = impl_->Fuse(loops);
  trace_.Append(ScheduleDesc::Step("Fuse", {{"loops", loops}}, {}, {result}));
  return result;
}

Expr IRSchedule::Fuse(const std::string& block_name,
                      const std::vector<int>& loops_index) {
  auto result = impl_->Fuse(block_name, loops_index);
  trace_.Append(ScheduleDesc::Step(
      "FuseWithName",
      {},
      {{"block_name", block_name}, {"loops_index", loops_index}},
      {result}));
  return result;
}

Expr IRSchedule::Fuse(const Expr& block, const std::vector<int>& loops_index) {
  auto result = impl_->Fuse(block, loops_index);
  trace_.Append(ScheduleDesc::Step("FuseWithBlock",
                                   {{"block", std::vector<Expr>({block})}},
                                   {{"loops_index", loops_index}},
                                   {result}));
  return result;
}

void IRSchedule::ComputeAt(const Expr& block,
                           const Expr& loop,
                           bool keep_unit_loops) {
  impl_->ComputeAt(block, loop, keep_unit_loops);
  trace_.Append(ScheduleDesc::Step("ComputeAt",
                                   {{"block", std::vector<Expr>({block})},
                                    {"loop", std::vector<Expr>({loop})}},
                                   {{"keep_unit_loops", keep_unit_loops}},
                                   {}));
}

void IRSchedule::SimpleComputeAt(const Expr& block, const Expr& loop) {
  impl_->SimpleComputeAt(block, loop);
  trace_.Append(ScheduleDesc::Step("SimpleComputeAt",
                                   {{"block", std::vector<Expr>({block})},
                                    {"loop", std::vector<Expr>({loop})}},
                                   {},
                                   {}));
}

void IRSchedule::ReverseComputeAt(const Expr& block,
                                  const Expr& loop,
                                  bool keep_unit_loops) {
  impl_->ReverseComputeAt(block, loop, keep_unit_loops);
  trace_.Append(ScheduleDesc::Step("ReverseComputeAt",
                                   {{"block", std::vector<Expr>({block})},
                                    {"loop", std::vector<Expr>({loop})}},
                                   {{"keep_unit_loops", keep_unit_loops}},
                                   {}));
}

Expr IRSchedule::GetRootBlock(const Expr& expr) const {
  auto result = impl_->GetRootBlock(expr);
  trace_.Append(ScheduleDesc::Step(
      "GetRootBlock", {{"expr", std::vector<Expr>({expr})}}, {}, {result}));
  return result;
}

Expr IRSchedule::CacheRead(const Expr& block,
                           int read_buffer_index,
                           const std::string& memory_type) {
  auto result = impl_->CacheRead(block, read_buffer_index, memory_type);
  trace_.Append(ScheduleDesc::Step(
      "CacheRead",
      {{"block", std::vector<Expr>({block})}},
      {{"read_buffer_index", read_buffer_index}, {"memory_type", memory_type}},
      {result}));
  return result;
}

Expr IRSchedule::CacheWrite(const Expr& block,
                            int write_buffer_index,
                            const std::string& memory_type) {
  auto result = impl_->CacheWrite(block, write_buffer_index, memory_type);
  trace_.Append(ScheduleDesc::Step("CacheWrite",
                                   {{"block", std::vector<Expr>({block})}},
                                   {{"write_buffer_index", write_buffer_index},
                                    {"memory_type", memory_type}},
                                   {result}));
  return result;
}

void IRSchedule::SyncThreads(const Expr& ir_node, bool after_node) {
  impl_->SyncThreads(ir_node, after_node);
  trace_.Append(ScheduleDesc::Step("SyncThreads",
                                   {{"ir_node", std::vector<Expr>({ir_node})}},
                                   {{"after_node", after_node}},
                                   {}));
}

void IRSchedule::SetBuffer(Expr& block,
                           const std::string& memory_type,
                           bool fixed) {
  impl_->SetBuffer(block, memory_type, fixed);
  trace_.Append(
      ScheduleDesc::Step("SetBuffer",
                         {{"block", std::vector<Expr>({block})}},
                         {{"memory_type", memory_type}, {"fixed", fixed}},
                         {}));
}

Expr IRSchedule::Reorder(const std::vector<Expr>& loops) {
  Expr ret = impl_->Reorder(loops);
  trace_.Append(ScheduleDesc::Step("Reorder", {{"loops", loops}}, {}, {ret}));
  return ret;
}

Expr IRSchedule::Reorder(const std::string& block_name,
                         const std::vector<int>& loops_index) {
  Expr ret = impl_->Reorder(block_name, loops_index);
  trace_.Append(ScheduleDesc::Step(
      "ReorderWithName",
      {},
      {{"block_name", block_name}, {"loops_index", loops_index}},
      {ret}));
  return ret;
}

Expr IRSchedule::Reorder(const Expr& block,
                         const std::vector<int>& loops_index) {
  Expr ret = impl_->Reorder(block, loops_index);
  trace_.Append(ScheduleDesc::Step("ReorderWithBlock",
                                   {{"block", std::vector<Expr>({block})}},
                                   {{"loops_index", loops_index}},
                                   {ret}));
  return ret;
}

void IRSchedule::Parallel(const Expr& loop) {
  impl_->Parallel(loop);
  trace_.Append(ScheduleDesc::Step(
      "Parallel", {{"loop", std::vector<Expr>({loop})}}, {}, {}));
}

void IRSchedule::Vectorize(const Expr& loop, int factor) {
  impl_->Vectorize(loop, factor);
  trace_.Append(ScheduleDesc::Step("Vectorize",
                                   {{"loop", std::vector<Expr>({loop})}},
                                   {{"factor", factor}},
                                   {}));
}

void IRSchedule::Unroll(const Expr& loop) {
  impl_->Unroll(loop);
  trace_.Append(ScheduleDesc::Step(
      "Unroll", {{"loop", std::vector<Expr>({loop})}}, {}, {}));
}

void IRSchedule::ComputeInline(const Expr& schedule_block) {
  impl_->ComputeInline(schedule_block);
  trace_.Append(ScheduleDesc::Step(
      "ComputeInline",
      {{"schedule_block", std::vector<Expr>({schedule_block})}},
      {},
      {}));
}

void IRSchedule::ReverseComputeInline(const Expr& schedule_block) {
  impl_->ReverseComputeInline(schedule_block);
  trace_.Append(ScheduleDesc::Step(
      "ReverseComputeInline",
      {{"schedule_block", std::vector<Expr>({schedule_block})}},
      {},
      {}));
}

void IRSchedule::Bind(const Expr& loop, const std::string& thread_axis) {
  impl_->Bind(loop, thread_axis);
  trace_.Append(ScheduleDesc::Step("Bind",
                                   {{"loop", std::vector<Expr>({loop})}},
                                   {{"thread_axis", thread_axis}},
                                   {}));
}

Expr IRSchedule::Rfactor(const Expr& rf_loop, int rf_axis) {
  auto result = impl_->Rfactor(rf_loop, rf_axis);
  trace_.Append(ScheduleDesc::Step("Rfactor",
                                   {{"rf_loop", std::vector<Expr>({rf_loop})}},
                                   {{"rf_axis", rf_axis}},
                                   {result}));
  return result;
}

void IRSchedule::Annotate(const Expr& block,
                          const std::string& key,
                          const attr_t& value) {
  impl_->Annotate(block, key, value);

#define TRACE_ANNOTATE_ITEM(data_type, step_name)               \
  if (absl::holds_alternative<data_type>(value)) {              \
    trace_.Append(ScheduleDesc::Step(                           \
        #step_name,                                             \
        {{"block", std::vector<Expr>({block})}},                \
        {{"key", key}, {"value", absl::get<data_type>(value)}}, \
        {}));                                                   \
    return;                                                     \
  }
  TRACE_ANNOTATE_ITEM(int, AnnotateIntAttr)
  TRACE_ANNOTATE_ITEM(bool, AnnotateBoolAttr)
  TRACE_ANNOTATE_ITEM(float, AnnotateFloatAttr)
  TRACE_ANNOTATE_ITEM(std::string, AnnotateStringAttr)
#undef TRACE_ANNOTATE_ITEM

  LOG(FATAL) << "Value of attribute:" << key << " input unsupported data type";
}

void IRSchedule::Unannotate(Expr& block, const std::string& key) {
  impl_->Unannotate(block, key);
  trace_.Append(ScheduleDesc::Step("Unannotate",
                                   {{"block", std::vector<Expr>({block})}},
                                   {{"key", key}},
                                   {}));
}

void IRSchedule::FlattenLoops(const std::vector<Expr>& loops,
                              const bool force_flat) {
  impl_->FlattenLoops(loops, force_flat);
  trace_.Append(ScheduleDesc::Step("FlattenLoops",
                                   {{"loop", std::vector<Expr>({loops})}},
                                   {{"force_flat", force_flat}},
                                   {}));
}

void IRSchedule::CopyTransformAndLoopInfo(const Expr& block,
                                          const Expr& block_target) {
  impl_->CopyTransformAndLoopInfo(block, block_target);
  // don't support to trace, because we can't ensure both blocks are from the
  // same ModuleExpr
}

void IRSchedule::CopyTransformAndLoopInfo(
    const std::string& block_name, const std::string& block_target_name) {
  impl_->CopyTransformAndLoopInfo(block_name, block_target_name);
  // don't support to trace, because we can't ensure both blocks are from the
  // same ModuleExpr
}

std::vector<Expr> IRSchedule::SamplePerfectTile(
    const Expr& loop,
    int n,
    int max_innermost_factor,
    const std::vector<int>& decision) {
  std::vector<Expr> factors;
  std::vector<int> new_decision;
  if (decision.empty()) {
    factors =
        impl_->SamplePerfectTile(&rand_seed_, loop, n, max_innermost_factor);
    std::transform(factors.begin(),
                   factors.end(),
                   std::back_inserter(new_decision),
                   [](Expr x) { return x.as_int32(); });
  } else {
    new_decision = decision;
    std::transform(decision.begin(),
                   decision.end(),
                   std::back_inserter(factors),
                   [](int x) { return Expr(x); });
  }
  trace_.Append(
      ScheduleDesc::Step("SamplePerfectTile",
                         {{"loop", std::vector<Expr>({loop})}},
                         {{"n", n},
                          {"max_innermost_factor", max_innermost_factor},
                          {"decision", new_decision}},
                         factors));
  return factors;
}

void IRSchedule::TagPostSchedule() {
  trace_.Append(ScheduleDesc::Step("TagPostSchedule", {}, {}, {}));
}

Expr IRSchedule::SampleCategorical(const std::vector<int>& candidates,
                                   const std::vector<float>& probs,
                                   const std::vector<int>& decision) {
  Expr result;
  std::vector<int> new_decision;
  if (decision.empty()) {
    result = impl_->SampleCategorical(&rand_seed_, candidates, probs);
    new_decision.push_back(result.as_int32());
  } else {
    new_decision = decision;
    for (auto ndco : new_decision) {
      result = Expr(ndco);
    }
  }
  trace_.Append(ScheduleDesc::Step("SampleCategorical",
                                   {},
                                   {{"candidates", candidates},
                                    {"probs", probs},
                                    {"decision", new_decision}},
                                   {result}));
  return result;
}

}  // namespace ir
}  // namespace cinn
