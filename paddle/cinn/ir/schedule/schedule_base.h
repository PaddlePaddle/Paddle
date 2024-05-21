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

#pragma once
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/utils/error.h"
#include "paddle/cinn/utils/random_engine.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

PD_DECLARE_int32(cinn_error_message_level);

namespace cinn {
namespace ir {

struct BroadcastInfo {
  std::vector<int64_t> broadcast_axes;
  std::vector<int64_t> output_shape;
  std::vector<symbol::DimExpr> output_dim_expr;

  bool with_constrain{false};
  bool first_broadcast{false};
  bool full_broadcast{false};
  std::string op_name;

  bool split_first{false};
  std::vector<std::pair<int, std::vector<int>>> split_info;
};

/**
 * A struct representing a module that contains Expr. This struct is only used
 * in Schedule process.
 */
class ModuleExpr {
 public:
  ModuleExpr() = default;
  ModuleExpr(const ModuleExpr& mod_expr) = default;
  ModuleExpr(ModuleExpr&& mod_expr) = default;

  ModuleExpr& operator=(const ModuleExpr& mod_expr) = default;

  explicit ModuleExpr(const std::vector<Expr>& exprs) : exprs_(exprs) {}
  explicit ModuleExpr(std::vector<Expr>&& exprs) : exprs_(std::move(exprs)) {}

  //! Get all the Expr in this ModuleExpr.
  std::vector<Expr> GetExprs() { return exprs_; }

  std::vector<Expr> GetExprs() const { return exprs_; }

  void SetExprs(const std::vector<Expr>& exprs) { exprs_ = exprs; }

 private:
  //! Exprs stored in ModuleExpr. Each one is an AST, representing a computation
  //! kernel.
  std::vector<Expr> exprs_;
};

/**
 * Define the interface for scheduling primitives,
 * with subclasses DyScheduleImpl and StScheduleImpl.
 */
class ScheduleBase {
 public:
  ScheduleBase() = delete;
  explicit ScheduleBase(const ModuleExpr& module_expr,
                        bool debug_flag = false,
                        utils::ErrorMessageLevel err_msg_level =
                            utils::ErrorMessageLevel::kGeneral)
      : module_expr_(module_expr), debug_flag_(debug_flag) {
    err_msg_level_ = static_cast<utils::ErrorMessageLevel>(
        FLAGS_cinn_error_message_level || static_cast<int>(err_msg_level));
  }
  explicit ScheduleBase(ModuleExpr&& module_expr)
      : module_expr_(std::move(module_expr)) {}

  static std::unique_ptr<ScheduleBase> Make(
      const ModuleExpr& module_expr,
      bool debug_flag = false,
      utils::ErrorMessageLevel err_msg_level =
          utils::ErrorMessageLevel::kGeneral,
      bool is_dynamic = false);

  static std::unique_ptr<ScheduleBase> Make(ModuleExpr&& module_expr,
                                            bool is_dynamic = false);

  void SetDebugFlag(bool debug_flag) { debug_flag_ = debug_flag; }

  const ModuleExpr& GetModule() const { return module_expr_; }

  void SetExprs(const std::vector<Expr>& exprs) {
    module_expr_.SetExprs(exprs);
  }

  virtual void MergeExprs() = 0;
  virtual bool HasBlock(const std::string& block_name) const = 0;
  virtual std::vector<Expr> GetLoops(const Expr& block) const = 0;
  virtual std::vector<Expr> GetLoops(const std::string& block_name) const = 0;
  virtual std::vector<Expr> GetAllBlocks() const = 0;
  virtual std::vector<Expr> GetChildBlocks(const Expr& expr) const = 0;
  virtual Expr GetBlock(const std::string& block_name) const = 0;

  virtual std::vector<Expr> Split(const Expr& loop,
                                  const std::vector<int>& factors) = 0;
  virtual std::vector<Expr> Split(const Expr& loop,
                                  const std::vector<Expr>& factors) = 0;
  virtual std::vector<Expr> SamplePerfectTile(
      utils::LinearRandomEngine::StateType* rand_seed,
      const Expr& loop,
      int n,
      int max_innermost_factor) = 0;
  virtual Expr Fuse(const std::vector<Expr>& loops) = 0;
  virtual Expr Fuse(const std::string& block_name,
                    const std::vector<int>& loops_index) = 0;
  virtual Expr Fuse(const Expr& block, const std::vector<int>& loops_index) = 0;
  virtual void ComputeAt(const Expr& block,
                         const Expr& loop,
                         bool keep_unit_loops) = 0;
  virtual void SimpleComputeAt(const Expr& block, const Expr& loop) = 0;
  virtual void ReverseComputeAt(const Expr& block,
                                const Expr& loop,
                                bool keep_unit_loops) = 0;
  virtual Expr GetRootBlock(const Expr& expr) const = 0;
  virtual Expr CacheRead(const Expr& block,
                         int read_buffer_index,
                         const std::string& memory_type) = 0;
  virtual Expr CacheWrite(const Expr& block,
                          int write_buffer_index,
                          const std::string& memory_type) = 0;
  virtual void SyncThreads(const Expr& ir_node, bool after_node = true) = 0;
  virtual void SetBuffer(Expr& block,  // NOLINT
                         const std::string& memory_type,
                         bool fixed = false) = 0;
  virtual Expr Reorder(const std::vector<Expr>& loops) = 0;
  virtual Expr Reorder(const std::string& block_name,
                       const std::vector<int>& loops_index) = 0;
  virtual Expr Reorder(const Expr& block,
                       const std::vector<int>& loops_index) = 0;
  virtual DeviceAPI GetDeviceAPI() const = 0;
  virtual void MutateForType(const Expr& loop,
                             ForType for_type,
                             int factor = -1) = 0;
  virtual void Parallel(const Expr& loop) = 0;
  virtual void Vectorize(const Expr& loop, int factor) = 0;
  virtual void Unroll(const Expr& loop) = 0;
  virtual void ComputeInline(const Expr& schedule_block) = 0;
  virtual void ReverseComputeInline(const Expr& schedule_block) = 0;
  virtual void Bind(const Expr& loop, const std::string& thread_axis) = 0;
  virtual Expr Rfactor(const Expr& rf_loop, int rf_axis) = 0;
  virtual Expr FactorizeReduction(const Expr& rf_loop,
                                  int rf_axis,
                                  bool with_write_back_block_init = true) = 0;
  virtual Expr AddUnitLoop(const Expr& block) const = 0;
  virtual void Annotate(const Expr& block,
                        const std::string& key,
                        const attr_t& value) = 0;
  virtual void Unannotate(Expr& block, const std::string& key) = 0;  // NOLINT
  virtual void FlattenLoops(const std::vector<Expr>& loops,
                            const bool force_flat = false) = 0;
  virtual void CopyTransformAndLoopInfo(const Expr& block,
                                        const Expr& block_target) = 0;
  virtual void CopyTransformAndLoopInfo(
      const std::string& block_name, const std::string& block_target_name) = 0;
  virtual Expr SampleCategorical(
      utils::LinearRandomEngine::StateType* rand_seed,
      const std::vector<int>& candidates,
      const std::vector<float>& probs) = 0;

 protected:
  void Replace(const Expr& src_sref, const Expr& tgt_stmt);

  ModuleExpr module_expr_;
  bool debug_flag_{false};
  utils::ErrorMessageLevel err_msg_level_ = utils::ErrorMessageLevel::kGeneral;
};

}  // namespace ir
}  // namespace cinn
