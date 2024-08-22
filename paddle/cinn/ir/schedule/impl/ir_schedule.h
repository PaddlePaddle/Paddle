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
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/schedule/schedule_base.h"
// #include "paddle/cinn/ir/schedule/utils/ir_schedule_util.h"
#include "paddle/cinn/ir/utils/ir_copy.h"

PD_DECLARE_int32(cinn_error_message_level);

namespace cinn {
namespace ir {

/**
 * A struct helps to implement dynamic shape Schedule primitives.
 */
class DyScheduleImpl : public ScheduleBase {
 public:
  DyScheduleImpl() = delete;
  explicit DyScheduleImpl(const ModuleExpr& module_expr,
                          bool debug_flag = false,
                          utils::ErrorMessageLevel err_msg_level =
                              utils::ErrorMessageLevel::kGeneral)
      : ScheduleBase(module_expr, false, err_msg_level) {}
  explicit DyScheduleImpl(ModuleExpr&& module_expr)
      : ScheduleBase(std::move(module_expr)) {}

  void MergeExprs();
  bool HasBlock(const std::string& block_name) const;
  std::vector<Expr> GetLoops(const Expr& block) const;
  std::vector<Expr> GetLoops(const std::string& block_name) const;
  std::vector<Expr> GetAllBlocks() const;
  std::vector<Expr> GetChildBlocks(const Expr& expr) const;
  Expr GetBlock(const std::string& block_name) const;
  std::vector<Expr> Split(const Expr& loop, const std::vector<int>& factors);
  std::vector<Expr> Split(const Expr& loop, const std::vector<Expr>& factors);
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
  Expr FactorizeReduction(const Expr& rf_loop,
                          int rf_axis,
                          bool with_write_back_block_init = true);
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
};

}  // namespace ir
}  // namespace cinn
