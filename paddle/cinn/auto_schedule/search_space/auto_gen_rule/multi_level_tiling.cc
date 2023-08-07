// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/multi_level_tiling.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/cinn/auto_schedule/analysis/analyze_ir.h"
#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/ir/utils/ir_printer.h"

namespace cinn {
namespace auto_schedule {

MultiLevelTiling::MultiLevelTiling(const common::Target& target,
                                   const Config& config)
    : AutoGenRule(target), config_(config) {
  for (int i = 0; i < config_.tile_struct.size(); ++i) {
    if (config_.tile_struct[i] == 'S') {
      s_indices_.push_back(i);
    } else if (config_.tile_struct[i] == 'R') {
      r_indices_.push_back(i);
    } else {
      CHECK(false) << "Illegal tiling structure string";
    }
  }
}

bool MultiLevelTiling::MeetCondition(
    const ir::ScheduleBlockRealize& sche_block_realize) const {
  return NeedsMultiLevelTiling(sche_block_realize);
}

RuleApplyType MultiLevelTiling::Init(ir::IRSchedule* ir_schedule) {
  ir_schedule_ = ir_schedule;
  all_block_realizes_ = ir_schedule_->GetAllBlocks();
  applicable_indices_.clear();
  num_applicable_ = 0;
  for (size_t i = 0; i < all_block_realizes_.size(); ++i) {
    ir::ScheduleBlockRealize* sche_block_realize =
        all_block_realizes_[i].As<ir::ScheduleBlockRealize>();
    AnalyzeScheduleBlockReadWriteBuffer(
        sche_block_realize->schedule_block.As<ir::ScheduleBlock>());
    if (MeetCondition(*sche_block_realize)) {
      ++num_applicable_;
      applicable_indices_.push_back(i);
    }
  }

  return num_applicable_ > 0 ? RuleApplyType::kApplyAndPruneOtherRules
                             : RuleApplyType::kCannotApply;
}

void MultiLevelTiling::Apply(int index) {
  CHECK(ir_schedule_ != nullptr) << "Run MultiLevelTiling::Apply without Init";
  CHECK(num_applicable_ > 0 && applicable_indices_.size() == num_applicable_)
      << "MultiLevelTiling::Apply pre-condition doesn't meet";
  CHECK(index >= 0 && num_applicable_ > index)
      << "Invalid index for MultiLevelTiling::Apply, the index needs 0 <= "
         "index && index < NumberApplicable(), "
      << "Currently index = " << index
      << ",  NumberApplicable() = " << num_applicable_;

  int apply_index = applicable_indices_[index];
  std::string block_name = all_block_realizes_[apply_index]
                               .As<ir::ScheduleBlockRealize>()
                               ->schedule_block.As<ir::ScheduleBlock>()
                               ->name;
  Expr block_expr = all_block_realizes_[apply_index];
  ApplyTiling(ir_schedule_, block_expr);
  block_expr = ir_schedule_->GetBlock(block_name);
  ApplyCacheRead(ir_schedule_, block_expr);
  block_expr = ir_schedule_->GetBlock(block_name);
  ApplyCacheWrite(ir_schedule_, block_expr);

  VLOG(4) << "Returning the result of MultiLevelTiling";
  return;
}

std::string MultiLevelTiling::GetRuleName() const { return "MultiLevelTiling"; }

RuleApplyType MultiLevelTiling::AnalyseApplyType(
    SearchState state, const std::string& block_name) const {
  Expr block_expr = state->ir_schedule.GetBlock(block_name);
  auto* block_realize = block_expr.As<ir::ScheduleBlockRealize>();
  CHECK(block_realize) << "stmt is not a ScheduleBlockRealize:" << block_expr;
  AnalyzeScheduleBlockReadWriteBuffer(
      block_realize->schedule_block.As<ir::ScheduleBlock>());

  return NeedsMultiLevelTiling(*block_realize)
             ? RuleApplyType::kApplyAndPruneOtherRules
             : RuleApplyType::kCannotApply;
}

std::vector<SearchState> MultiLevelTiling::ApplyOnBlock(
    SearchState state, const std::string& block_name) {
  SearchState new_state = state.Copy();
  ir::IRSchedule* ir_sch = &new_state->ir_schedule;
  Expr block_expr = ir_sch->GetBlock(block_name);
  ApplyTiling(ir_sch, block_expr);
  block_expr = ir_sch->GetBlock(block_name);
  ApplyCacheRead(ir_sch, block_expr);
  block_expr = ir_sch->GetBlock(block_name);
  ApplyCacheWrite(ir_sch, block_expr);

  VLOG(4) << "Returning the result of MultiLevelTiling";
  return {new_state};
}

void MultiLevelTiling::ApplyTiling(ir::IRSchedule* ir_schedule,
                                   ir::Expr& block_expr) {
  ir::ScheduleBlockRealize* sche_block_realize =
      block_expr.As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sche_block =
      sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
  tile_loops_.clear();
  tile_loops_.resize(config_.tile_struct.size());
  std::vector<Expr> for_exprs = ir_schedule->GetLoops(block_expr);

  VLOG(5) << "The number of loops to split in MultiLevelTiling is "
          << for_exprs.size();
  for (int i = for_exprs.size() - 1; i >= 0; --i) {
    ir::For* ir_for = for_exprs[i].As<ir::For>();
    VLOG(6) << "Applying Split for MultiLevelTiling on: " << Expr(ir_for);
    const std::vector<int>* idx = nullptr;
    if (sche_block->iter_vars[i]->is_reduce_axis) {
      idx = &r_indices_;
    } else {
      idx = &s_indices_;
    }  // TODO(zhhsplendid): support more iterator variable types

    int extent = ir_for->extent.as_int32();  // maybe int64?

    int num_split = idx->size();
    if (num_split > 1) {
      std::vector<Expr> tile_split_factor =
          ir_schedule->SamplePerfectTile(Expr(ir_for), num_split, 64);
      std::vector<Expr> splited =
          ir_schedule->Split(Expr(ir_for), tile_split_factor);
      VLOG(6) << "Finish Split for MultiLevelTiling on above loop";
      for (int j = 0; j < num_split; ++j) {
        tile_loops_[idx->at(j)].push_back(splited[j]);
      }
    } else {
      tile_loops_[idx->at(0)].push_back(for_exprs[i]);
    }
  }
  VLOG(5) << "Finish Split in MultiLevelTiling, before Reorder.";

  // Have to GetLoops again because Split can change Block Expr(s)
  for_exprs = ir_schedule->GetLoops(sche_block->name);
  std::unordered_map<std::string, int> loop_var_name_to_idx;
  for (int i = 0; i < for_exprs.size(); ++i) {
    loop_var_name_to_idx[for_exprs[i].As<ir::For>()->loop_var->name] = i;
  }
  CHECK(loop_var_name_to_idx.size() == for_exprs.size())
      << "Loops contain duplicate loop var names after split";

  std::vector<Expr> splited_loops;
  for (auto& t : tile_loops_) {
    std::reverse(t.begin(), t.end());
    for (auto& tile_loop_expr : t) {
      const ir::For* tile_loop = tile_loop_expr.As<ir::For>();
      CHECK(tile_loop) << "tiles store non For Expr";
      int idx = loop_var_name_to_idx[tile_loop->loop_var->name];
      splited_loops.push_back(for_exprs[idx]);
    }
  }

  Expr reordered_expr = ir_schedule->Reorder(splited_loops);
  VLOG(5) << "Finish Reorder in MultiLevelTiling, now do Fuse and Binding on "
             "the main loop chain";

  int num_binds = std::min(config_.bind_axis.size(), tile_loops_.size());
  for (int i = 0; i < num_binds; ++i) {
    loop_var_name_to_idx.clear();
    for_exprs = ir_schedule->GetLoops(sche_block->name);
    for (int j = 0; j < for_exprs.size(); ++j) {
      loop_var_name_to_idx[for_exprs[j].As<ir::For>()->loop_var->name] = j;
    }
    CHECK(loop_var_name_to_idx.size() == for_exprs.size())
        << "Loops contain duplicate loop var names before Fusion";

    // Some loops extent may exceed the limited max factor (For example,
    // exceed the limit number of CUDA threads), here we check whether
    // the fused loop extent, which is the production of extends of loops
    // to be fused, is less or equal to the max factor.
    //
    // If yes, we fuse those loops and bind the fused loop
    // If no, we bind the first loop whose extent is less than the factor.
    int extent_prod = 1;
    int first_idx_less_than_max_factor = -1;
    for (int j = 0; j < tile_loops_[i].size(); ++j) {
      const ir::For* tile_loop = tile_loops_[i][j].As<ir::For>();
      CHECK(tile_loop) << "tiles store non For Expr";
      int idx = loop_var_name_to_idx[tile_loop->loop_var->name];
      tile_loops_[i][j] = for_exprs[idx];
      int extent = tile_loop->extent.as_int32();  // maybe int64?
      extent_prod *= extent;
      if (first_idx_less_than_max_factor == -1 && extent <= max_factor_) {
        first_idx_less_than_max_factor = idx;
      }
    }

    if (extent_prod <= max_factor_) {
      Expr fused = ir_schedule->Fuse(tile_loops_[i]);
      ir_schedule->Bind(fused, config_.bind_axis[i]);
    } else if (first_idx_less_than_max_factor != -1) {
      ir_schedule->Bind(for_exprs[first_idx_less_than_max_factor],
                        config_.bind_axis[i]);
    }
  }

  VLOG(5) << "Do Fuse and Binding on the non-main loop chains";
  Expr sche_block_top_loop = ir_schedule->GetLoops(sche_block->name)[0];

  if (reordered_expr.As<ir::Block>()) {
    for (Expr& top_loop : reordered_expr.As<ir::Block>()->stmts) {
      if (top_loop != sche_block_top_loop) {
        std::vector<Expr> scan_loop_blocks = ir_schedule->GetAllBlocks();
        Expr other_loop_chain_schedule;
        for (Expr& block : scan_loop_blocks) {
          std::vector<Expr> loop_chain = ir_schedule->GetLoops(block);
          if (loop_chain[0] == top_loop) {
            other_loop_chain_schedule = block;
            break;
          }
        }
        if (!other_loop_chain_schedule.defined()) {
          LOG(WARNING) << "Has non-main loop chain, but not corresponding "
                          "ScheduleBlock in MultiLevelTiling";
          continue;
        }

        std::string other_loop_schedule_name =
            other_loop_chain_schedule.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name;
        VLOG(6) << "Found other_loop_schedule_name = "
                << other_loop_schedule_name;
        int fuse_index = 0;
        for (int i = 0; i < num_binds; ++i) {
          for_exprs = ir_schedule->GetLoops(other_loop_schedule_name);

          // Some loops extent may exceed the limited max factor (For example,
          // exceed the limit number of CUDA threads), here we check whether
          // the fused loop extent, which is the production of extends of loops
          // to be fused, is less or equal to the max factor.
          //
          // If yes, we fuse those loops and bind the fused loop
          // If no, we bind the first loop whose extent is less than the factor.
          int extent_prod = 1;
          int first_idx_less_than_max_factor = -1;
          for (int j = 0; j < tile_loops_[i].size(); ++j) {
            int extent =
                for_exprs[fuse_index + j].As<ir::For>()->extent.as_int32();
            extent_prod *= extent;
            if (first_idx_less_than_max_factor == -1 && extent <= max_factor_) {
              first_idx_less_than_max_factor = fuse_index + j;
            }
          }
          if (extent_prod <= max_factor_) {
            std::vector<Expr> loops_to_fuse(
                for_exprs.begin() + fuse_index,
                for_exprs.begin() + fuse_index + tile_loops_[i].size());
            Expr fused = ir_schedule->Fuse(loops_to_fuse);
            ir_schedule->Bind(fused, config_.bind_axis[i]);
            fuse_index += 1;
          } else if (first_idx_less_than_max_factor != -1) {
            ir_schedule->Bind(for_exprs[first_idx_less_than_max_factor],
                              config_.bind_axis[i]);
            fuse_index += tile_loops_[i].size();
          }
        }
      }
    }
  }
}

void MultiLevelTiling::ApplyCacheRead(ir::IRSchedule* ir_schedule,
                                      ir::Expr& block_expr) {
  ir::ScheduleBlockRealize* sch_block_realize =
      block_expr.As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sch_block =
      sch_block_realize->schedule_block.As<ir::ScheduleBlock>();
  std::string block_name = sch_block->name;

  // Analyze which buffers can be cached
  std::vector<int> read_buffer_indexes;
  for (int i = 0; i < sch_block->read_buffers.size(); ++i) {
    bool is_read_write = false;
    for (int j = 0; j < sch_block->write_buffers.size(); ++j) {
      if (sch_block->read_buffers[i] == sch_block->write_buffers[j]) {
        is_read_write = true;
        break;
      }
    }
    if (!is_read_write) {
      read_buffer_indexes.push_back(i);
    }
  }

  // Schedule
  for (int read_buffer_index : read_buffer_indexes) {
    for (int level : config_.read_cache_levels) {
      // 1.find target loop
      const auto loops = tile_loops_.at(level - 1);
      if (loops.size() == 0) {
        continue;
      }

      // 2.Do CacheRead and get the cache block
      ir::Expr cache_block = ir_schedule->CacheRead(
          block_expr, read_buffer_index, config_.read_cache_memory_type);
      std::string cache_block_name =
          cache_block.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>()
              ->name;

      std::string target_for_loop_name =
          loops.back().As<ir::For>()->loop_var->name;

      // 3.Place the cache_block under target_for_loop
      // The original block expr is invalid after the CacheRead schedule,
      // so we reacquire the block expr after the schedule according to the
      // block name
      block_expr = ir_schedule->GetBlock(block_name);
      std::vector<Expr> for_exprs = ir_schedule->GetLoops(block_expr);
      for (const Expr& for_expr : for_exprs) {
        if (for_expr.As<ir::For>()->loop_var->name.find(target_for_loop_name) !=
            std::string::npos) {
          ir_schedule->ComputeAt(cache_block, for_expr, true);
          break;
        }
      }

      // 4.Threads under the same block cooperative fetch data from global
      // memory.
      Expr new_cache_block = ir_schedule->GetBlock(cache_block_name);
      auto cache_block_loops = ir_schedule->GetLoops(new_cache_block);
      std::vector<std::string> compute_at_extra_var = utils::Split(
          absl::get<std::string>(new_cache_block.As<ir::ScheduleBlockRealize>()
                                     ->schedule_block.As<ir::ScheduleBlock>()
                                     ->attrs.at("compute_at_extra_var")),
          ",");
      std::vector<Expr> buffer_loops;
      // int nthreads = 1;
      for (const Expr& for_expr : cache_block_loops) {
        if (std::find(compute_at_extra_var.begin(),
                      compute_at_extra_var.end(),
                      for_expr.As<ir::For>()->loop_var->name) !=
            compute_at_extra_var.end()) {
          buffer_loops.push_back(for_expr);
        }
      }
      auto fused_buffer_loop = ir_schedule->Fuse(buffer_loops);
      // TODO(BiynXu): Implement vectorize fetching data and pass in vector
      // length
      ir_schedule->Annotate(ir_schedule->GetBlock(cache_block_name),
                            ir::attr::cooperative_process,
                            0);
    }
  }
}

void MultiLevelTiling::ApplyCacheWrite(ir::IRSchedule* ir_schedule,
                                       ir::Expr& block_expr) {
  ir::Expr cache_block =
      ir_schedule->CacheWrite(block_expr, 0, config_.write_cache_memory_type);

  for (int level : config_.write_cache_levels) {
    const auto loops = tile_loops_.at(level - 1);
    if (loops.size() == 0) {
      continue;
    }
    std::string target_for_loop_name =
        loops.back().As<ir::For>()->loop_var->name;
    // Because the block name is changed in CacheWrite, we need to calculate the
    // derived name according to the logic of CacheWrite and find the loop
    // structure according to the derived name.
    const std::string original_block_name =
        block_expr.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>()
            ->name;
    const std::string derivative_block_name = original_block_name + "_" +
                                              config_.write_cache_memory_type +
                                              "_temp_buffer";
    std::vector<Expr> for_exprs = ir_schedule->GetLoops(derivative_block_name);
    for (const Expr& for_expr : for_exprs) {
      if (for_expr.As<ir::For>()->loop_var->name.find(target_for_loop_name) !=
          std::string::npos) {
        ir_schedule->ReverseComputeAt(
            ir_schedule->GetBlock(original_block_name), for_expr, true);
      }
    }

    const std::string reduce_init_block_name =
        original_block_name + "__reduce_init";
    for_exprs = ir_schedule->GetLoops(derivative_block_name);
    for (const Expr& for_expr : for_exprs) {
      if (for_expr.As<ir::For>()->loop_var->name.find(target_for_loop_name) !=
              std::string::npos &&
          ir_schedule->HasBlock(reduce_init_block_name)) {
        ir_schedule->SimpleComputeAt(
            ir_schedule->GetBlock(reduce_init_block_name), for_expr);
      }
    }
  }
}

const std::unordered_map<common::Target::Arch, MultiLevelTiling::Config>
    MultiLevelTiling::kConfigs{
        {common::Target::Arch::NVGPU,
         MultiLevelTiling::Config{
             /*bind_axis*/ std::vector<std::string>{"blockIdx.x",
                                                    "threadIdx.x"},
             /*tile_struct*/ std::string("SSSRRSRS"),
             /*read_cache_memory_type*/ std::string("shared"),
             /*read_cache_levels*/ std::vector<int>{4},
             /*write_cache_memory_type*/ std::string("local"),
             /*write_cache_levels*/ std::vector<int>{3},
         }},
        {common::Target::Arch::X86,
         MultiLevelTiling::Config{
             /*bind_axis*/ std::vector<std::string>{},
             /*tile_struct*/ std::string("SSRSRS"),
             /*read_cache_memory_type*/ std::string("local"),
             /*read_cache_levels*/ std::vector<int>{3},
             /*write_cache_memory_type*/ std::string("local"),
             /*write_cache_levels*/ std::vector<int>{2},
         }}};

}  // namespace auto_schedule
}  // namespace cinn
