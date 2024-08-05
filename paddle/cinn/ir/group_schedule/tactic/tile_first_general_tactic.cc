// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/tile_first_general_tactic.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"

// #include "device_launch_parameters.h"

namespace cinn {
namespace ir {

bool IsInnerThreadSpatialLoopGT(const ScheduleConfig& config, int num) {
  return config.tile_config.spatial_inner_num > num;
}

bool IsReduceBlock(const ScheduleConfig& config, const std::string& block_id) {
  return config.base_info->reduce_tensor_names.count(block_id) > 0;
}

bool HasReduceAxis(const ScheduleConfig& config) {
  return config.base_info->reduce_axis.size() > 0;
}

bool IsWarpReduce(const ScheduleConfig& config) {
  const auto& MatchWarpReduce = cinn::adt::match{
      [&](const ir::NoneReduceMethod&) { return false; },
      [&](const ir::WarpReduceMethod&) { return true; },
      [&](const ir::BlockReduceMethod&) { return false; },
      [&](const ir::DiscreteReduceMethod&) { return false; },
  };
  return std::visit(MatchWarpReduce, config.tile_config.reduce_method);
}

bool UseContinuousDataTile(const ScheduleConfig& config) {
  if (config.base_info->reduce_axis.empty()) {
    return true;
  }
  int64_t min_stride = INT_MAX;
  int64_t min_reduce_stride = INT_MAX;
  int64_t last_axis = 0;
  int64_t last_reduce_axis = 0;
  for (size_t i = 0; i < config.base_info->loop_strides.size(); i++) {
    if (config.base_info->loop_strides[i] < min_stride &&
        config.base_info->loop_strides[i] != 0) {
      min_stride = config.base_info->loop_strides[i];
      last_axis = i;
    }
  }
  for (int64_t axis : config.base_info->reduce_axis) {
    if (config.base_info->loop_strides[axis] < min_reduce_stride) {
      min_reduce_stride = config.base_info->loop_strides[axis];
      last_reduce_axis = axis;
    }
  }
  return last_axis == last_reduce_axis;
}

class TileFirstGeneralTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;
  void ApplyContinuousDataTile(ir::IRSchedule* sch,
                               const std::string& block_id);

  std::string TacticName() const override { return "TileFirstGeneralTactic"; }

 private:
  void AlignToReduceInput(ir::IRSchedule* sch, const std::string& block_id);
  void MergeFlattenAxis(ir::IRSchedule* sch, const std::string& block_id);
  void MergeDiscreteFlattenAxis(ir::IRSchedule* sch,
                                const std::string& block_id);
  void MergeReduceAxis(ir::IRSchedule* sch, const std::string& block_id);
  void SplitSptialInner(ir::IRSchedule* sch, const std::string& block_id);
  void SplitReduceInner(ir::IRSchedule* sch, const std::string& block_id);
  void VariableTypeAssignment(ir::IRSchedule* sch, const std::string& block_id);
  void SetReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void SetDiscreteReduceType(ir::IRSchedule* sch, const std::string& block_id);
  void BindCudaInfo(ir::IRSchedule* sch, const std::string& block_id);
  void Vectorize(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  std::vector<int32_t> vec_spatial_axis_first_;
  std::vector<int32_t> vec_spatial_axis_last_;
  std::vector<int32_t> vec_flatten_axis_;
  std::vector<int32_t> vec_reduce_axis_;
  int reduce_current_axis_{0};
};

void TileFirstGeneralTactic::Init(ScheduleContext* context) {
  context_ = context;
  reduce_current_axis_ =
      IsInnerThreadSpatialLoopGT(context_->config, 1) ? 2 : 1;
  if (context_->config.base_info->is_reduce_all) {
    reduce_current_axis_ = 0;
  }
  // reduce axis have be re-order to last
  vec_flatten_axis_.clear();
  vec_reduce_axis_.clear();
  int32_t reduce_start_idx = context_->config.base_info->data_rank -
                             context_->config.base_info->reduce_axis.size();
  for (int32_t i = 0; i < context_->config.base_info->data_rank; ++i) {
    if (i >= reduce_start_idx) {
      vec_reduce_axis_.push_back(i);
    } else {
      vec_flatten_axis_.push_back(i);
    }
  }
  vec_spatial_axis_first_.clear();
  vec_spatial_axis_last_.clear();

  if (!context_->config.base_info->reduce_axis.empty()) {
    int64_t first_reduce_axis = context_->config.base_info->reduce_axis.front();
    for (auto axis : context_->config.base_info->reduce_axis) {
      if (context->config.base_info->loop_strides[axis] >
          context->config.base_info->loop_strides[first_reduce_axis]) {
        first_reduce_axis = axis;
      }
    }
    for (int32_t i = 0; i < reduce_start_idx; ++i) {
      if (i < first_reduce_axis) {
        vec_spatial_axis_first_.push_back(i);
      } else {
        vec_spatial_axis_last_.push_back(i);
      }
    }
  }
}

void TileFirstGeneralTactic::Apply(ir::IRSchedule* sch,
                                   const std::string& block_id) {
  if (ir::IsReduceInitTensorName(block_id)) return;

  AlignToReduceInput(sch, block_id);
  VLOG(6) << "After AlignToReduceInput on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  if (UseContinuousDataTile(context_->config)) {
    VLOG(4) << "Using ApplyContinuousDataTile";
    ApplyContinuousDataTile(sch, block_id);
    return;
  }

  MergeReduceAxis(sch, block_id);
  VLOG(6) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  MergeDiscreteFlattenAxis(sch, block_id);
  VLOG(6) << "After MergeDiscreteFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitSptialInner(sch, block_id);
  VLOG(6) << "After SplitSptialInner on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SplitReduceInner(sch, block_id);
  VLOG(6) << "After SplitReduceInner on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  BindCudaInfo(sch, block_id);
  VLOG(6) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  VLOG(6) << "nihao";
  // Vectorize(sch, block_id);
  VLOG(6) << "After Vectorize on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  VariableTypeAssignment(sch, block_id);
  VLOG(6) << "After VariableTypeAssignment on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];
  SetDiscreteReduceType(sch, block_id);
}

void TileFirstGeneralTactic::ApplyContinuousDataTile(
    ir::IRSchedule* sch, const std::string& block_id) {
  const auto sp_thread = context_->config.tile_config.warp_num * 32 /
                         context_->config.tile_config.tree_reduce_num;
  const auto sp_loop = context_->config.tile_config.spatial_inner_num;
  const auto rd_thread = context_->config.tile_config.tree_reduce_num;
  VLOG(4) << "ApplyContinuousDataTile sp_thread=" << sp_thread;
  VLOG(4) << "ApplyContinuousDataTile sp_loop=" << sp_loop;
  VLOG(4) << "ApplyContinuousDataTile rd_thread=" << rd_thread;
  VLOG(4) << "ApplyContinuousDataTile vec_flatten_axis: "
          << utils::Join(vec_flatten_axis_, ", ");
  VLOG(4) << "ApplyContinuousDataTile vec_reduce_axis: "
          << utils::Join(vec_reduce_axis_, ", ");

  // Merge reduce axes
  MergeReduceAxis(sch, block_id);
  VLOG(4) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Merge spatial axes
  MergeFlattenAxis(sch, block_id);
  VLOG(4) << "After MergeFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Split spatial axes -> [sp_block, sp_loop, sp_thread]
  int current_reduce_axis = 0;
  if (vec_flatten_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    if (sp_loop > 1 && sp_thread > 1) {
      // [S, R] => [S(-1), S(inner_loop), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop, sp_thread});
      current_reduce_axis = 3;
    } else if (sp_loop > 1 || sp_thread > 1) {
      // [S, R] => [S(-1), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop > 1 ? sp_loop : sp_thread});
      current_reduce_axis = 2;
    } else {
      // [S, R] => [S, R]
      current_reduce_axis = 1;
    }
  }
  VLOG(4) << "After SplitSptial on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Split reduce axes -> [rd_loop, rd_thread]
  if (vec_reduce_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    auto reduce_loop = loops[current_reduce_axis].As<ir::For>();
    // [S..S, R] => [S..S, R(-1), R(thread)]
    sch->Split(loops[current_reduce_axis], {-1, rd_thread});
    VLOG(4) << "Before ReorderReduction on block: [" << block_id
            << "], loop nest:\n"
            << sch->GetModule().GetExprs().front();

    // TODO(lshpku): the Reorder is unneeded if the later FactorizeReduction
    // supports rf_axis=1.
    loops = sch->GetLoops(block_id);
    // [S..S, R(-1), R(thread)] => [S..S, R(thread), R(-1)]
    sch->Reorder({loops[current_reduce_axis + 1], loops[current_reduce_axis]});
    VLOG(4) << "Before FactorizeReduction on block: [" << block_id
            << "], loop nest:\n"
            << sch->GetModule().GetExprs().front();

    if (IsReduceBlock(context_->config, block_id)) {
      loops = sch->GetLoops(block_id);
      sch->FactorizeReduction(loops[current_reduce_axis],
                              /* rf_axis = */ 0,
                              /* with_write_back_block_init = */ false);
    }
  }
  VLOG(4) << "After SplitReduce on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  bool do_reduce_vec = false;
  bool do_flatten_vec = false;

  // Vectorize
  const auto DoVectorize = [&](const std::vector<ir::Expr>& loops) {
    std::vector<size_t> unroll_loops_idx = [&] {
      if (!vec_flatten_axis_.empty() && sp_thread > 1) {
        if (vec_reduce_axis_.empty()) {
          return std::vector<size_t>{current_reduce_axis -
                                     1};  // the last flatten dim
        } else {
          return std::vector<size_t>{
              current_reduce_axis};  // todo: reduce may not done.
        }
      }
      return std::vector<size_t>{current_reduce_axis};
    }();

    const auto DoVec =
        [&](const std::vector<ir::Expr>& loops, int vec_dim, int vec_factor) {
          if (loops.size() > vec_dim &&
              loops[vec_dim].As<ir::For>()->extent.is_constant()) {
            sch->Vectorize(loops[vec_dim], vec_factor);
          }
        };

    // this is used for the homogeneous multi-gpus per node
    // compute the parallel resource
    const auto GetSPCount = [&]() {
      cudaDeviceProp devProp;
      cudaGetDeviceProperties(&devProp, 0);

      int sps = 0;
      int mp = devProp.multiProcessorCount;
      switch (devProp.major) {
        case 7:  // Volta and Turing
          if ((devProp.minor == 0) || (devProp.minor == 5))
            sps = mp * 4;
          else
            printf("Unknown device type\n");
          break;
        case 8:  // Ampere
          if (devProp.minor == 0)
            sps = mp * 4;
          else if (devProp.minor == 6)
            sps = mp * 4;
          else if (devProp.minor == 9)
            sps = mp * 4;  // ada lovelace
          else
            printf("Unknown device type\n");
          break;
        case 9:  // Hopper
          if (devProp.minor == 0)
            sps = mp * 4;
          else
            printf("Unknown device type\n");
          break;
        default:
          printf("Unknown device type\n");
          break;
      }
      return sps;
    };

    const auto getThreadInnerExtent = [&](const std::vector<ir::Expr>& loops) {
      int start_idx = 0;
      if (!vec_flatten_axis_.empty() && sp_thread > 1) {
        if (vec_reduce_axis_.empty()) {
          start_idx = current_reduce_axis - 1;  // the last flatten dim
        } else {
          start_idx = current_reduce_axis - 1;  // todo: reduce may not done.
        }
      }

      // this is used for static extent, but the dynamic extent is
      // not done
      int total_extents = 1;
      for (int loop_idx = start_idx; loop_idx < loops.size(); loop_idx++) {
        if (loops[loop_idx].As<ir::For>()->extent.is_constant()) {
          total_extents *= loops[loop_idx].As<ir::For>()->extent.as_int64();
        }
      }
      return total_extents;
    };

    const auto GetFuncBlocks = [&](const std::vector<ir::Expr>& loops) {
      int num_blocks = 1;
      if (loops[0].As<ir::For>()->extent.is_constant()) {
        num_blocks = loops[0].As<ir::For>()->extent.as_int64();
      }

      return num_blocks;
    };

    const auto GetFuncWarps = [&](const std::vector<ir::Expr>& loops) {
      int num_warps = 1;
      for (int loop_idx = 1; loop_idx < loops.size(); loop_idx++) {
        if (loops[loop_idx].As<ir::For>()->extent.is_constant()) {
          num_warps *= loops[loop_idx].As<ir::For>()->extent.as_int64();
        }
      }

      return (num_warps + 31) / 32;
    };

    int SP_count = GetSPCount();

    int num_blocks = GetFuncBlocks(sch->GetLoops(block_id));

    int num_warps_per_block = GetFuncWarps(sch->GetLoops(block_id));

    int vec_factor = 4;  // for float and int

    float Bef_SMSP_usage =
        (1.0 * num_blocks * num_warps_per_block / SP_count) > 1
            ? 1
            : (1.0 * num_blocks * num_warps_per_block / SP_count);
    float Aft_SMSP_usage =
        (1.0 * num_blocks * num_warps_per_block / vec_factor / SP_count) > 1
            ? 1
            : (1.0 * num_blocks * num_warps_per_block / SP_count);

    bool is_vec = (Aft_SMSP_usage == Bef_SMSP_usage);
    bool is_merge_threads =
        (getThreadInnerExtent(sch->GetLoops(block_id)) == 1l);

    VLOG(-1) << "is_vec: " << is_vec;
    VLOG(-1) << "is_merge_threads: " << is_merge_threads;

    // is_vec = false;
    //    start vectorizing
    if (is_vec) {
      int vec_dim = 0;

      // elementwise
      if (vec_reduce_axis_.empty()) {
        // condition:
        // 1\ sp_loop > 1 && sp_thread > 1
        //    shape: [-1, sp_loop, sp_thread]
        //    is_merge_threads -> sp_loop < factor
        //    if is_merge_threads flase,  then "split [-1,
        //    sp_loop/factor, sp_thread, factor] current_rudece_axis =
        //    4; if is_merge_threads true, then "[-1, sp_thread /
        //    (factor / sp_loop), factor]" current_reduce_axis = 3
        // 2\ sp_loop > 1 || sp_thread > 1
        //    shape: [-1, max(sp_loop, sp_thread)]
        //    is_merge_threads -> always true
        //    split [-1, max(sp_loop, sp_thread) / factor, factor]
        // 3\ else : (sp_loop == 1 && sp_thread == 1)
        //    do nothing
        if (!vec_flatten_axis_.empty()) {
          if (sp_loop > 1 && sp_thread > 1) {
            do_flatten_vec = true;
            bool is_merge_threads = sp_loop < vec_factor;
            if (is_merge_threads) {
              if (vec_factor % sp_loop == 0) {
                // merge the sp_thread with (vec_factor / sp_loop)
                sch->Fuse(block_id, std::vector<int>({1, 2}));
                auto loops = sch->GetLoops(block_id);
                sch->Split(
                    loops[1],
                    std::vector<int>({-1,
                                      (sp_thread / (vec_factor / sp_loop)),
                                      vec_factor}));
                current_reduce_axis = 3;

                loops = sch->GetLoops(block_id);
                sch->Bind(loops[1], "threadIdx.x");
                vec_dim = current_reduce_axis - 1;  // the last flatten dim
                DoVec(sch->GetLoops(block_id), vec_dim, vec_factor);

                if (IsReduceBlock(context_->config, block_id) &&
                    sch->HasBlock(block_id + "_rf")) {
                  DoVec(sch->GetLoops(block_id + "_rf"),
                        vec_dim,
                        vec_factor);  // todo: reduce may have bugs
                }
              }
            } else if (sp_loop % vec_factor == 0) {
              do_flatten_vec = true;
              // fuse the sp_loop and sp_threads, and resplit with [-1,
              // sp_loop/factor, sp_thread, factor]
              sch->Fuse(block_id, std::vector<int>({1, 2}));
              auto loops = sch->GetLoops(block_id);
              sch->Split(
                  loops[1],
                  std::vector<int>(
                      {-1, (sp_loop / vec_factor), sp_thread, vec_factor}));
              current_reduce_axis = 4;

              loops = sch->GetLoops(block_id);
              sch->Bind(loops[1], "threadIdx.x");
              vec_dim = current_reduce_axis - 1;  // the last flatten dim
              DoVec(sch->GetLoops(block_id), vec_dim, vec_factor);

              if (IsReduceBlock(context_->config, block_id) &&
                  sch->HasBlock(block_id + "_rf")) {
                DoVec(sch->GetLoops(block_id + "_rf"),
                      vec_dim,
                      vec_factor);  // todo: reduce may have bugs
              }
            }
          } else if (sp_loop > 1 || sp_thread > 1) {
            // is_merge_threads always be true
            do_flatten_vec = true;

            int thread_dim = sp_loop > 1 ? sp_loop : sp_thread;
            if (thread_dim % vec_factor == 0) {
              auto loops = sch->GetLoops(block_id);
              sch->Split(
                  loops[1],
                  std::vector<int>({-1, thread_dim / vec_factor, vec_factor}));

              current_reduce_axis = 3;

              loops = sch->GetLoops(block_id);
              sch->Bind(loops[1], "threadIdx.x");
              vec_dim = current_reduce_axis - 1;  // the last flatten dim
              DoVec(sch->GetLoops(block_id), vec_dim, vec_factor);

              if (IsReduceBlock(context_->config, block_id) &&
                  sch->HasBlock(block_id + "_rf")) {
                DoVec(sch->GetLoops(block_id + "_rf"),
                      vec_dim,
                      vec_factor);  // todo: reduce may have bugs
              }
            }
          }
        }
      } else {
        // reduce
        //   shape[rd_thread, rd_loop]
        //   condition:
        //    fuse rd_thread, rd_loop
        //    if rd_thread * rd_loop > factor, then split [-1,
        //    rd_thread, factor]
        auto loops = sch->GetLoops(block_id);
        // auto forloop = loops[current_reduce_axis + 1].As<ir::For>();
        // rd_loop > factor, then vectorize
        bool do_reduce_vec = false;

        if ((loops.size() > (current_reduce_axis + 1)) &&
            loops[current_reduce_axis + 1]
                .As<ir::For>()
                ->extent.is_constant() &&
            ((loops[current_reduce_axis + 1]
                  .As<ir::For>()
                  ->extent.type()
                  .is_int(32) &&
              loops[current_reduce_axis + 1].As<ir::For>()->extent.as_int32() >=
                  vec_factor) ||
             (loops[current_reduce_axis + 1]
                  .As<ir::For>()
                  ->extent.type()
                  .is_int(64) &&
              loops[current_reduce_axis + 1].As<ir::For>()->extent.as_int64() >=
                  vec_factor))) {
          do_reduce_vec = true;
        }
        if ((loops.size() > current_reduce_axis) &&
            loops[current_reduce_axis].As<ir::For>()->extent.is_constant() &&
            ((loops[current_reduce_axis].As<ir::For>()->extent.type().is_int(
                  32) &&
              loops[current_reduce_axis].As<ir::For>()->extent.as_int32() >=
                  vec_factor * rd_thread) ||
             (loops[current_reduce_axis].As<ir::For>()->extent.type().is_int(
                  64) &&
              loops[current_reduce_axis].As<ir::For>()->extent.as_int64() >=
                  vec_factor * rd_thread))) {
          do_reduce_vec = true;
        }

        if (do_reduce_vec) {
          sch->Fuse(
              block_id,
              std::vector<int>({current_reduce_axis, current_reduce_axis + 1}));
          auto loops = sch->GetLoops(block_id);
          sch->Split(loops[current_reduce_axis],
                     std::vector<int>({-1, rd_thread, vec_factor}));
          loops = sch->GetLoops(block_id);
          sch->Reorder(
              {loops[current_reduce_axis + 1], loops[current_reduce_axis]});

          loops = sch->GetLoops(block_id);
          sch->Bind(loops[current_reduce_axis], "threadIdx.x");
          vec_dim = current_reduce_axis + 2;  // the last flatten dim
          DoVec(sch->GetLoops(block_id), vec_dim, vec_factor);

          if (IsReduceBlock(context_->config, block_id) &&
              sch->HasBlock(block_id + "_rf")) {
            DoVec(sch->GetLoops(block_id + "_rf"),
                  vec_dim,
                  vec_factor);  // todo: reduce may have bugs
          }
        }
      }

      // if (is_vec) {
      //   int vec_dim = 0;
      //   if (is_merge_threads) {
      //     if (!vec_flatten_axis_.empty() && sp_thread > 1) {
      //       if (vec_reduce_axis_.empty()) {
      //         // split the thread.x with factor
      //         auto split_loops = sch->Split(loops[current_reduce_axis
      //         - 1],
      //                                       std::vector<int>({-1,
      //                                       vec_factor}));

      //        current_reduce_axis++;
      //        vec_dim = current_reduce_axis - 1;  // the last flatten
      //        dim
      //      } else {
      //        vec_dim = current_reduce_axis - 1;  // todo: reduce may
      //        not
      // done.
      //      }
      //    }
      //    DoVec(sch->GetLoops(block_id), vec_dim, vec_factor);
      //    if (IsReduceBlock(context_->config, block_id) &&
      //        sch->HasBlock(block_id + "_rf")) {
      //      DoVec(sch->GetLoops(block_id + "_rf"),
      //            vec_dim,
      //            vec_factor);  // todo: reduce may have bugs
      //    }
      //  } else {
      //    if (!vec_flatten_axis_.empty() && sp_thread > 1) {
      //      if (vec_reduce_axis_.empty()) {
      //        // fuse the sp_loop and sp_threads, and resplit with
      //        [-1, sp_thread,
      //        // sp_loop]
      //        sch->Fuse(block_id, std::vector<int>({1, 2}));
      //        auto loops = sch->GetLoops(block_id);
      //        sch->Split(loops[1], std::vector<int>({-1, sp_loop}));
      //        loops = sch->GetLoops(block_id);
      //        if (sp_loop > vec_factor) {
      //          sch->Split(loops[current_reduce_axis - 1],
      //                     std::vector<int>({-1, vec_factor}));
      //          current_reduce_axis++;
      //        }

      //        loops = sch->GetLoops(block_id);
      //        sch->Bind(loops[1], "threadIdx.x");
      //        vec_dim = current_reduce_axis - 1;  // the last flatten
      //        dim
      //      } else {
      //        vec_dim = current_reduce_axis - 1;  // todo: reduce may
      //        not
      // done.
      //      }
      //      DoVec(sch->GetLoops(block_id), vec_dim, vec_factor);
      //      if (IsReduceBlock(context_->config, block_id) &&
      //          sch->HasBlock(block_id + "_rf")) {
      //        DoVec(sch->GetLoops(block_id + "_rf"),
      //              vec_dim,
      //              vec_factor);  // todo: reduce may have bugs
      //      }
      //    }
      //  }
      //}
    }
  };

  VLOG(-1) << "Start Vectorize";
  DoVectorize(sch->GetLoops(block_id));
  VLOG(6) << "After Vectorize on block: [" << block_id << "], loop nest:\n"
          << sch->GetLoops(block_id)[0];

  // Bind CUDA info
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    std::string sp_axis_type = "threadIdx.y";
    std::string rd_axis_type = "threadIdx.x";
    sch->Bind(loops[0], "blockIdx.x");
    if (!vec_flatten_axis_.empty() && sp_thread > 1) {
      if (vec_reduce_axis_.empty()) {
        if (!do_flatten_vec) {
          sch->Bind(loops[current_reduce_axis - 1], rd_axis_type);
        }
      } else {
        // [S..S, R..R] =>
        // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R..R]
        sch->Bind(loops[current_reduce_axis - 1], sp_axis_type);
      }
    }
    if (!vec_reduce_axis_.empty() && current_reduce_axis > 0) {
      if (!do_reduce_vec) {
        // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R..R] =>
        // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R(threadIdx.x),
        // R(inner_loop)]
        sch->Bind(loops[current_reduce_axis], rd_axis_type);
      }
    }
  };
  DoBind(sch->GetLoops(block_id));
  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    DoBind(sch->GetLoops(block_id + "_rf"));
  }
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  VariableTypeAssignment(sch, block_id);
  SetReduceType(sch, block_id);
}

void TileFirstGeneralTactic::AlignToReduceInput(ir::IRSchedule* sch,
                                                const std::string& block_id) {
  const auto& loop_strides = context_->config.base_info->loop_strides;
  if (loop_strides.empty()) {
    return;
  }

  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  std::vector<int64_t> loop_perm(loops.size());
  std::iota(loop_perm.begin(), loop_perm.end(), 0);

  const auto IsReduce = [&](int64_t axis) {
    auto& reduce_axis = context_->config.base_info->reduce_axis;
    return std::find(reduce_axis.begin(), reduce_axis.end(), axis) !=
           reduce_axis.end();
  };

  std::sort(loop_perm.begin(), loop_perm.end(), [&](int64_t a, int64_t b) {
    if (IsReduce(a) == IsReduce(b)) {
      return loop_strides[a] > loop_strides[b];
    }
    return IsReduce(b);
  });
  VLOG(4) << "loop_perm: " << utils::Join(loop_perm, ", ");

  // Reorder S/R loops seperately, otherwise reduce_init will be de-inlined.
  std::vector<Expr> sp_loops, rd_loops;
  for (auto i : loop_perm) {
    if (IsReduce(i)) {
      rd_loops.push_back(loops[i]);
    } else if (loop_strides[i] != 0) {
      sp_loops.push_back(loops[i]);
    }
  }
  sch->Reorder(sp_loops);
  sch->Reorder(rd_loops);
}

void TileFirstGeneralTactic::MergeFlattenAxis(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (vec_flatten_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_flatten_axis_);
  }
}

void TileFirstGeneralTactic::MergeDiscreteFlattenAxis(
    ir::IRSchedule* sch, const std::string& block_id) {
  // Note: We need to fuse loops from bottom to top,
  // because the loop index will be changed when the upper loops fused.
  if (vec_spatial_axis_last_.size() >= 2) {
    sch->Fuse(block_id, vec_spatial_axis_last_);
  }
  if (vec_spatial_axis_first_.size() >= 2) {
    sch->Fuse(block_id, vec_spatial_axis_first_);
  }
}

void TileFirstGeneralTactic::MergeReduceAxis(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  int32_t max_loop_idx = 0;
  for (int32_t idx : vec_reduce_axis_) {
    max_loop_idx = std::max(max_loop_idx, idx);
    PADDLE_ENFORCE_EQ(idx < loops.size() || loops.size() == 1,
                      true,
                      ::common::errors::InvalidArgument(
                          "The reduce axis should meet: axis's idx < "
                          "loops.size() or loops.size() == 1, but received "
                          "idx= %d ,loops.size() = %d",
                          idx,
                          loops.size()));
  }
  if (max_loop_idx < loops.size() && vec_reduce_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_reduce_axis_);
  }
}

void TileFirstGeneralTactic::SplitSptialInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  if (loops.size() == 3) {
    // [S, S', R] => [S, S'(-1), S'(32), R]
    auto split_loops = sch->Split(loops[1], std::vector<int>({-1, 32}));
    // [S, S'(-1), S'(32), R] => [S, S'(32), R]
    sch->Fuse(block_id, std::vector<int>{0, 1});
  } else if (loops.size() == 2) {
    // [S, R] => [S(-1), S(32), R]
    auto split_loops = sch->Split(loops[0], std::vector<int>({-1, 32}));
  }
}

void TileFirstGeneralTactic::SplitReduceInner(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  // [S(-1), S(32), R] => [S(-1), S(32), R(16), R(-1)]
  sch->Split(loops[2], std::vector<int>{16, -1});

  loops = sch->GetLoops(block_id);
  if (IsReduceBlock(context_->config, block_id)) {
    sch->FactorizeReduction(loops[2],
                            0,
                            /* with_write_back_block_init = */ false);
  }
}

void TileFirstGeneralTactic::VariableTypeAssignment(
    ir::IRSchedule* sch, const std::string& block_id) {
  const auto IsOutputTensor = [&](const std::string& tensor_name) -> bool {
    return context_->config.base_info->direct_output_var_names.count(
               tensor_name) > 0;
  };
  const auto HasConsumers = [&](const ir::Expr& block) -> bool {
    return !ir::analyzer::GetConsumerSBlocks(block, sch->GetRootBlock(block))
                .empty();
  };

  auto block = sch->GetBlock(block_id);
  if (!IsOutputTensor(block_id) && HasConsumers(block)) {
    sch->SetBuffer(block, "local", false);
  }

  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    auto block = sch->GetBlock(block_id + "_rf");
    sch->SetBuffer(block, "local", false);
  }
}

void TileFirstGeneralTactic::SetReduceType(ir::IRSchedule* sch,
                                           const std::string& block_id) {
  if (IsReduceBlock(context_->config, block_id)) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = context_->config.tile_config.reduce_method;
  }
}

void TileFirstGeneralTactic::SetDiscreteReduceType(
    ir::IRSchedule* sch, const std::string& block_id) {
  if (IsReduceBlock(context_->config, block_id)) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = cinn::ir::DiscreteReduceMethod();
  }
}

void TileFirstGeneralTactic::BindCudaInfo(ir::IRSchedule* sch,
                                          const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);

  // [S(-1), S(32), R(16), R(-1)] =>
  // [S(blockIdx.x), S(threadIdx.x), R(threadIdx.y), R(inner_loop)]
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    sch->Bind(loops[1], "threadIdx.x");
    sch->Bind(loops[2], "threadIdx.y");
  };

  DoBind(sch->GetLoops(block_id));

  if (IsReduceBlock(context_->config, block_id) &&
      sch->HasBlock(block_id + "_rf")) {
    auto loops = sch->GetLoops(block_id + "_rf");
    DoBind(sch->GetLoops(block_id + "_rf"));
  }
}

std::unique_ptr<ScheduleTactic> CreateTileFirstGeneralTactic() {
  return std::make_unique<TileFirstGeneralTactic>();
}

}  // namespace ir
}  // namespace cinn
