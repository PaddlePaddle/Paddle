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

#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"

namespace cinn {
namespace ir {

using TileConfig = ScheduleConfig::TileConfig;
using GroupVectorizeInfo = hlir::framework::pir::GroupVectorizeInfo;

using TileConfigMap =
    std::unordered_map<BucketInfo, TileConfig, BucketInfoHash>;

namespace {

const int kMaxNumel = BucketInfo::kMaxNumel;
constexpr int kWarpSize = 32;
constexpr int KMaxWarpSizePerSM = 64;
constexpr int KMaxBlockSizePerSM = 32;
constexpr int KMaxRegistersPerSM = 65536;

int64_t CeilPow2(int64_t n) {
  int64_t pow = 1;
  while (pow < n) {
    pow *= 2;
  }
  return pow;
}

int64_t FloorPow2(int64_t n) {
  int64_t pow = 1;
  while (pow * 2 <= n) {
    pow *= 2;
  }
  return pow;
}

int64_t CeilDiv(int64_t n, int64_t m) { return (n + m - 1) / m; }

int64_t Trim(int64_t n, int64_t min, int64_t max) {
  return std::min(std::max(n, min), max);
}

struct TileConfigCollector {
  void operator()(const BucketInfo& bucket_info,
                  const TileConfig& tile_config) {
    configs_.emplace(bucket_info, tile_config);
  }

  TileConfigMap GetResult() { return configs_; }

 private:
  TileConfigMap configs_;
};

}  // namespace

BucketInfo::BucketInfo(int sp_lower_bound,
                       int sp_upper_bound,
                       int rb_lower_bound,
                       int rb_upper_bound,
                       bool sp_is_dynamic = false,
                       bool rb_is_dynamic = false) {
  if (sp_is_dynamic || sp_lower_bound != 1 || sp_upper_bound != 1) {
    BucketInfo::Dimension sp_dimension(
        sp_lower_bound, sp_upper_bound, "S", sp_is_dynamic);
    this->space.push_back(sp_dimension);
  }
  if (rb_is_dynamic || rb_lower_bound != 1 || rb_upper_bound != 1) {
    BucketInfo::Dimension rb_dimension(
        rb_lower_bound, rb_upper_bound, "R", rb_is_dynamic);
    this->space.push_back(rb_dimension);
  }
  if (this->space.empty()) {
    this->space.emplace_back(1, 1, "S", /* is_dynamic = */ false);
  }
}

BucketInfo::BucketInfo(const std::vector<BucketInfo::Dimension>& dims) {
  for (auto& dim : dims) {
    if (dim.is_dynamic || dim.lower_bound != 1 || dim.upper_bound != 1) {
      this->space.push_back(dim);
    }
  }
  if (this->space.empty()) {
    this->space.emplace_back(1, 1, "S", /* is_dynamic = */ false);
  }
}

bool BucketInfo::operator==(const BucketInfo& other) const {
  if (this->bucket_priority != other.bucket_priority) {
    return false;
  }
  if (this->space.size() != other.space.size()) {
    return false;
  }
  int length = this->space.size();
  for (int i = 0; i < length; i++) {
    if (this->space[i].is_dynamic != other.space[i].is_dynamic ||
        this->space[i].iter_type != other.space[i].iter_type ||
        this->space[i].lower_bound != other.space[i].lower_bound ||
        this->space[i].upper_bound != other.space[i].upper_bound) {
      return false;
    }
  }
  return true;
}

std::string BucketInfo::ToString() const {
  std::stringstream ss;
  ss << "BucketInfo: [";
  for (const auto& dim : space) {
    ss << dim.iter_type << "(" << dim.lower_bound << " - " << dim.upper_bound
       << "), ";
  }
  ss << "]";
  return ss.str();
}

ir::IterSpaceType GetIterSpaceType(
    const std::shared_ptr<FusionGroupInfo>& group_info,
    const std::set<int64_t>& reduce_dim_loc) {
  // Sort loops by their loop strides (if loop strides are known), so that we
  // get the loop index corresponding to the original memory layout.
  std::vector<int> loop_index(group_info->loop_ranges.size());
  std::iota(loop_index.begin(), loop_index.end(), 0);
  if (!group_info->loop_strides.empty()) {
    std::stable_sort(loop_index.begin(), loop_index.end(), [&](int i, int j) {
      return group_info->loop_strides[i] > group_info->loop_strides[j];
    });
  }

  ir::IterSpaceType iter_space_type;
  for (int i : loop_index) {
    int64_t loop_range = group_info->loop_ranges[i];
    if (loop_range == 1) {
      continue;
    }
    std::string iter_type = reduce_dim_loc.count(i) > 0 ? "R" : "S";
    std::string shape_type = loop_range == -1 ? "dynamic" : "static";
    if (iter_space_type.empty() || iter_space_type.back().first != iter_type) {
      iter_space_type.push_back({iter_type, shape_type});
    } else if (shape_type == "dynamic") {
      iter_space_type.back().second = shape_type;
    }
  }

  if (iter_space_type.empty()) {
    iter_space_type.push_back({"S", "static"});
  }
  return iter_space_type;
}

std::shared_ptr<ScheduleConfig::BaseInfo> InitBasicInfo(
    const std::shared_ptr<FusionGroupInfo>& group_info) {
  std::shared_ptr<ScheduleConfig::BaseInfo> base_info =
      std::make_shared<ScheduleConfig::BaseInfo>();
  base_info->reduce_axis = group_info->reduce_axis;
  base_info->loop_ranges = group_info->loop_ranges;
  base_info->loop_strides = group_info->loop_strides;
  base_info->can_apply_grid_reduce = group_info->can_apply_grid_reduce;

  std::set<int64_t> reduce_dim_loc(group_info->reduce_axis.begin(),
                                   group_info->reduce_axis.end());

  base_info->spatial_numel = 1;
  base_info->reduce_numel = 1;
  for (int64_t i = 0; i < base_info->loop_ranges.size(); ++i) {
    if (reduce_dim_loc.count(i)) {
      if (group_info->loop_ranges[i] == -1)
        base_info->has_dynamic_reduce = true;
      base_info->reduce_numel *= group_info->loop_ranges[i];
    } else {
      if (group_info->loop_ranges[i] == -1)
        base_info->has_dynamic_spatial = true;
      base_info->spatial_numel *= group_info->loop_ranges[i];
    }
  }

  base_info->iter_space_type = GetIterSpaceType(group_info, reduce_dim_loc);
  return base_info;
}

namespace {

int CalculateSMsNeeded(int blocks_needed, int max_effective_blocks_per_sm) {
  return CeilDiv(blocks_needed, max_effective_blocks_per_sm);
}

int CalculateMaxEffectiveBlocksPerSM(const SMConfig& sm_config,
                                     int threads_per_block) {
  int max_blocks_per_sm_by_threads =
      sm_config.max_threads_per_sm / threads_per_block;
  return std::min(sm_config.max_blocks_per_sm, max_blocks_per_sm_by_threads);
}

std::pair<int, int> CalculateBlocksAndSMsNeeded(const SMConfig& sm_config,
                                                int block_size,
                                                int blocks_needed) {
  int max_effective_blocks_per_sm =
      CalculateMaxEffectiveBlocksPerSM(sm_config, block_size);
  int sms_needed =
      CalculateSMsNeeded(blocks_needed, max_effective_blocks_per_sm);
  return {max_effective_blocks_per_sm, sms_needed};
}

bool ShouldUpdateWarpNums(int diff_to_fill_sm,
                          int min_diff_to_full_sm,
                          int threads_per_block,
                          int best_warp_nums) {
  return (diff_to_fill_sm < min_diff_to_full_sm) ||
         (diff_to_fill_sm == min_diff_to_full_sm &&
          threads_per_block > best_warp_nums * kWarpSize);
}

// Only proceed with vectorization if SM utilization exceeds 100%
bool CheckSmUtilization(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const SMConfig& sm_config,
    int input_size,
    int block_size) {
  const auto& last_dim = base_info->iter_space_type.back().first;

  if (last_dim != "S" && last_dim != "R") {
    VLOG(5) << "Invalid last_dim in SmUtilization Check: " << last_dim;
    return false;
  }

  int blocks_needed =
      (last_dim == "S") ? CeilDiv(input_size, block_size) : input_size;
  auto [max_effective_blocks_per_sm, sms_needed] =
      CalculateBlocksAndSMsNeeded(sm_config, block_size, blocks_needed);
  float sm_utilization = static_cast<float>(sms_needed) / sm_config.sm_count;

  if (sm_utilization < 1) {
    VLOG(5) << "SM utilization is not sufficient for vectorization: "
            << sm_utilization * 100 << "% (" << sms_needed << "/"
            << sm_config.sm_count << " SMs)";
    return false;
  }
  return true;
}

// By default, warp_nums can be a maximum of 8 (256 threads)
// The Grid value should be divisible by the SM number as much as possible to
// avoid Tail Effect.
int CalculateWarpNums(const SMConfig& sm_config, int total_threads_needed) {
  int best_warp_nums = 8;
  int min_diff_to_full_sm = sm_config.sm_count;

  std::vector<int> thread_configs = {1024, 512, 256};
  for (int threads_per_block : thread_configs) {
    int current_warp_count = threads_per_block / kWarpSize;
    int blocks_needed =
        std::ceil(static_cast<float>(total_threads_needed) / threads_per_block);
    auto [max_effective_blocks_per_sm, sms_needed] =
        CalculateBlocksAndSMsNeeded(
            sm_config, threads_per_block, blocks_needed);

    if (sms_needed <= sm_config.sm_count) return best_warp_nums;
    int remaining_sms = sms_needed % sm_config.sm_count;
    int remaining_blocks = remaining_sms * max_effective_blocks_per_sm;
    int diff_to_fill_sm = std::abs(remaining_blocks - sm_config.sm_count);

    if (remaining_blocks < sm_config.sm_count) {
      if (ShouldUpdateWarpNums(diff_to_fill_sm,
                               min_diff_to_full_sm,
                               threads_per_block,
                               best_warp_nums)) {
        min_diff_to_full_sm = diff_to_fill_sm;
        best_warp_nums = current_warp_count;
      }
    }
  }
  return best_warp_nums;
}

int UpdateWarpNumsInDifferentCase(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const GroupVectorizeInfo& group_vectorize_info,
    int warp_nums) {
  const auto& last_dim = base_info->iter_space_type.back().first;
  if (group_vectorize_info.has_if_else_op && last_dim == "R") {
    warp_nums = Trim(warp_nums, 1, 16);
  } else if (!group_vectorize_info.args_broadcast_axis_info.empty() &&
             last_dim == "S") {
    warp_nums = Trim(warp_nums, 1, 8);
  } else {
    warp_nums = Trim(warp_nums, 1, 32);
  }
  return warp_nums;
}

static bool CheckThreadDimensionCanVectorize(int threads,
                                             int nums,
                                             int factor,
                                             bool is_reduce) {
  const int deal_elements_in_warp = threads * factor;
  if (is_reduce && nums == deal_elements_in_warp) {
    return true;
  }

  if (!is_reduce && nums % deal_elements_in_warp == 0) {
    return true;
  }
  return false;
}

bool ReduceRegionCanVectorize(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const SMConfig& sm_config,
    const int warp_nums,
    const int factor) {
  const int64_t spatial_numel = base_info->spatial_numel;
  const int64_t reduce_numel = base_info->reduce_numel;
  if (warp_nums < 4 && spatial_numel > 1) return false;

  int rd_thread_num = warp_nums * kWarpSize;
  if ((warp_nums > 1 || spatial_numel < warp_nums * 64) &&
      CheckThreadDimensionCanVectorize(
          rd_thread_num, reduce_numel, factor, true) &&
      CheckSmUtilization(
          base_info, sm_config, spatial_numel * factor, rd_thread_num)) {
    return true;
  }
  return false;
}

bool SpatialRegionCanVectorize(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const GroupVectorizeInfo& group_vectorize_info,
    const SMConfig& sm_config,
    const int warp_nums,
    const int factor) {
  const int64_t spatial_numel = base_info->spatial_numel;
  const int64_t reduce_numel = base_info->reduce_numel;
  const int sp_thread_num = kWarpSize * warp_nums;
  if (group_vectorize_info.has_select_op) return false;
  if (CheckThreadDimensionCanVectorize(
          sp_thread_num, spatial_numel, factor, false) &&
      CheckSmUtilization(base_info, sm_config, spatial_numel, sp_thread_num)) {
    return true;
  }
  return false;
}

int CalculateVectorizeTensorRegisterNums(
    const GroupVectorizeInfo& group_vectorize_info,
    const int vectorize_factor) {
  constexpr int register_bits = 32;
  int register_sum = 0;
  for (auto& [tensor_name, tensor] : group_vectorize_info.vetorize_args) {
    if (group_vectorize_info.args_broadcast_axis_info.count(tensor_name))
      continue;
    auto* tensor_ptr = tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor_ptr,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node, but received nullptr."));
    int data_type_bits = tensor_ptr->type().bits();
    int vectorize_data_bits = data_type_bits * vectorize_factor;
    int register_nums = CeilDiv(vectorize_data_bits, register_bits);
    register_sum += register_nums;
  }
  return register_sum;
}
/*
Broadcast tensor will deal with different memory in vectorize situation.
for example, 256 threads in one block situation.
for i in range(1024):
  for j in range(256):
  A[i, 0] = B[0, j]

after vectorize:
A tensor will deal with scalar tensor with only one data type bits register.

B tensor will deal with vectorize tensor or
  local buffer with vectorize factor size memory.
*/
int IsScalarTensorPreload(
    const std::vector<int64_t>& loop_ranges,
    const std::vector<std::vector<bool>> broadcast_axis_infos,
    const int warp_nums,
    const int vectorize_factor) {
  const int threads_deal_elements = warp_nums * kWarpSize * vectorize_factor;
  bool is_scalar_tensor = true;
  for (int i = 0; i < broadcast_axis_infos.size(); i++) {
    int last_dim = broadcast_axis_infos[i].size() - 1;
    int broadcast_nums = 1;
    for (int j = last_dim; j >= 0; j--) {
      if (broadcast_axis_infos[i][j]) {
        broadcast_nums *= loop_ranges[j];
        continue;
      }
      break;
    }
    if (broadcast_nums >= threads_deal_elements) continue;
    is_scalar_tensor = false;
  }
  return is_scalar_tensor;
}

int CalculateBroadcastTensorRegisterNums(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const GroupVectorizeInfo& group_vectorize_info,
    const int vectorize_factor,
    const int warp_nums) {
  // current only support [S, R] and [S] situation.
  // thread parellization only current at last dimension in R or S dimension.
  constexpr int register_bits = 32;
  int register_sum = 0;
  for (auto& [tensor_name, tensor] : group_vectorize_info.vetorize_args) {
    if (!group_vectorize_info.args_broadcast_axis_info.count(tensor_name))
      continue;
    auto* tensor_ptr = tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor_ptr,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in load, but received nullptr."));
    int data_type_bits = tensor_ptr->type().bits();
    int tensor_buffer_size = 1;
    // after deal with Grid dimension tiling, broadcast tensor can be deal with
    // scalar tensor or local buffer.
    if (!IsScalarTensorPreload(
            base_info->loop_ranges,
            group_vectorize_info.args_broadcast_axis_info.at(tensor_name),
            warp_nums,
            vectorize_factor)) {
      tensor_buffer_size *= vectorize_factor;
    }
    int vectorize_data_bits = tensor_buffer_size * data_type_bits;
    int register_nums = CeilDiv(vectorize_data_bits, register_bits);
    register_sum += register_nums;
  }
  return register_sum;
}

int CalculateOtherRegisterNums(const GroupVectorizeInfo& group_vectorize_info,
                               const int vectorize_factor) {
  constexpr int register_bits = 32;
  int register_sum = 0;
  for (auto& [tensor_name, tensor] : group_vectorize_info.vetorize_args) {
    if (group_vectorize_info.args_broadcast_axis_info.count(tensor_name))
      continue;
    auto* tensor_ptr = tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        tensor_ptr,
        ::common::errors::InvalidArgument(
            "Expected _Tensor_ node in load, but received nullptr."));
    int data_type_bits = tensor_ptr->type().bits();
    int vectorize_data_bits = vectorize_factor * data_type_bits;
    int register_nums = CeilDiv(vectorize_data_bits, register_bits);
    register_sum += register_nums;
    break;
  }
  register_sum = group_vectorize_info.has_if_else_op ? register_sum : 0;
  return register_sum;
}

bool RegisterNumsLimitedCheckInCTACanApplyVectorize(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const GroupVectorizeInfo& group_vectorize_info,
    const int vectorize_factor,
    const int warp_nums) {
  int thread_register_occupy_sum = 0;
  int vectorize_tensor_registers = CalculateVectorizeTensorRegisterNums(
      group_vectorize_info, vectorize_factor);
  VLOG(5) << "calculate vectorize tensor registers is : "
          << vectorize_tensor_registers << "\n";
  thread_register_occupy_sum += vectorize_tensor_registers;
  int broadcast_tensor_thread_registers = CalculateBroadcastTensorRegisterNums(
      base_info, group_vectorize_info, vectorize_factor, warp_nums);
  thread_register_occupy_sum += broadcast_tensor_thread_registers;
  VLOG(5) << "calculate broadcast tensor registers is : "
          << broadcast_tensor_thread_registers << "\n";
  // other register size is according to human experience.
  int other_register_occupy_sum =
      CalculateOtherRegisterNums(group_vectorize_info, vectorize_factor);
  thread_register_occupy_sum += other_register_occupy_sum;
  VLOG(5) << "calculate other registers is : " << other_register_occupy_sum
          << "\n";
  int max_blocks_per_sm =
      Trim(CeilDiv(KMaxWarpSizePerSM, warp_nums), 1, KMaxBlockSizePerSM);
  int best_register_nums_per_thread =
      KMaxRegistersPerSM / max_blocks_per_sm / warp_nums / kWarpSize;
  VLOG(5) << "calculatet thread register occupy sum is : "
          << thread_register_occupy_sum
          << ", best register nums per thread is : "
          << best_register_nums_per_thread;
  // if has if_else_op, overflow register have more influence in CTA.
  if (group_vectorize_info.has_if_else_op &&
      thread_register_occupy_sum >= best_register_nums_per_thread) {
    return false;
  }
  best_register_nums_per_thread = best_register_nums_per_thread * 1.3;
  if (thread_register_occupy_sum >= best_register_nums_per_thread) {
    return false;
  }

  return true;
}

bool AvoidTensorAddrCalculateWithLongInVectorize(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info) {
  int64_t iter_space_size = 1;
  for (auto range : base_info->loop_ranges) {
    iter_space_size *= range;
  }
  if (iter_space_size >= 2147483647ll) return false;
  return true;
}

bool CheckPerformanceLimitInVectorize(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const GroupVectorizeInfo& group_vectorize_info,
    const int vectorize_factor,
    const int warp_nums) {
  if (!RegisterNumsLimitedCheckInCTACanApplyVectorize(
          base_info, group_vectorize_info, vectorize_factor, warp_nums)) {
    VLOG(5) << "According to the limit of register, current schedule block "
               "can't enable vectorize!";
    return false;
  }

  if (!AvoidTensorAddrCalculateWithLongInVectorize(base_info)) {
    VLOG(5) << "avoid tensor addr calculate with long in vectorize, current "
               "schedule block "
               "can't enable vectorize!";
    return false;
  }
  return true;
}

}  // namespace

TileConfigMap BuildVectorizeConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const GroupVectorizeInfo& group_vectorize_info,
    const common::Target& target) {
  if (!group_vectorize_info.meet_vectorization_condition) return {};

  // TileFirstGeneralTactic apply Vectorize current
  // only support [S, R] and [S]
  const int64_t iters_dim = base_info->iter_space_type.size();
  const auto& last_dim = base_info->iter_space_type.back().first;
  if (!((iters_dim == 2 && last_dim == "R") ||
        (iters_dim == 1 && last_dim == "S"))) {
    return {};
  }

  const std::vector<int> vectorize_factors{4, 2};
  int64_t spatial_numel = base_info->spatial_numel;
  int64_t reduce_numel = base_info->reduce_numel;
  int sp_thread_num = 1;
  int rd_thread_num = 1;
  int warp_nums = 1;
  int vectorize_factor = 1;
  bool can_vectorize = false;
  bool is_sm_fully_utilized = true;
  ReduceMethod reduce_method = NoneReduceMethod();
  SMConfig sm_config(target.get_max_threads_per_sm(),
                     target.get_max_blocks_per_sm(),
                     target.get_multi_processor_count());

  // Reduce Region
  if (last_dim == "R") {
    for (auto factor : vectorize_factors) {
      vectorize_factor = factor;
      const int elements_in_warp = kWarpSize * vectorize_factor;
      warp_nums = CeilDiv(reduce_numel, elements_in_warp);
      warp_nums = Trim(warp_nums, 1, 32);
      rd_thread_num = warp_nums * kWarpSize;
      if (ReduceRegionCanVectorize(
              base_info, sm_config, warp_nums, vectorize_factor)) {
        can_vectorize = true;
        reduce_method = BlockReduceMethod();
        break;
      }
    }
  } else if (iters_dim == 1 && last_dim == "S") {  // Spatial Region
    for (auto factor : vectorize_factors) {
      vectorize_factor = factor;
      const int elements_in_warp = kWarpSize * vectorize_factor;
      warp_nums = CeilDiv(spatial_numel, elements_in_warp);
      int max_warp_nums =
          CalculateWarpNums(sm_config, spatial_numel / vectorize_factor);
      warp_nums = Trim(warp_nums, 1, max_warp_nums);
      sp_thread_num = kWarpSize * warp_nums;
      if (SpatialRegionCanVectorize(base_info,
                                    group_vectorize_info,
                                    sm_config,
                                    warp_nums,
                                    vectorize_factor)) {
        can_vectorize = true;
        break;
      }
    }
  }

  warp_nums =
      UpdateWarpNumsInDifferentCase(base_info, group_vectorize_info, warp_nums);

  if (can_vectorize &&
      !CheckPerformanceLimitInVectorize(
          base_info, group_vectorize_info, vectorize_factor, warp_nums)) {
    can_vectorize = false;
  }

  if (!can_vectorize) {
    return {};
  }

  base_info->can_apply_vectorize = true;
  int64_t sp_upper_bound = base_info->spatial_numel > 1 ? kMaxNumel : 1;
  int64_t rd_upper_bound = base_info->reduce_numel > 1 ? kMaxNumel : 1;
  BucketInfo bucket_info{1, sp_upper_bound, 1, rd_upper_bound};
  TileConfig tile_config{warp_nums,
                         /* tree_reduce_num = */ rd_thread_num,
                         /* grid_reduce_num = */ 1,
                         /* spatial_inner_num = */ 1,
                         /* vectorize_factor = */ vectorize_factor,
                         /* reduce_inner_num = */ -1,
                         reduce_method};
  return {{bucket_info, tile_config}};
}

std::pair<int64_t, int64_t> FindBestReduceBlockThreadNum(int64_t reduce_numel,
                                                         int64_t sp_thread_num,
                                                         int64_t rd_thread_num,
                                                         int64_t sp_block_num,
                                                         int sm_count) {
  float max_sm_occupacy = 0.0f;
  int64_t best_rd_block_num = 1;
  int64_t best_rd_thread_num = rd_thread_num;

  // The basic principle for selecting rd_block_num is to choose the largest
  // possible value as long as the total number of blocks (rd_block_num *
  // sp_block_num) doesn't exceed the SM count.
  //
  // However, if the SM count is not a perfect multiple of sp_block_num, we may
  // get an underutilized rd_block_num. To solve this problem, we use a factor
  // to split the rd_thread_num in change of more available blocks. In this way,
  // we would have a larger range to choose an rd_block_num that better fits
  // into the available blocks.
  //
  // For example, if sm_count = 80 and sp_block_num = 64, different factors and
  // their occupancy are:
  //   factor  avail_blocks  sp_blocks  rd_blocks  all_blocks  occupancy
  //        1            80         64          1          64        80%
  //        2           160         64          2         128        80%
  //        4           320         64          5         320       100%
  // Therefore, the best factor = 4 and the best rd_block_num = 5.
  //
  // Note: a max factor of 4 should be sufficient in most cases.
  for (int factor = 1; factor <= 4; factor *= 2) {
    if (factor > rd_thread_num) break;
    int64_t new_rd_thread_num = rd_thread_num / factor;
    int64_t avail_blocks_per_sm = 1024 / (sp_thread_num * new_rd_thread_num);
    int64_t avail_blocks = sm_count * avail_blocks_per_sm;

    // First, assign all remaining available blocks to rd_block_num.
    int64_t rd_block_num = avail_blocks / sp_block_num;

    // To constrain the cost of grid-level synchronization, rd_block_num should
    // not exceed the SM count.
    rd_block_num = Trim(rd_block_num, 1, sm_count);

    // To compensate for the block launching cost, we also require that the
    // reduce inner loops be at least two times of the rd_block_num, and be at
    // least 32. The constraints can be written as:
    //   rd_inner_num * rd_block_num * rd_thread_num = reduce_numel  (Cond.0)
    //   rd_inner_num >= rd_block_num * 2                            (Cond.1)
    //   rd_inner_num >= 32                                          (Cond.2)
    int64_t remain_reduce_numel = CeilDiv(reduce_numel, new_rd_thread_num);
    int64_t limit_cond_1 = std::sqrt((remain_reduce_numel + 1) / 2.0);
    int64_t limit_cond_2 = CeilDiv(remain_reduce_numel, 32);
    int64_t limit = std::min(limit_cond_1, limit_cond_2);
    if (limit < rd_block_num) {
      rd_block_num = std::max(limit, int64_t(1));
    }

    // Find the best rd_block/thread_num with the highest SM occupacy.
    float sm_occupacy =
        static_cast<float>(sp_block_num * rd_block_num) / avail_blocks;
    if (sm_occupacy > max_sm_occupacy) {
      max_sm_occupacy = sm_occupacy;
      best_rd_block_num = rd_block_num;
      best_rd_thread_num = new_rd_thread_num;
    }
  }

  return {best_rd_block_num, best_rd_thread_num};
}

TileConfigMap BuildPureStaticShapeConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const GroupVectorizeInfo& vectorize_info,
    const common::Target& target) {
  const auto& last_dim = base_info->iter_space_type.back().first;
  const int sm_count = target.get_multi_processor_count();
  int64_t spatial_numel = base_info->spatial_numel;
  int64_t reduce_numel = base_info->reduce_numel;
  ReduceMethod reduce_method = NoneReduceMethod();

  // Try to use vectorization first
  auto config_map = BuildVectorizeConfig(base_info, vectorize_info, target);
  if (!config_map.empty()) return std::move(config_map);

  // 1. Allocate spatial/reduce threads
  // Principals:
  //   1) The low 32 threads are assigned to the last dimension to ensure
  //      coalesced memory access.
  //   2) The remaining threads are assigned to either the reduce or spatial
  //      dimension, based on which dimension is the bottleneck.
  int64_t sp_thread_num = 1;
  int64_t rd_thread_num = 1;
  if (last_dim == "R") {
    rd_thread_num = 32;
    int64_t remain_reduce_numel = CeilDiv(reduce_numel, 32);
    if ((remain_reduce_numel <= 8 && spatial_numel > 1) ||
        (spatial_numel > remain_reduce_numel * 128)) {
      sp_thread_num = Trim(spatial_numel, 1, 8);
      reduce_method = WarpReduceMethod();
    } else {
      rd_thread_num *= Trim(remain_reduce_numel, 1, 32);
      reduce_method = BlockReduceMethod();
    }
  } else {  // last_dim == "S"
    sp_thread_num = 32;
    int64_t remain_spatial_numel = CeilDiv(spatial_numel, 32);
    if (reduce_numel <= 16) {
      sp_thread_num *= Trim(remain_spatial_numel, 1, 8);
    } else {
      rd_thread_num = Trim(reduce_numel, 1, 16);
      reduce_method = DiscreteReduceMethod();
    }
  }
  spatial_numel = CeilDiv(spatial_numel, sp_thread_num);

  // 2. Allocate grid reduce blocks
  // Principals:
  //   1) Choose the largest reduce block number as long as the total number of
  //      blocks (rd_block * sp_block) doesn't exceed the SM count.
  //   2) Do not allocate too many reduce blocks when reduce_numel is small.
  int64_t rd_block_num = 1;
  if (base_info->can_apply_grid_reduce) {
    int64_t sp_block_num = std::max(spatial_numel, int64_t(1));
    std::pair<int64_t, int64_t> res = FindBestReduceBlockThreadNum(
        reduce_numel, sp_thread_num, rd_thread_num, sp_block_num, sm_count);
    rd_block_num = res.first;
    rd_thread_num = res.second;
  }

  // 3. Allocate spatial inner loops
  // Principals:
  //   1) Choose an appropriate spatial inner loop number so that the spatial
  //      block number had better not exceed 4 times of SM count.
  //   2) Loops can only be assigned to either reduce or spatial, otherwise the
  //      index expression will be complex.
  int64_t sp_inner_num = [&]() -> int64_t {
    int64_t rd_inner_num = CeilDiv(reduce_numel, rd_block_num * rd_thread_num);
    if (rd_inner_num > 1) {
      return 1;
    }
    int64_t expected = spatial_numel / (sm_count * 4);
    return CeilPow2(Trim(expected, 1, 4));
  }();

  int64_t sp_upper_bound = base_info->spatial_numel > 1 ? kMaxNumel : 1;
  int64_t rd_upper_bound = base_info->reduce_numel > 1 ? kMaxNumel : 1;
  int64_t warp_num = Trim(sp_thread_num * rd_thread_num / 32, 1, 32);
  BucketInfo bucket_info{1, sp_upper_bound, 1, rd_upper_bound};
  TileConfig tile_config{warp_num,
                         /* tree_reduce_num = */ rd_thread_num,
                         /* grid_reduce_num = */ rd_block_num,
                         /* spatial_inner_num = */ sp_inner_num,
                         /* vectorize_factor = */ 1,
                         /* reduce_inner_num = */ -1,
                         reduce_method};
  return {{bucket_info, tile_config}};
}

TileConfigMap BuildStaticSpatialConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  const auto& last_dim = base_info->iter_space_type.back().first;
  const int sm_count = target.get_multi_processor_count();
  const int64_t spatial_numel = base_info->spatial_numel;
  const int64_t min_loops = 4;

  TileConfigCollector collector;
  // { sp_lower, sp_upper, rb_lower, rb_upper },
  // { warp_num, tree_reduce, grid_reduce, spatial_inner, reduce_method }

  if (last_dim == "R") {
    int64_t sp_block_num = std::max(spatial_numel, int64_t(1));
    int64_t rd_block_num = FloorPow2(sm_count / sp_block_num);

    collector({1, kMaxNumel, 1, 2048},
              {8, 256, 1, 1, 1, -1, BlockReduceMethod()});

    if (rd_block_num > 1 && base_info->can_apply_grid_reduce) {
      int64_t rd_threshold = rd_block_num * min_loops * 1024;
      collector({1, kMaxNumel, 2049, rd_threshold},
                {32, 1024, 1, 1, 1, -1, BlockReduceMethod()});
      collector({1, kMaxNumel, rd_threshold + 1, kMaxNumel},
                {32, 1024, rd_block_num, 1, 1, -1, BlockReduceMethod()});
    } else {
      collector({1, kMaxNumel, 2049, kMaxNumel},
                {32, 1024, 1, 1, 1, -1, BlockReduceMethod()});
    }

  } else {  // last_dim == "S"
    int64_t sp_block_num = std::max(CeilDiv(spatial_numel, 32), int64_t(1));
    int64_t rd_block_num = FloorPow2(sm_count / sp_block_num);

    if (rd_block_num > 1 && base_info->can_apply_grid_reduce) {
      int64_t rd_threshold = rd_block_num * min_loops * 16;
      collector({1, kMaxNumel, 1, rd_threshold},
                {16, 16, 1, 1, 1, -1, DiscreteReduceMethod()});
      collector({1, kMaxNumel, rd_threshold + 1, kMaxNumel},
                {16, 16, rd_block_num, 1, 1, -1, DiscreteReduceMethod()});
    } else {
      collector({1, kMaxNumel, 1, kMaxNumel},
                {16, 16, 1, 1, 1, -1, DiscreteReduceMethod()});
    }
  }

  return collector.GetResult();
}

TileConfigMap BuildStaticReduceConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  const auto& last_dim = base_info->iter_space_type.back().first;

  TileConfigCollector collector;
  // { sp_lower, sp_upper, rd_lower, rd_upper },
  // { warp, rd_thread, rd_block, sp_inner, vec_factor, rd_inner, rd_method }

  if (last_dim == "R") {
    if (base_info->reduce_numel <= 256) {
      int64_t spatial_inner_num = 256 / CeilPow2(base_info->reduce_numel);
      collector({1, kMaxNumel, 1, 256},
                {8, 32, 1, spatial_inner_num, 1, -1, WarpReduceMethod()});
    } else if (base_info->reduce_numel <= 2048) {
      int64_t reduce_block = CeilDiv(base_info->reduce_numel, 256) * 256;
      int64_t warp_num = reduce_block / 256;
      int64_t reduce_inner_num = 8;
      int64_t tree_reduce_num = reduce_block / reduce_inner_num;
      collector({1, kMaxNumel, 257, 2048},
                {warp_num, tree_reduce_num, 1, 1, 1, -1, BlockReduceMethod()});
    } else {
      collector({1, kMaxNumel, 2049, kMaxNumel},
                {32, 1024, 1, 1, 1, -1, BlockReduceMethod()});
    }
  } else {  // last_dim == "S"
    if (base_info->reduce_numel == 1) {
      collector({1, 1023, 1, 1}, {-1, 1, 1, 1, 1, -1, NoneReduceMethod()});
      collector({1024, kMaxNumel, 1, 1},
                {32, 1, 1, 4, 1, -1, NoneReduceMethod()});
    } else if (base_info->reduce_numel <= 16) {
      collector({1, kMaxNumel, 1, 1}, {8, 1, 1, 1, 1, -1, NoneReduceMethod()});
    } else {
      collector({1, kMaxNumel, 1, 1},
                {16, 16, 1, 1, 1, -1, DiscreteReduceMethod()});
    }
  }

  return collector.GetResult();
}

TileConfigMap BuildDynamicShapeConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  const auto& last_dim = base_info->iter_space_type.back().first;

  TileConfigCollector collector;
  // { sp_lower, sp_upper, rd_lower, rd_upper },
  // { warp, rd_thread, rd_block, sp_inner, vec_factor, rd_inner, rd_method }

  if (last_dim == "R") {
    collector({1, kMaxNumel, 1, 256}, {8, 32, 1, 1, 1, 8, WarpReduceMethod()});
    collector({1, kMaxNumel, 257, 2048},
              {8, 256, 1, 1, 1, 8, BlockReduceMethod()});
    collector({1, kMaxNumel, 2049, kMaxNumel},
              {32, 1024, 1, 1, 1, -1, BlockReduceMethod()});
  } else {  // last_dim == "S"
    collector({1, kMaxNumel, 1, kMaxNumel},
              {16, 16, 1, 1, 1, -1, DiscreteReduceMethod()});
  }
  return collector.GetResult();
}

std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>
CombineBaseInfoAndConfig(
    const TileConfigMap& config_map,
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info) {
  std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash> combined;
  for (const auto& bucket_config : config_map) {
    ScheduleConfig sch_config{base_info, std::move(bucket_config.second)};
    combined.insert({std::move(bucket_config.first), std::move(sch_config)});
  }
  return combined;
}

std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>
BuildScheduleConfig(const std::shared_ptr<FusionGroupInfo>& group_info,
                    const common::Target& target) {
  std::shared_ptr<ScheduleConfig::BaseInfo> base_info =
      InitBasicInfo(group_info);
  if (!base_info->has_dynamic_reduce && !base_info->has_dynamic_spatial) {
    VLOG(6) << "Building static spatial and static reduce config.";
    return CombineBaseInfoAndConfig(
        BuildPureStaticShapeConfig(
            base_info, group_info->vectorize_info, target),
        base_info);
  } else if (base_info->has_dynamic_reduce && !base_info->has_dynamic_spatial) {
    VLOG(6) << "Building static spatial and dynamic reduce config.";
    return CombineBaseInfoAndConfig(BuildStaticSpatialConfig(base_info, target),
                                    base_info);
  } else if (!base_info->has_dynamic_reduce && base_info->has_dynamic_spatial) {
    VLOG(6) << "Building dynamic spatial and static reduce config.";
    return CombineBaseInfoAndConfig(BuildStaticReduceConfig(base_info, target),
                                    base_info);
  } else {  // (base_info->has_dynamic_reduce && base_info->has_dynamic_spatial)
    VLOG(6) << "Building dynamic spatial and dynamic reduce config.";
    return CombineBaseInfoAndConfig(BuildDynamicShapeConfig(base_info, target),
                                    base_info);
  }
}

}  // namespace ir
}  // namespace cinn
