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

#include "paddle/cinn/runtime/flags.h"

#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <thread>
#include <unordered_set>

#include "paddle/cinn/common/target.h"
#include "paddle/common/flags.h"

#ifdef CINN_WITH_CUDNN
PD_DEFINE_bool(
    cinn_cudnn_deterministic,
    false,
    "Whether allow using an autotuning algorithm for convolution "
    "operator. The autotuning algorithm may be non-deterministic. If "
    "true, the algorithm is deterministic.");
#endif

using ::paddle::flags::BoolFromEnv;
using ::paddle::flags::DoubleFromEnv;
using ::paddle::flags::Int32FromEnv;
using ::paddle::flags::Int64FromEnv;
using ::paddle::flags::StringFromEnv;

PD_DEFINE_string(cinn_x86_builtin_code_root,
                 StringFromEnv("FLAGS_cinn_x86_builtin_code_root", ""),
                 "");

PD_DEFINE_string(cinn_nvcc_cmd_path,
                 StringFromEnv("FLAGS_cinn_nvcc_cmd_path",
                               "/usr/local/cuda/bin"),
                 "Setting nvcc default path!");

PD_DEFINE_int32(cinn_parallel_compile_thread,
                Int32FromEnv("FLAGS_cinn_parallel_compile_thread",
                             (std::thread::hardware_concurrency() >> 1)),
                "How much thread the parallel compile used.");

PD_DEFINE_bool(cinn_use_op_fusion,
               BoolFromEnv("FLAGS_cinn_use_op_fusion", true),
               "Whether to use op fusion pass.");

PD_DEFINE_bool(general_fusion_merge_pass,
               BoolFromEnv("FLAGS_general_fusion_merge_pass", true),
               "Whether to use general fusion_merge pass.");

PD_DEFINE_bool(cinn_new_group_scheduler,
               BoolFromEnv("FLAGS_cinn_new_group_scheduler", false),
               "Whether to use new group scheduler.");

PD_DEFINE_bool(cinn_bucket_compile,
               BoolFromEnv("FLAGS_cinn_bucket_compile", false),
               "Whether to enable bucket compile for dynamic shape.");

PD_DEFINE_bool(group_schedule_tiling_first,
               BoolFromEnv("FLAGS_group_schedule_tiling_first", false),
               "Whether to enable new group scheduler tiling first strategy.");

PD_DEFINE_bool(cinn_use_common_subexpression_elimination,
               BoolFromEnv("FLAGS_cinn_use_common_subexpression_elimination",
                           false),
               "Whether to use common subexpression elimination pass.");

PD_DEFINE_string(
    cinn_custom_call_deny_ops,
    StringFromEnv("FLAGS_cinn_custom_call_deny_ops", ""),
    "a blacklist of op are denied by MarkCustomCallOps pass, separated by ;");

PD_DEFINE_bool(cinn_enable_map_expr,
               BoolFromEnv("FLAGS_cinn_enable_map_expr", false),
               "It controls whether to use cinn with map_expr");

PD_DEFINE_bool(cinn_enable_map_expr_schedule,
               BoolFromEnv("FLAGS_cinn_enable_map_expr_schedule", false),
               "It controls whether to schedule by map_expr");

PD_DEFINE_bool(cinn_enable_map_expr_inline,
               BoolFromEnv("FLAGS_cinn_enable_map_expr_inline", false),
               "It controls whether to inline by map_expr");

PD_DEFINE_bool(cinn_enable_map_expr_dynamic_shape,
               BoolFromEnv("FLAGS_cinn_enable_map_expr_dynamic_shape", false),
               "It controls whether map_expr uses dynamic_shape");

PD_DEFINE_bool(cinn_enable_map_expr_index_detail,
               BoolFromEnv("FLAGS_cinn_enable_map_expr_index_detail", false),
               "It controls whether to display datail tensor index");

PD_DEFINE_bool(
    cinn_use_custom_call,
    BoolFromEnv("FLAGS_cinn_use_custom_call", true),
    "Whether to use custom_call for ops with external_api registered");

PD_DEFINE_bool(cinn_use_fill_constant_folding,
               BoolFromEnv("FLAGS_cinn_use_fill_constant_folding", false),
               "Whether use the FillConstantFolding pass.");

PD_DEFINE_string(cinn_check_fusion_accuracy_pass,
                 StringFromEnv("FLAGS_cinn_check_fusion_accuracy_pass", ""),
                 "Check the correct of fusion kernels, if the results not "
                 "satisfied 'allclose(rtol=1e-05f, atol=1e-08f)', "
                 "report error and exited.");

PD_DEFINE_bool(cinn_use_cuda_vectorize,
               BoolFromEnv("FLAGS_cinn_use_cuda_vectorize", false),
               "Whether use cuda vectroize on schedule config");

PD_DEFINE_bool(use_reduce_split_pass,
               BoolFromEnv("FLAGS_use_reduce_split_pass", false),
               "Whether use reduce split pass.");

PD_DEFINE_bool(cinn_use_dense_merge_pass,
               BoolFromEnv("FLAGS_cinn_use_dense_merge_pass", false),
               "Whether use dense merge pass.");

PD_DEFINE_bool(
    nvrtc_compile_to_cubin,
    BoolFromEnv("FLAGS_nvrtc_compile_to_cubin", false),
    "Whether nvrtc compile cuda source into cubin instead of ptx (only "
    "works after cuda-11.1).");

PD_DEFINE_bool(cinn_compile_with_nvrtc,
               BoolFromEnv("FLAGS_cinn_compile_with_nvrtc", true),
               "Whether nvrtc compile cuda source with nvrtc(default nvcc).");

PD_DEFINE_bool(
    cinn_nvrtc_cubin_with_fmad,
    BoolFromEnv("FLAGS_cinn_nvrtc_cubin_with_fmad", true),
    "Whether nvrtc enables fmad when compile to cubin. This flag only works "
    "when FLAGS_nvrtc_compile_to_cubin=true. Fmad is the cuda speed up "
    "technique which contract fp mulitplication and addition/subtraction into "
    "multiply-add operation. It may result in different fp precision.");

// FLAGS for performance analysis and accuracy debug
PD_DEFINE_bool(cinn_sync_run,
               BoolFromEnv("FLAGS_cinn_sync_run", false),
               "Whether sync all devices after each instruction run, which is "
               "used for debug.");

PD_DEFINE_string(
    cinn_self_check_accuracy,
    StringFromEnv("FLAGS_cinn_self_check_accuracy", ""),
    "Whether self-check accuracy after each instruction run, which "
    "is used for debug.");

PD_DEFINE_int64(
    cinn_self_check_accuracy_num,
    Int64FromEnv("FLAGS_cinn_self_check_accuracy_num", 0L),
    "Set self-check accuracy print numel, which is used for debug.");

PD_DEFINE_string(
    cinn_fusion_groups_graphviz_dir,
    StringFromEnv("FLAGS_cinn_fusion_groups_graphviz_dir", ""),
    "Specify the directory path of dot file of graph, which is used "
    "for debug.");

PD_DEFINE_string(
    cinn_source_code_save_path,
    StringFromEnv("FLAGS_cinn_source_code_save_path", ""),
    "Specify the directory path of generated source code, which is "
    "used for debug.");

PD_DEFINE_string(
    cinn_dump_group_lowered_func,
    StringFromEnv("FLAGS_cinn_dump_group_lowered_func", ""),
    "Specify the path for dump lowered functions by group, which is "
    "used for debug.");

PD_DEFINE_string(
    cinn_dump_group_source_code,
    StringFromEnv("FLAGS_cinn_dump_group_source_code", ""),
    "Specify the path for dump source code by group, which is used for debug.");

PD_DEFINE_string(
    cinn_dump_group_ptx,
    StringFromEnv("FLAGS_cinn_dump_group_ptx", ""),
    "Specify the path for dump ptx by group, which is used for debug.");

PD_DEFINE_string(
    cinn_dump_group_instruction,
    StringFromEnv("FLAGS_cinn_dump_group_instruction", ""),
    "Specify the path for dump instruction by group, which is used for debug.");

PD_DEFINE_string(cinn_pass_visualize_dir,
                 StringFromEnv("FLAGS_cinn_pass_visualize_dir", ""),
                 "Specify the directory path of pass visualize file of graph, "
                 "which is used for debug.");

PD_DEFINE_bool(cinn_runtime_display_debug_info,
               false,
               "Whether to display debug information in runtime");

PD_DEFINE_bool(enable_auto_tuner,
               BoolFromEnv("FLAGS_enable_auto_tuner", false),
               "Whether enable auto tuner.");

PD_DEFINE_bool(auto_schedule_use_cost_model,
               BoolFromEnv("FLAGS_auto_schedule_use_cost_model", true),
               "Whether to use cost model in auto schedule, this is an "
               "on-developing flag and it will be removed when "
               "cost model is stable.");

PD_DEFINE_bool(
    enhance_vertical_fusion_with_recompute,
    BoolFromEnv("FLAGS_enhance_vertical_fusion_with_recompute", true),
    "Whether to enhance check logic on vertical fusion with recompute");

PD_DEFINE_bool(verbose_function_register,
               BoolFromEnv("FLAGS_verbose_function_register", false),
               "Whether to verbose function regist log. This will only work if "
               "CINN build with flag -DWITH_DEBUG=ON.");

PD_DEFINE_int32(
    cinn_profiler_state,
    Int32FromEnv("FLAGS_cinn_profiler_state", -1),
    "Specify the ProfilerState by Int in CINN, 0 for kDisabled, 1 for "
    "kCPU, 2 for kCUDA, 3 for kAll, default 0.");

PD_DEFINE_int32(cinn_error_message_level,
                Int32FromEnv("FLAGS_cinn_error_message_level", 0),
                "Specify the level of printing error message in the schedule."
                "0 means short, 1 means detailed.");

PD_DEFINE_double(cinn_infer_model_version,
                 DoubleFromEnv("FLAGS_cinn_infer_model_version", 2.0),
                 "Paddle has different model format in inference model. We use "
                 "a flag to load different versions.");

PD_DEFINE_bool(cinn_use_cutlass,
               BoolFromEnv("FLAGS_cinn_use_cutlass", false),
               "Whether to use cutlass kernels");

PD_DEFINE_string(cinn_convert_static_dim_to_dynamic_dim,
                 StringFromEnv("FLAGS_cinn_convert_static_dim_to_dynamic_dim",
                               ""),
                 "A test flag whether to convert static dim to dynamic, e.g.: "
                 "FLAGS_cinn_convert_static_dim_to_dynamic_dim=128:s0,299:s1");

PD_DEFINE_string(cinn_convert_dynamic_dim_to_static_dim,
                 StringFromEnv("FLAGS_cinn_convert_dynamic_dim_to_static_dim",
                               ""),
                 "A test flag whether to convert dynamic to static dim, e.g.: "
                 "FLAGS_cinn_convert_dynamic_dim_to_static_dim=s0:128,s1:299");

namespace cinn {
namespace runtime {

bool CheckStringFlagTrue(const std::string& flag) {
  // from gflag FlagValue::ParseFrom:
  // https://github.com/gflags/gflags/blob/master/src/gflags.cc#L292
  static const std::unordered_set<std::string> kTrue = {
      "1", "t", "true", "y", "yes", "T", "True", "TRUE", "Y", "yes"};
  return kTrue.count(flag);
}

bool CheckStringFlagFalse(const std::string& flag) {
  // from gflag FlagValue::ParseFrom:
  // https://github.com/gflags/gflags/blob/master/src/gflags.cc#L292
  static const std::unordered_set<std::string> kFalse = {
      "0", "f", "false", "n", "no", "F", "False", "FALSE", "N", "No", "NO"};
  return flag.empty() || kFalse.count(flag);
}

void SetCinnCudnnDeterministic(bool state) {
#ifdef CINN_WITH_CUDNN
  FLAGS_cinn_cudnn_deterministic = state;
#else
  LOG(WARNING) << "CINN is compiled without cuDNN, this api is invalid!";
#endif
}

bool GetCinnCudnnDeterministic() {
#ifdef CINN_WITH_CUDNN
  return FLAGS_cinn_cudnn_deterministic;
#else
  LOG(FATAL) << "CINN is compiled without cuDNN, this api is invalid!";
  return false;
#endif
}

uint64_t RandomSeed::seed_ = 0ULL;

uint64_t RandomSeed::GetOrSet(uint64_t seed) {
  if (seed != 0ULL) {
    seed_ = seed;
  }
  return seed_;
}

uint64_t RandomSeed::Clear() {
  auto old_seed = seed_;
  seed_ = 0ULL;
  return old_seed;
}

bool CanUseNvccCompiler() {
  std::string nvcc_dir = FLAGS_cinn_nvcc_cmd_path + "/nvcc";
  return (access(nvcc_dir.c_str(), 0) == -1 ? false : true) &&
         (!FLAGS_cinn_compile_with_nvrtc);
}

bool IsCompiledWithCUDA() {
#if !defined(CINN_WITH_CUDA)
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithCUDNN() {
#if !defined(CINN_WITH_CUDNN)
  return false;
#else
  return true;
#endif
}

cinn::common::Target CurrentTarget::target_ = cinn::common::DefaultTarget();

void CurrentTarget::SetCurrentTarget(const cinn::common::Target& target) {
  if (!IsCompiledWithCUDA() &&
      target.arch == cinn::common::Target::Arch::NVGPU) {
    LOG(FATAL) << "Current CINN version does not support NVGPU, please try to "
                  "recompile with -DWITH_CUDA.";
  } else {
    target_ = target;
  }
}

cinn::common::Target& CurrentTarget::GetCurrentTarget() { return target_; }

}  // namespace runtime
}  // namespace cinn
