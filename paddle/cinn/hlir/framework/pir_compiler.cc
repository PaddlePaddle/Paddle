// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include "paddle/cinn/common/shape_constraint.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"
#include "paddle/cinn/hlir/framework/pir/broadcast_with_cf.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"
#include "paddle/cinn/runtime/arch_device.h"
#include "paddle/cinn/utils/multi_threading.h"
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

PD_DECLARE_bool(enable_cinn_compile_cache);
PD_DECLARE_int64(cinn_compile_thread_num);
COMMON_DECLARE_bool(enable_cinn_kernel_cache);
COMMON_DECLARE_string(cinn_kernel_cache_save_path);

namespace cinn::hlir::framework {
class CompilationContextMapper {
 public:
  CompilationContextMapper(const Target& target,
                           const std::vector<pir::OpLoweringGroupPtr>& groups) {
    Construct(target, groups);
  }
  std::vector<GroupCompilationContext>& UniqueCompilationContexts() {
    return group_compilation_contexts_;
  }
  std::vector<std::shared_ptr<pir::CompilationResult>>&
  MutableCompilationResult() {
    return compilation_results_;
  }

  std::vector<pir::CINNKernelInfo> RecoverKernelInfos();
  void UpdateGlobalCache();
  void SetFinalize(bool val) { is_finalized_ = val; }

 private:
  void Construct(const Target& target,
                 const std::vector<pir::OpLoweringGroupPtr>& groups);
  std::vector<size_t> mapper_index_;
  std::vector<pir::FusionInfo> fusion_infos_;
  std::vector<GroupCompilationContext> group_compilation_contexts_;
  std::vector<std::shared_ptr<pir::CompilationResult>> compilation_results_;

  bool is_finalized_{false};
};

static size_t GetThreadNum(size_t task_size) {
  size_t thread_size = task_size;
  if (!FLAGS_enable_cinn_compile_cache) {
    thread_size = 1;
  } else if (FLAGS_cinn_compile_thread_num > 0) {
    thread_size = FLAGS_cinn_compile_thread_num;
  }
  return thread_size;
}

// Helper function: Write any primitive type to file
template <typename T>
void WriteBinary(std::ofstream& ofs, const T& value) {
  ofs.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Helper function: Read any primitive type from file
template <typename T>
bool ReadBinary(std::ifstream& ifs, T* value) {
  if (ifs.read(reinterpret_cast<char*>(value), sizeof(T))) {
    return true;
  }
  std::cerr << "Error: Failed to read binary data of size " << sizeof(T)
            << std::endl;
  return false;
}

// Save kernel metadata to file
bool SaveKernelMetaData(const pir::CINNKernelInfo* group_info,
                        const std::string& filepath) {
  std::ofstream ofs(filepath, std::ios::binary);
  if (!ofs.is_open()) {
    VLOG(3) << "Error: Could not open file for writing: " << filepath;
    return false;
  }

  // Serialize temp_space_sizes
  const auto& temp_sizes = group_info->temp_space_sizes;
  size_t temp_size = temp_sizes.size();
  WriteBinary(ofs, temp_size);
  if (temp_size > 0) {
    ofs.write(reinterpret_cast<const char*>(temp_sizes.data()),
              temp_size * sizeof(int64_t));
  }

  // Serialize symbol_args_map
  const auto& symbol_map = group_info->symbol_args_map;
  size_t map_size = symbol_map.size();
  WriteBinary(ofs, map_size);

  for (const auto& [key, bind_info] : symbol_map) {
    WriteBinary(ofs, key);

    std::visit(
        [&](auto&& arg) {
          using T = std::decay_t<decltype(arg)>;
          using ArgValueIdx = pir::CINNKernelInfo::ArgValueIdx;

          int type_index = -1;

          if constexpr (std::is_same_v<T, pir::CINNKernelInfo::ArgDimIdx>) {
            type_index = 0;
            WriteBinary(ofs, type_index);
            // ArgDimIdx only contains dim_idx
            WriteBinary(ofs, arg.arg_idx);
            WriteBinary(ofs, arg.dim_idx);
          } else if constexpr (std::is_same_v<T, ArgValueIdx>) {
            type_index = 1;
            WriteBinary(ofs, type_index);
            // ArgValueIdx contains input_idx and value_idx
            WriteBinary(ofs, arg.arg_idx);
            WriteBinary(ofs, arg.value_idx);
          } else {
            LOG(FATAL) << "Should not reach here";
          }
        },
        bind_info);
  }

  ofs.close();
  return true;
}

// Load kernel metadata from file
bool LoadKernelMetaData(pir::CINNKernelInfo* group_info,
                        const std::string& filepath) {
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs.is_open()) {
    VLOG(3) << "Error: Could not open file for reading: " << filepath;
    return false;
  }

  // Deserialize temp_space_sizes
  auto& temp_sizes = group_info->temp_space_sizes;
  size_t temp_size = 0;
  if (!ReadBinary(ifs, &temp_size)) return false;

  if (temp_size > 0) {
    temp_sizes.resize(temp_size);
    if (!ifs.read(reinterpret_cast<char*>(temp_sizes.data()),
                  temp_size * sizeof(int64_t))) {
      VLOG(3) << "Error: Failed to read temp_space_sizes content.";
      return false;
    }
  } else {
    temp_sizes.clear();
  }

  // Deserialize symbol_args_map
  auto& symbol_map = group_info->symbol_args_map;
  symbol_map.clear();
  size_t map_size = 0;
  if (!ReadBinary(ifs, &map_size)) return false;

  for (size_t i = 0; i < map_size; ++i) {
    int key = 0;
    int type_index = -1;
    if (!ReadBinary(ifs, &key)) return false;
    if (!ReadBinary(ifs, &type_index)) return false;

    pir::CINNKernelInfo::SymbolArgBindInfo bind_info;

    if (type_index == 0) {  // ArgDimIdx
      pir::CINNKernelInfo::ArgDimIdx dim_info;
      // ArgDimIdx only contains dim_idx (inferred from serialization logic)
      if (!ReadBinary(ifs, &dim_info.arg_idx)) return false;
      if (!ReadBinary(ifs, &dim_info.dim_idx)) return false;
      bind_info = dim_info;
    } else if (type_index == 1) {  // ArgValueIdx
      pir::CINNKernelInfo::ArgValueIdx value_info;
      // ArgValueIdx contains arg_idx and value_idx
      if (!ReadBinary(ifs, &value_info.arg_idx)) return false;
      if (!ReadBinary(ifs, &value_info.value_idx)) return false;
      bind_info = value_info;
    } else {
      VLOG(3) << "Error: Unknown SymbolArgBindInfo type index: " << type_index;
      return false;
    }

    symbol_map.emplace(key, bind_info);
  }

  ifs.close();
  return true;
}

std::vector<pir::CINNKernelInfo> PirCompiler::Build(
    const std::vector<pir::OpLoweringGroupPtr>& groups) {
  CompilationContextMapper ctx_mapper(
      target_, groups);  // construct and append to compilation_results_
  auto& group_compilation_contexts = ctx_mapper.UniqueCompilationContexts();
  auto& compilation_results =
      ctx_mapper.MutableCompilationResult();  // may be empty if it's not new
                                              // and unique
  const size_t task_size = group_compilation_contexts.size();
  const size_t thread_size = GetThreadNum(task_size);
  VLOG(5) << "Found " << task_size << " new groups parsed from "
          << groups.size() << " and compiles with " << thread_size;
  cinn::ir::InitScheduleConfig();
  if (task_size > 0) {
    // See
    // https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
    // for details.
    const auto device_id = runtime::GetArchDevice(target_);
    auto worker_fn = [&](int index) {
      auto& shape_analysis_manager =
          ::pir::ShapeAnalysisManager::Instance().Get(
              group_compilation_contexts[index].GetGroup()->GetParentProgram());
      cinn::common::ShapeConstraintManager::Instance().Init(
          shape_analysis_manager.constraints_manager());
      runtime::SetArchDevice(target_, device_id);
      std::string cache_dir =
          FLAGS_cinn_kernel_cache_save_path + "/" +
          std::to_string(device_id.value()) + "/" +
          group_compilation_contexts[index].GetGroup()->FuncName();
      llvm::sys::fs::create_directories(cache_dir);
      std::string cache_so_path = cache_dir + "/" + CINN_CACHE_SO;
      std::string meta_filepath = cache_dir + "/" + CINN_CACHE_META;
      // Check if .so exists
      if (FLAGS_enable_cinn_kernel_cache &&
          std::ifstream(cache_so_path).good()) {
        // 1. Declare temporary structure
        pir::CINNKernelInfo loaded_kernel_info;

        // 2. Load metadata
        bool load_success =
            LoadKernelMetaData(&loaded_kernel_info, meta_filepath);

        PADDLE_ENFORCE_EQ(load_success,
                          true,
                          ::common::errors::Unavailable(
                              "Failed to load kernel metadata "
                              "from cache file: %s. Cache system is "
                              "broken or corrupted. Please delete the cache "
                              "directory and retry.",
                              meta_filepath));
        VLOG(4) << "Successfully loaded metadata.";

        // 3. Construct CompilationResult
        auto result = std::make_shared<pir::CompilationResult>(
            target_,
            false,
            group_compilation_contexts[index].GetGroup()->FuncName());

        // 4. Construct BackendResource (using loaded data!)
        auto resource = std::make_shared<pir::BackendResource>(
            target_,
            group_compilation_contexts[index].GetGroup()->FuncName(),
            group_compilation_contexts[index].GetGroup()->FuncName() +
                "_infer_shape",
            loaded_kernel_info.symbol_args_map,  // Load from meta
            loaded_kernel_info.temp_space_sizes  // Load from meta
        );

        // 5. Load .so
        resource->GetBackendCompiler()->SetFuncName(
            group_compilation_contexts[index].GetGroup()->FuncName());
        resource->GetBackendCompiler()->LoadAndRegisterFromCache();

        result->SetBackendResource(resource);
        compilation_results[index] = result;

      } else {
        // Compilation path
        compilation_results[index] =
            Compile(&group_compilation_contexts[index]);

        // Save metadata
        pir::CINNKernelInfo info_to_save =
            compilation_results[index]->GetKernelInfo();

        if (FLAGS_enable_cinn_kernel_cache) {
          SaveKernelMetaData(&info_to_save, meta_filepath);
        }
      }
      group_compilation_contexts[index].GetGroup()->symbol_args_map().size();
    };
    // Parallel compilation
    utils::parallel_run(worker_fn,
                        utils::SequenceDispatcher(0, task_size),
                        /*thread_num=*/thread_size);
  }

  ctx_mapper.SetFinalize(true);
  ctx_mapper.UpdateGlobalCache();
  return ctx_mapper.RecoverKernelInfos();
}

std::shared_ptr<pir::CompilationResult> PirCompiler::Compile(
    GroupCompilationContext* ctx) {
  std::shared_ptr<pir::CompilationResult> compile_result;
  CompilationTask task(ctx);

  const auto& optional_broadcast_optimize_groups =
      pir::GetBroadcastGroupListForOptimize(ctx->GetGroup());

  if (optional_broadcast_optimize_groups.has_value()) {
    const auto& broadcast_switch_case_groups =
        optional_broadcast_optimize_groups.value();
    std::vector<GroupCompilationContext> switch_group_ctxs;
    for (const auto& group : broadcast_switch_case_groups) {
      switch_group_ctxs.emplace_back(target_, group);
    }

    const auto& ParallelLowering = [&]() {
      const size_t task_size = switch_group_ctxs.size();
      auto worker_fn = [&](int index) {
        auto& shape_analysis_manager =
            ::pir::ShapeAnalysisManager::Instance().Get(
                switch_group_ctxs[index].GetGroup()->GetParentProgram());
        cinn::common::ShapeConstraintManager::Instance().Init(
            shape_analysis_manager.constraints_manager());
        CompilationTask lowering_task(&switch_group_ctxs[index]);
        lowering_task.Lowering();
      };
      const size_t thread_size = GetThreadNum(task_size);
      utils::parallel_run(worker_fn,
                          utils::SequenceDispatcher(0, task_size),
                          /*thread_num=*/thread_size);
    };

    ParallelLowering();
    std::unordered_map<int, ir::Var> symbolic_shape_var_index;
    UnifyBroadcastGroupFuncArgs(
        &switch_group_ctxs, ctx->GetGroup(), &symbolic_shape_var_index);
    compile_result = task.CompileBroadcastModules(&switch_group_ctxs,
                                                  symbolic_shape_var_index);
  } else {
    compile_result = task();
  }
  compile_result->SetFuncName(ctx->GetGroup()->FuncName());

  // Triggering llvm compilation in thread
  compile_result->GetKernelInfo();
  return compile_result;
}

std::string RemoveKernelSuffixNumber(const std::string& func_name) {
  if (func_name.empty()) {
    return func_name;
  }

  // 1. Find the position of the first non-digit character from the end.
  // suffix_start will point to the start index of the numeric suffix.
  size_t suffix_start = func_name.length();
  while (suffix_start > 0 && std::isdigit(func_name[suffix_start - 1])) {
    suffix_start--;
  }

  // Initialize cut_idx: assume no removal needed
  size_t cut_idx = func_name.length();

  // 2. Check if the numeric suffix is valid (must have digits and be preceded
  // by underscore)
  if (suffix_start < func_name.length() && suffix_start > 0 &&
      func_name[suffix_start - 1] == '_') {
    // Extract and validate numeric suffix
    std::string suffix = func_name.substr(suffix_start);
    size_t pos;

    try {
      // Validate if the suffix is a valid numeric value
      std::stoi(suffix, &pos);

      // Check if the entire suffix was converted
      if (pos != suffix.length()) {
        LOG(FATAL) << "Kernel suffix conversion failed for '" << func_name
                   << "'. Suffix '" << suffix
                   << "' contains non-digit characters after parsing.";
      }

      // Validation successful: set cut_idx to the start position of the numeric
      // suffix
      cut_idx = suffix_start;
    } catch (const std::exception& e) {
      // Conversion failed: suffix is not a valid number or out of range (Fatal
      // Error)
      LOG(FATAL) << "Kernel suffix conversion failed for '" << func_name
                 << "'. Suffix '" << suffix
                 << "' is not a valid integer. Exception: " << e.what();
    }

  } else {
    // Case 1: No numeric suffix (suffix_start == func_name.length())
    // Case 2: Has digits but not preceded by underscore (e.g., "fn123")
    // In both cases, keep cut_idx as func_name.length() and only remove
    // underscores
    cut_idx = func_name.length();
  }

  // 3. Final step: Remove all trailing underscore separators
  // Whether we removed numeric suffix (cut_idx = suffix_start), or kept the
  // full string (cut_idx = func_name.length()), we remove underscores starting
  // from cut_idx backwards.
  size_t final_cut_idx = cut_idx;
  while (final_cut_idx > 0 && func_name[final_cut_idx - 1] == '_') {
    final_cut_idx--;
  }

  // Return cleaned name without suffix and separators
  return func_name.substr(0, final_cut_idx);
}

void CompilationContextMapper::Construct(
    const Target& target, const std::vector<pir::OpLoweringGroupPtr>& groups) {
  std::unordered_set<size_t> unique_infos;
  const auto IsNewAndUnique =
      [&unique_infos](const pir::FusionInfo& info) -> bool {
    const bool is_unique = unique_infos.find(info.hash()) == unique_infos.end();
    const bool is_new = !CompilationCache::Instance().Has(info);
    return is_new && is_unique;
  };

  for (size_t i = 0; i < groups.size(); ++i) {
    cinn::dialect::ir::details::UpdateGroupShapeOrDataExprs(groups[i]);
    fusion_infos_.emplace_back(*groups[i]);

    // Rename FuncName
    auto fusion_info_hash = fusion_infos_[i].hash();
    auto func_name = groups[i]->FuncName();
    auto new_func_name = RemoveKernelSuffixNumber(func_name);
    const auto device_id = runtime::GetArchDevice(target);
    groups[i]->RenewFuncName(new_func_name + "__" +
                             std::to_string(fusion_info_hash));
    // If FLAGS_enable_cinn_compile_cache=False, Cache strategy will not take
    // effects.
    if (IsNewAndUnique(fusion_infos_[i]) || !FLAGS_enable_cinn_compile_cache) {
      mapper_index_.push_back(i);
      auto fusion_info_hash = fusion_infos_[i].hash();
      group_compilation_contexts_.emplace_back(target, groups[i]);
      VLOG(5) << "CompilerCache hashKey is " << fusion_info_hash;
      VLOG(5) << "CompilerCache FuncName is "
              << group_compilation_contexts_.back().GetGroup()->FuncName();
      compilation_results_.push_back(std::make_shared<pir::CompilationResult>(
          target, false, new_func_name));
    }
    unique_infos.insert(fusion_infos_[i].hash());
  }
}

std::vector<pir::CINNKernelInfo>
CompilationContextMapper::RecoverKernelInfos() {
  PADDLE_ENFORCE_EQ(
      is_finalized_,
      true,
      ::common::errors::PreconditionNotMet(
          "Required is_finalized_ = true, please call SetFinalize() firstly."));
  PADDLE_ENFORCE_EQ(group_compilation_contexts_.size(),
                    compilation_results_.size(),
                    ::common::errors::PreconditionNotMet(
                        "Required group_compilation_contexts_.size() = "
                        "compilation_results_.size()."));

  std::vector<pir::CINNKernelInfo> kernel_infos(fusion_infos_.size());
  for (size_t i = 0; i < fusion_infos_.size(); ++i) {
    const auto& compilation_result =
        FLAGS_enable_cinn_compile_cache
            ? CompilationCache::Instance().Get(fusion_infos_[i])
            : compilation_results_[i];
    kernel_infos[i] = compilation_result->GetKernelInfo();
  }
  return kernel_infos;
}

void CompilationContextMapper::UpdateGlobalCache() {
  PADDLE_ENFORCE_EQ(
      is_finalized_,
      true,
      ::common::errors::PreconditionNotMet(
          "Required is_finalized_ = true, please call SetFinalize() firstly."));
  for (size_t i = 0; i < compilation_results_.size(); ++i) {
    PADDLE_ENFORCE_LT(mapper_index_[i],
                      fusion_infos_.size(),
                      ::common::errors::PreconditionNotMet(
                          "Required mapper_index < fusion_infos_.size()."));
    const auto& fusion_info = fusion_infos_[mapper_index_[i]];
    VLOG(4) << "============== Insert new compiled result into cache, "
               "fusion_info: ==============\n"
            << fusion_info << ", host func name: "
            << compilation_results_[i]->GetHostFuncName();
    CompilationCache::Instance().Insert(fusion_info, compilation_results_[i]);
  }
}
}  // namespace cinn::hlir::framework
