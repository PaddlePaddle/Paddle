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
#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"

#include "paddle/cinn/common/shape_constraint.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"
#include "paddle/cinn/hlir/framework/pir/broadcast_with_cf.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/runtime/arch_device.h"
#include "paddle/cinn/utils/multi_threading.h"
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include <fstream> // 必须包含

PD_DECLARE_bool(enable_cinn_compile_cache);
PD_DECLARE_int64(cinn_compile_thread_num);

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

// ==========================================================
//  Helper Functions for Serialization (Placed inside namespace)
// ==========================================================

// 辅助函数：将任意基础类型写入文件
template <typename T>
void WriteBinary(std::ofstream& ofs, const T& value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// 辅助函数：从文件读取任意基础类型
template <typename T>
bool ReadBinary(std::ifstream& ifs, T& value) {
    if (ifs.read(reinterpret_cast<char*>(&value), sizeof(T))) {
        return true;
    }
    std::cerr << "Error: Failed to read binary data of size " << sizeof(T) << std::endl;
    return false;
}

// 保存元数据
bool SaveKernelMetaData(const pir::CINNKernelInfo* group_info, const std::string& filepath) {
    // 1. 打开文件流
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        VLOG(3) << "Error: Could not open file for writing: " << filepath;
        return false;
    }

    // ----------------------------------------------------
    // A. 序列化 temp_space_sizes
    // ----------------------------------------------------
    const auto& temp_sizes = group_info->temp_space_sizes; 
    size_t temp_size = temp_sizes.size();
    WriteBinary(ofs, temp_size);
    if (temp_size > 0) {
        ofs.write(reinterpret_cast<const char*>(temp_sizes.data()), temp_size * sizeof(int64_t));
    }

    // ----------------------------------------------------
    // B. 序列化 symbol_args_map
    // ----------------------------------------------------
    const auto& symbol_map = group_info->symbol_args_map;
    size_t map_size = symbol_map.size();
    WriteBinary(ofs, map_size);

    for (const auto& [key, bind_info] : symbol_map) {
        WriteBinary(ofs, key);

        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            
            int type_index = -1;
            
            if constexpr (std::is_same_v<T, pir::CINNKernelInfo::ArgDimIdx>) {
                type_index = 0;
                WriteBinary(ofs, type_index);
                // ArgDimIdx 只有 dim_idx
                WriteBinary(ofs, arg.arg_idx);
                WriteBinary(ofs, arg.dim_idx);
            } else if constexpr (std::is_same_v<T, pir::CINNKernelInfo::ArgValueIdx>) {
                type_index = 1;
                WriteBinary(ofs, type_index);
                // ArgValueIdx 有 input_idx 和 value_idx
                WriteBinary(ofs, arg.arg_idx); 
                WriteBinary(ofs, arg.value_idx);
            } else {
                 // 应该不会到达这里
            }
            
        }, bind_info);
    }
    
    ofs.close();
    return true;
}

// 加载 Kernel 元数据函数
bool LoadKernelMetaData(pir::CINNKernelInfo* group_info, const std::string& filepath) {
    // 1. 打开文件流
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        VLOG(3) << "Error: Could not open file for reading: " << filepath;
        return false;
    }

    // ----------------------------------------------------
    // A. 反序列化 temp_space_sizes
    // ----------------------------------------------------
    auto& temp_sizes = group_info->temp_space_sizes;
    size_t temp_size = 0;
    if (!ReadBinary(ifs, temp_size)) return false;
    
    if (temp_size > 0) {
        temp_sizes.resize(temp_size);
        if (!ifs.read(reinterpret_cast<char*>(temp_sizes.data()), temp_size * sizeof(int64_t))) {
            VLOG(3) << "Error: Failed to read temp_space_sizes content.";
            return false;
        }
    } else {
        temp_sizes.clear();
    }

    // ----------------------------------------------------
    // B. 反序列化 symbol_args_map
    // ----------------------------------------------------
    auto& symbol_map = group_info->symbol_args_map;
    symbol_map.clear();
    size_t map_size = 0;
    if (!ReadBinary(ifs, map_size)) return false;

    for (size_t i = 0; i < map_size; ++i) {
        int key = 0;
        int type_index = -1;
        if (!ReadBinary(ifs, key)) return false;
        if (!ReadBinary(ifs, type_index)) return false;

        pir::CINNKernelInfo::SymbolArgBindInfo bind_info;

        if (type_index == 0) { // ArgDimIdx
            pir::CINNKernelInfo::ArgDimIdx dim_info;
            // ArgDimIdx 只有 dim_idx (根据序列化逻辑反推)
            if (!ReadBinary(ifs, dim_info.arg_idx)) return false;
            if (!ReadBinary(ifs, dim_info.dim_idx)) return false;
            bind_info = dim_info;
        } else if (type_index == 1) { // ArgValueIdx
            pir::CINNKernelInfo::ArgValueIdx value_info;
            // ArgValueIdx 有 arg_idx 和 value_idx
            if (!ReadBinary(ifs, value_info.arg_idx)) return false;
            if (!ReadBinary(ifs, value_info.value_idx)) return false;
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
  CompilationContextMapper ctx_mapper(target_, groups); // construct and 往compilation_results_后追加
  VLOG(5) << "YUHAN!!! CompilationContextMapper Constructed ";
  auto& group_compilation_contexts = ctx_mapper.UniqueCompilationContexts();
  auto& compilation_results = ctx_mapper.MutableCompilationResult(); // 可能是空的，如果它不是NewAndQuique
  VLOG(5) << "YUHAN!!! CompilationContextMapper compilation_results.size() =  " << compilation_results.size();
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
      VLOG(5) << "YUHAN!!! Before Compile Parallell group_compilation_contexts[" << index << "].fusion_hash = " << group_compilation_contexts[index].GetFusionHash();
      auto fusion_info_hash = group_compilation_contexts[index].GetFusionHash();
      std::string source_hash = std::to_string(fusion_info_hash);
      std::string cache_dir = "/tmp/cinn/" + source_hash; // 建议先定义目录
      std::string cache_so_path = cache_dir + "/cinn_cache.so";
      std::string meta_filepath = cache_dir + "/cinn_cache.meta";
      // 检查 .so 是否存在 (这里假设 good() 是有效的检查)
      if (std::ifstream(cache_so_path).good()) {
        VLOG(4) << "Cache hit for hash: " << source_hash;

        // 1. 声明临时结构体
        pir::CINNKernelInfo loaded_kernel_info; 
        
        // 2. 加载元数据
        bool load_success = LoadKernelMetaData(&loaded_kernel_info, meta_filepath);

        PADDLE_ENFORCE_EQ(
            load_success,
            true,
            ::common::errors::Unavailable("Failed to load kernel metadata "
                                        "from cache file: %s. Cache system is "
                                        "broken or corrupted. Please delete the cache "
                                        "directory and retry.", meta_filepath));
        VLOG(4) << "Successfully loaded metadata.";
        
        // A. 构造 CompilationResult
        auto result = std::make_shared<pir::CompilationResult>(
            target_, false, fusion_info_hash);
        
        // B. 构造 BackendResource (使用加载的数据!)
        auto resource = std::make_shared<pir::BackendResource>(
            target_,
            group_compilation_contexts[index].GetGroup()->FuncName(),
            group_compilation_contexts[index].GetGroup()->FuncName() + "_infer_shape",
            loaded_kernel_info.symbol_args_map,  // Load from meta
            loaded_kernel_info.temp_space_sizes  // Load from meta
        );

        // C. Load .so
        resource->GetBackendCompiler()->SetFusionHash(fusion_info_hash);
        resource->GetBackendCompiler()->LoadAndRegisterFromCache(source_hash);
        
        result->SetBackendResource(resource);
        compilation_results[index] = result;

      } else {
        // 编译路径
        compilation_results[index] = Compile(&group_compilation_contexts[index]);
        
        // 保存元数据
        pir::CINNKernelInfo info_to_save = compilation_results[index]->GetKernelInfo();
        
        // 确保目录存在 (此处略去 mkdir 逻辑，假设已由其他部分保证或手动创建)
        // system(("mkdir -p " + cache_dir).c_str()); 
        
        SaveKernelMetaData(&info_to_save, meta_filepath);
      }
      VLOG(5) << "YUHAN!!! group_compilation_contexts[index].GetGroup()->symbol_args_map().size() = " << 
      group_compilation_contexts[index].GetGroup()->symbol_args_map().size();
      VLOG(5) << "YUHAN!!! After Compile Parallell group_compilation_contexts[" << index << "].fusion_hash = " << group_compilation_contexts[index].GetFusionHash();
    };
    // 并行编译
    utils::parallel_run(worker_fn,
                        utils::SequenceDispatcher(0, task_size),
                        /*thread_num=*/thread_size);
  }
  VLOG(5) << "Finished compiling " << task_size << " Cinn Kernel info.";
  ctx_mapper.SetFinalize(true);
  ctx_mapper.UpdateGlobalCache();
  return ctx_mapper.RecoverKernelInfos();
}

std::shared_ptr<pir::CompilationResult> PirCompiler::Compile(
    GroupCompilationContext* ctx) {
  std::shared_ptr<pir::CompilationResult> compile_result;
  VLOG(5) << "Inside Compile() ctx->GetFusionHash() = " << ctx->GetFusionHash();
  CompilationTask task(ctx);

  const auto& optional_broadcast_optimize_groups =
      pir::GetBroadcastGroupListForOptimize(ctx->GetGroup());

  if (optional_broadcast_optimize_groups.has_value()) {
    const auto& broadcast_switch_case_groups =
        optional_broadcast_optimize_groups.value();
    std::vector<GroupCompilationContext> switch_group_ctxs;
    for (const auto& group : broadcast_switch_case_groups) {
      switch_group_ctxs.emplace_back(target_, group);
      switch_group_ctxs.back().SetFusionHash(ctx->GetFusionHash());
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

  // Triggering llvm compilation in thread
  compile_result->GetKernelInfo();
  return compile_result;
}

void CompilationContextMapper::Construct(
    const Target& target, const std::vector<pir::OpLoweringGroupPtr>& groups) {
  std::unordered_set<size_t> unique_infos;
  const auto IsNewAndUnique =
      [&unique_infos](const pir::FusionInfo& info) -> bool {
    const bool is_unique = unique_infos.find(info.hash()) == unique_infos.end();
    const bool is_new = !CompilationCache::Instance().Has(info);
    return is_new && is_unique; //
  };

  for (size_t i = 0; i < groups.size(); ++i) {
    cinn::dialect::ir::details::UpdateGroupShapeOrDataExprs(groups[i]);
    fusion_infos_.emplace_back(*groups[i]);
    //
    // auto fusion_info_hash = fusion_infos_[i].hash();
    // std::string source_hash = std::to_string(fusion_info_hash);
    // std::string cache_so_path = "/tmp/cinn/" + source_hash + "/" + "cinn_cache.so";
    // if (std::ifstream(cache_so_path).good()) continue;
    //
    VLOG(4) << "Construct FusionInfo: " << fusion_infos_[i]
            << " for group: " << *groups[i];
    // If FLAGS_enable_cinn_compile_cache=False, Cache strategy will not take
    // effects.
    if (IsNewAndUnique(fusion_infos_[i]) || !FLAGS_enable_cinn_compile_cache) { //
      mapper_index_.push_back(i);
      auto fusion_info_hash = fusion_infos_[i].hash();
      group_compilation_contexts_.emplace_back(target, groups[i]);
      group_compilation_contexts_.back().SetFusionHash(fusion_info_hash);
      VLOG(4) << "YUHAN!!! compilation_results_.size() is " << compilation_results_.size();
      VLOG(4) << "YUHAN!!! compilation_results_.push_back hashKey is " << fusion_info_hash;
      compilation_results_.push_back(
          std::make_shared<pir::CompilationResult>(target, false, fusion_info_hash));
    }
    unique_infos.insert(fusion_infos_[i].hash()); //
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
