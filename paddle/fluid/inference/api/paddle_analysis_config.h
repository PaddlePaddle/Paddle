// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

// Here we include some header files with relative paths, for that in deploy,
// the abstract path of this header file will be changed.
#include "paddle_api.h"           // NOLINT
#include "paddle_pass_builder.h"  // NOLINT

namespace paddle {

class AnalysisPredictor;
// ==
//
// -----------------------------------------------------------------------------------
// NOTE: The following APIs are not mature yet, we are still working on them.
namespace contrib {

// NOTE WIP, not stable yet.
struct AnalysisConfig {
  AnalysisConfig() = default;
  explicit AnalysisConfig(const AnalysisConfig& other);
  explicit AnalysisConfig(const std::string& model_dir);
  explicit AnalysisConfig(const std::string& prog_file,
                          const std::string& params_file);

  // Model path related.
  void SetModel(const std::string& model_dir) { model_dir_ = model_dir; }
  void SetModel(const std::string& prog_file_path,
                const std::string& params_file_path);
  void SetProgFile(const std::string& x) { prog_file_ = x; }
  void SetParamsFile(const std::string& x) { params_file_ = x; }
  const std::string& model_dir() const { return model_dir_; }
  const std::string& prog_file() const { return prog_file_; }
  const std::string& params_file() const { return params_file_; }

  // GPU related.
  void EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id);
  void DisableGpu();
  bool use_gpu() const { return use_gpu_; }
  int gpu_device_id() const { return device_id_; }
  int memory_pool_init_size_mb() const { return memory_pool_init_size_mb_; }
  float fraction_of_gpu_memory_for_pool() const;

  // Determine whether to perform graph optimization.
  void SwitchIrOptim(int x = true) { enable_ir_optim_ = x; }
  bool ir_optim() const { return enable_ir_optim_; }

  void SwitchUseFeedFetchOps(int x = true) { use_feed_fetch_ops_ = x; }
  bool use_feed_fetch_ops_enabled() const { return use_feed_fetch_ops_; }

  void SwitchSpecifyInputNames(bool x = true) { specify_input_name_ = x; }
  bool specify_input_name() const { return specify_input_name_; }

  void EnableTensorRtEngine(int workspace_size = 1 << 20,
                            int max_batch_size = 1, int min_subgraph_size = 3);
  bool tensorrt_engine_enabled() const { return use_tensorrt_; }

  void SwitchIrDebug(int x = true) { ir_debug_ = x; }

  // NOTE this is just for internal development, please not use it.
  // NOT stable yet.
  void EnableMKLDNN();
  bool mkldnn_enabled() const { return use_mkldnn_; }

  // Set and get the number of cpu math library threads.
  void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);
  int cpu_math_library_num_threads() const {
    return cpu_math_library_num_threads_;
  }

  NativeConfig ToNativeConfig() const {
    NativeConfig config;
    config.model_dir = model_dir_;
    config.prog_file = prog_file_;
    config.param_file = params_file_;
    config.use_gpu = use_gpu_;
    config.device = device_id_;
    config.fraction_of_gpu_memory = fraction_of_gpu_memory_for_pool();
    config.specify_input_name = specify_input_name_;
    return config;
  }
  void SetMKLDNNOp(std::unordered_set<std::string> op_list) {
    mkldnn_enabled_op_types_ = op_list;
  }

  // Specify the memory buffer of program and parameter
  void SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size,
                      const char* program_buffer, size_t program_buffer_size);
  bool model_from_memory() const { return model_from_memory_; }

  friend class ::paddle::AnalysisPredictor;

  // NOTE just for developer, not an official API, easily to be broken.
  // Get a pass builder for customize the passes in IR analysis phase.
  PassStrategy* pass_builder() const;

 protected:
  // Update the config.
  void Update();

  std::string SerializeInfoCache();

 protected:
  // Model pathes.
  std::string model_dir_;
  std::string prog_file_;
  std::string params_file_;

  // GPU releated.
  bool use_gpu_{false};
  int device_id_{0};
  uint64_t memory_pool_init_size_mb_{100};  // initial size is 100MB.

  // TensorRT releated.
  bool use_tensorrt_{false};
  // For workspace_size, refer it from here:
  // https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#troubleshooting
  int tensorrt_workspace_size_;
  // While TensorRT allows an engine optimized for a given max batch size
  // to run at any smaller size, the performance for those smaller
  // sizes may not be as well-optimized. Therefore, Max batch is best
  // equivalent to the runtime batch size.
  int tensorrt_max_batchsize_;
  //  We transform the Ops that can be converted into TRT layer in the model,
  //  and aggregate these Ops into subgraphs for TRT execution.
  //  We set this variable to control the minimum number of nodes in the
  //  subgraph, 3 as default value.
  int tensorrt_min_subgraph_size_{3};

  bool use_mkldnn_{false};
  std::unordered_set<std::string> mkldnn_enabled_op_types_;

  bool model_from_memory_{false};

  bool enable_ir_optim_{true};
  bool use_feed_fetch_ops_{true};
  bool ir_debug_{false};

  bool specify_input_name_{false};

  int cpu_math_library_num_threads_{1};

  // A runtime cache, shouldn't be transferred to others.
  std::string serialized_info_cache_;

  mutable std::unique_ptr<PassStrategy> pass_builder_;
};

}  // namespace contrib
}  // namespace paddle
