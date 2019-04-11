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
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

/*! \file */

// Here we include some header files with relative paths, for that in deploy,
// the abstract path of this header file will be changed.
#include "paddle_api.h"           // NOLINT
#include "paddle_pass_builder.h"  // NOLINT
#ifdef PADDLE_WITH_MKLDNN
#include "paddle_mkldnn_quantizer_config.h"  // NOLINT
#endif

namespace paddle {

class AnalysisPredictor;
struct MkldnnQuantizerConfig;

// NOTE WIP, not stable yet.
struct AnalysisConfig {
  AnalysisConfig() = default;
  explicit AnalysisConfig(const AnalysisConfig& other);
  explicit AnalysisConfig(const std::string& model_dir);
  explicit AnalysisConfig(const std::string& prog_file,
                          const std::string& params_file);
  enum class Precision {
    kFloat32 = 0,
    kInt8,
  };

  /** Set model with a directory.
   */
  void SetModel(const std::string& model_dir) { model_dir_ = model_dir; }
  /** Set model with two specific pathes for program and parameters.
   */
  void SetModel(const std::string& prog_file_path,
                const std::string& params_file_path);
  /** Set program file path.
   */
  void SetProgFile(const std::string& x) { prog_file_ = x; }
  /** Set parameter composed file path.
   */
  void SetParamsFile(const std::string& x) { params_file_ = x; }
  /** Get the model directory path.
   */
  const std::string& model_dir() const { return model_dir_; }
  /** Get the program file path.
   */
  const std::string& prog_file() const { return prog_file_; }
  /** Get the composed parameters file.
   */
  const std::string& params_file() const { return params_file_; }

  // GPU related.

  /**
   * \brief Turn on GPU.
   * @param memory_pool_init_size_mb initial size of the GPU memory pool in MB.
   * @param device_id the GPU card to use (default is 0).
   */
  void EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id = 0);
  /** Turn off the GPU.
   */
  void DisableGpu();
  /** A bool state telling whether the GPU is turned on.
   */
  bool use_gpu() const { return use_gpu_; }
  /** Get the GPU device id.
   */
  int gpu_device_id() const { return device_id_; }
  /** Get the initial size in MB of the GPU memory pool.
   */
  int memory_pool_init_size_mb() const { return memory_pool_init_size_mb_; }
  /** Get the proportion of the initial memory pool size compared to the device.
   */
  float fraction_of_gpu_memory_for_pool() const;

  /** \brief Control whether to perform IR graph optimization.
   *
   * If turned off, the AnalysisConfig will act just like a NativeConfig.
   */
  void SwitchIrOptim(int x = true) { enable_ir_optim_ = x; }
  /** A boolean state tell whether the ir graph optimization is actived.
   */
  bool ir_optim() const { return enable_ir_optim_; }

  /** \brief INTERNAL Determine whether to use the feed and fetch operators.
   * Just for internal development, not stable yet.
   * When ZeroCopyTensor is used, this should turned off.
   */
  void SwitchUseFeedFetchOps(int x = true) { use_feed_fetch_ops_ = x; }
  /** A boolean state telling whether to use the feed and fetch operators.
   */
  bool use_feed_fetch_ops_enabled() const { return use_feed_fetch_ops_; }

  /** \brief Control whether to specify the inputs' names.
   *
   * The PaddleTensor type has a `name` member, assign it with the corresponding
   * variable name. This is used only when the input PaddleTensors passed to the
   * `PaddlePredictor.Run(...)` cannot follow the order in the training phase.
   */
  void SwitchSpecifyInputNames(bool x = true) { specify_input_name_ = x; }

  /** A boolean state tell whether the input PaddleTensor names specified should
   * be used to reorder the inputs in `PaddlePredictor.Run(...)`.
   */
  bool specify_input_name() const { return specify_input_name_; }

  /**
   * \brief Turn on the TensorRT engine.
   *
   * The TensorRT engine will accelerate some subgraphes in the original Fluid
   * computation graph. In some models such as TensorRT50, GoogleNet and so on,
   * it gains significant performance acceleration.
   *
   * @param workspace_size the memory size(in byte) used for TensorRT workspace.
   * @param max_batch_size the maximum batch size of this prediction task,
   * better set as small as possible, or performance loss.
   * @param min_subgrpah_size the minimum TensorRT subgraph size needed, if a
   * subgraph is less than this, it will not transfer to TensorRT engine.
   */
  void EnableTensorRtEngine(int workspace_size = 1 << 20,
                            int max_batch_size = 1, int min_subgraph_size = 3,
                            Precision precision = Precision::kFloat32,
                            bool use_static = false);
  /** A boolean state telling whether the TensorRT engine is used.
   */
  bool tensorrt_engine_enabled() const { return use_tensorrt_; }
  /**
   *  \brief Turn on the usage of Anakin sub-graph engine.
   */
  void EnableAnakinEngine(
      int max_batch_size = 1,
      std::map<std::string, std::vector<int>> max_input_shape = {},
      int min_subgraph_size = 6, Precision precision = Precision::kFloat32,
      std::vector<std::string> passes_filter = {},
      std::vector<std::string> ops_filter = {});

  /** A boolean state indicating whether the Anakin sub-graph engine is used.
  */
  bool anakin_engine_enabled() const { return use_anakin_; }

  /** \brief Control whether to debug IR graph analysis phase.
   *
   * This will generate DOT files for visualizing the computation graph after
   * each analysis pass applied.
   */
  void SwitchIrDebug(int x = true);

  /** Turn on MKLDNN.
   */
  void EnableMKLDNN();
  /** A boolean state telling whether to use the MKLDNN.
   */
  bool mkldnn_enabled() const { return use_mkldnn_; }

  /** Set and get the number of cpu math library threads.
   */
  void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);
  /** An int state telling how many threads are used in the CPU math library.
   */
  int cpu_math_library_num_threads() const {
    return cpu_math_library_num_threads_;
  }

  /** Transform the AnalysisConfig to NativeConfig.
   */
  NativeConfig ToNativeConfig() const;
  /** Specify the operator type list to use MKLDNN acceleration.
   * @param op_list the operator type list.
   */
  void SetMKLDNNOp(std::unordered_set<std::string> op_list) {
    mkldnn_enabled_op_types_ = op_list;
  }

  /** Turn on quantization.
   */
  void EnableMkldnnQuantizer();

  /** A boolean state telling whether the quantization is enabled.
  */
  bool mkldnn_quantizer_enabled() const { return use_mkldnn_quantizer_; }

  std::shared_ptr<MkldnnQuantizerConfig> mkldnn_quantizer_config() const;

  /** Specify the memory buffer of program and parameter
   * @param prog_buffer the memory buffer of program.
   * @param prog_buffer_size the size of the data.
   * @param params_buffer the memory buffer of the composed parameters file.
   * @param params_buffer_size the size of the commposed parameters data.
   */
  void SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size,
                      const char* params_buffer, size_t params_buffer_size);
  /** A boolean state telling whether the model is set from the CPU memory.
   */
  bool model_from_memory() const { return model_from_memory_; }
  void SetEngineOptInfo(std::map<std::string, std::string> engine_opt_info);

  /** Turn on memory optimize
   * NOTE still in development, will release latter.
   */
  void EnableMemoryOptim(bool static_optim = false,
                         bool force_update_static_cache = false);
  /** Tell whether the memory optimization is activated. */
  bool enable_memory_optim() const;

  friend class ::paddle::AnalysisPredictor;

  /** NOTE just for developer, not an official API, easily to be broken.
   * Get a pass builder for customize the passes in IR analysis phase.
   */
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

  // GPU related.
  bool use_gpu_{false};
  int device_id_{0};
  uint64_t memory_pool_init_size_mb_{100};  // initial size is 100MB.

  // TensorRT related.
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
  Precision tensorrt_precision_mode_;
  bool trt_use_static_engine_;

  // memory reuse related.
  bool enable_memory_optim_{false};
  bool static_memory_optim_{false};
  bool static_memory_optim_force_update_{false};

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

  bool use_anakin_{false};
  int anakin_max_batchsize_;
  int anakin_min_subgraph_size_{6};
  std::map<std::string, std::vector<int>> anakin_max_input_shape_;
  Precision anakin_precision_mode_;
  std::vector<std::string> anakin_passes_filter_;
  std::vector<std::string> anakin_ops_filter_;
  std::map<std::string, std::string> engine_opt_info_;

  bool use_mkldnn_quantizer_{false};
  std::shared_ptr<MkldnnQuantizerConfig> mkldnn_quantizer_config_;
};

}  // namespace paddle
