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

///
/// \file paddle_analysis_config.h
///
/// \brief Paddle Analysis Config API信息
///
/// \author paddle-infer@baidu.com
/// \date 2020-03-20
/// \since 1.7
///

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle_infer_declare.h"  // NOLINT

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

struct LiteNNAdapterConfig {
  bool use_nnadapter{false};
  std::string nnadapter_model_cache_dir;
  std::map<std::string, std::vector<char>> nnadapter_model_cache_buffers;
  std::vector<std::string> nnadapter_device_names;
  std::string nnadapter_context_properties;
  std::string nnadapter_subgraph_partition_config_path;
  std::string nnadapter_subgraph_partition_config_buffer;

  LiteNNAdapterConfig& SetDeviceNames(const std::vector<std::string>& names);

  LiteNNAdapterConfig& SetContextProperties(const std::string& properties);

  LiteNNAdapterConfig& SetModelCacheDir(const std::string& dir);

  LiteNNAdapterConfig& SetModelCacheBuffers(
      const std::string& model_cache_token,
      const std::vector<char>& model_cache_buffer);

  LiteNNAdapterConfig& SetSubgraphPartitionConfigPath(const std::string& path);

  LiteNNAdapterConfig& SetSubgraphPartitionConfigBuffer(
      const std::string& buffer);

  LiteNNAdapterConfig& Enable();
  LiteNNAdapterConfig& Disable();
};

struct DistConfig {
  bool use_dist_model() const { return use_dist_model_; }
  void EnableDistModel(bool use_dist_model) {
    use_dist_model_ = use_dist_model;
  }

  std::vector<std::string> trainer_endpoints() const {
    return trainer_endpoints_;
  }

  std::string current_endpoint() const { return current_endpoint_; }

  void SetEndpoints(const std::vector<std::string>& trainer_endpoints,
                    const std::string& current_endpoint) {
    trainer_endpoints_ = trainer_endpoints;
    current_endpoint_ = current_endpoint;
  }

  int64_t nranks() const { return nranks_; }

  int64_t rank() const { return rank_; }

  void SetRanks(int64_t nranks, int64_t rank) {
    nranks_ = nranks;
    rank_ = rank;
  }

  std::string comm_init_config() const { return comm_init_config_; }

  void SetCommInitConfig(const std::string& comm_init_config) {
    comm_init_config_ = comm_init_config;
  }

  void SetCarrierId(const std::string& carrier_id) { carrier_id_ = carrier_id; }

  std::string carrier_id() const { return carrier_id_; }

 protected:
  // DistModel Inference related
  bool use_dist_model_{false};  // whether use DistModel or not
  std::vector<std::string> trainer_endpoints_{};  // all trainers' endpoints
  std::string current_endpoint_{};                // current trainer's endpoint
  int64_t nranks_{1};               // total ranks (number of trainers)
  int64_t rank_{0};                 // rank
  std::string comm_init_config_{};  // converter config path
  std::string carrier_id_{"inference"};
};

///
/// \brief configuration manager for AnalysisPredictor.
/// \since 1.7.0
///
/// AnalysisConfig manages configurations of AnalysisPredictor.
/// During inference procedure, there are many parameters(model/params path,
/// place of inference, etc.)
/// to be specified, and various optimizations(subgraph fusion, memory
/// optimazation, TensorRT engine, etc.)
/// to be done. Users can manage these settings by creating and modifying an
/// AnalysisConfig,
/// and loading it into AnalysisPredictor.
///
struct PD_INFER_DECL AnalysisConfig {
  AnalysisConfig() = default;
  ///
  /// \brief Construct a new AnalysisConfig from another
  /// AnalysisConfig.
  ///
  /// \param[in] other another AnalysisConfig
  ///
  explicit AnalysisConfig(const AnalysisConfig& other);
  ///
  /// \brief Construct a new AnalysisConfig from a no-combined model.
  ///
  /// \param[in] model_dir model directory of the no-combined model.
  ///
  explicit AnalysisConfig(const std::string& model_dir);
  ///
  /// \brief Construct a new AnalysisConfig from a combined model.
  ///
  /// \param[in] prog_file model file path of the combined model.
  /// \param[in] params_file params file path of the combined model.
  ///
  explicit AnalysisConfig(const std::string& prog_file,
                          const std::string& params_file);
  ///
  /// \brief Precision of inference in TensorRT.
  ///
  enum class Precision {
    kFloat32 = 0,  ///< fp32
    kInt8,         ///< int8
    kHalf,         ///< fp16
  };

  ///
  /// \brief Set the no-combined model dir path.
  ///
  /// \param model_dir model dir path.
  ///
  void SetModel(const std::string& model_dir) { model_dir_ = model_dir; }

  ///
  /// \brief Set the combined model with two specific pathes for program and
  /// parameters.
  ///
  /// \param prog_file_path model file path of the combined model.
  /// \param params_file_path params file path of the combined model.
  ///
  void SetModel(const std::string& prog_file_path,
                const std::string& params_file_path);
  ///
  /// \brief Set the model file path of a combined model.
  ///
  /// \param x model file path.
  ///
  void SetProgFile(const std::string& x) { prog_file_ = x; }
  ///
  /// \brief Set the params file path of a combined model.
  ///
  /// \param x params file path.
  ///
  void SetParamsFile(const std::string& x) { params_file_ = x; }

  ///
  /// \brief Set the path of optimization cache directory.
  ///
  /// \param opt_cache_dir the path of optimization cache directory.
  ///
  void SetOptimCacheDir(const std::string& opt_cache_dir) {
    opt_cache_dir_ = opt_cache_dir;
  }
  ///
  /// \brief Get the model directory path.
  ///
  /// \return const std::string& The model directory path.
  ///
  const std::string& model_dir() const { return model_dir_; }
  ///
  /// \brief Get the program file path.
  ///
  /// \return const std::string& The program file path.
  ///
  const std::string& prog_file() const { return prog_file_; }
  ///
  /// \brief Get the combined parameters file.
  ///
  /// \return const std::string& The combined parameters file.
  ///
  const std::string& params_file() const { return params_file_; }

  // Padding related.

  ///
  /// \brief Turn off FC Padding.
  ///
  ///
  void DisableFCPadding();
  ///
  /// \brief A boolean state telling whether fc padding is used.
  ///
  /// \return bool Whether fc padding is used.
  ///
  bool use_fc_padding() const { return use_fc_padding_; }

  // GPU related.

  ///
  /// \brief Turn on GPU.
  ///
  /// \param memory_pool_init_size_mb initial size of the GPU memory pool in MB.
  /// \param device_id device_id the GPU card to use (default is 0).
  ///
  void EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id = 0);
  ///
  /// \brief Turn off GPU.
  ///
  ///
  void DisableGpu();

  ///
  /// \brief Turn on XPU.
  ///
  /// \param l3_workspace_size The size of the video memory allocated by the l3
  ///         cache, the maximum is 16M.
  /// \param locked Whether the allocated L3 cache can be locked. If false,
  ///       it means that the L3 cache is not locked, and the allocated L3
  ///       cache can be shared by multiple models, and multiple models
  ///       sharing the L3 cache will be executed sequentially on the card.
  /// \param autotune Whether to autotune the conv operator in the model. If
  ///       true, when the conv operator of a certain dimension is executed
  ///       for the first time, it will automatically search for a better
  ///       algorithm to improve the performance of subsequent conv operators
  ///       of the same dimension.
  /// \param autotune_file Specify the path of the autotune file. If
  ///       autotune_file is specified, the algorithm specified in the
  ///       file will be used and autotune will not be performed again.
  /// \param precision Calculation accuracy of multi_encoder
  /// \param adaptive_seqlen Is the input of multi_encoder variable length
  ///
  void EnableXpu(int l3_workspace_size = 0xfffc00, bool locked = false,
                 bool autotune = true, const std::string& autotune_file = "",
                 const std::string& precision = "int16",
                 bool adaptive_seqlen = false);

  ///
  /// \brief Turn on IPU.
  ///
  /// \param ipu_device_num the number of IPUs.
  /// \param ipu_micro_batch_size the batch size in the graph, only work with
  /// mutable input shapes.
  /// \param ipu_enable_pipelining enable pipelining.
  /// \param ipu_batches_per_step the number of batches per run in pipelining.
  ///
  void EnableIpu(int ipu_device_num = 1, int ipu_micro_batch_size = 1,
                 bool ipu_enable_pipelining = false,
                 int ipu_batches_per_step = 1);

  ///
  /// \brief Set IPU config.
  ///
  /// \param ipu_enable_fp16 enable fp16.
  /// \param ipu_replica_num the number of graph replication.
  /// \param ipu_available_memory_proportion the available memory proportion for
  /// matmul/conv.
  /// \param ipu_enable_half_partial enable fp16 partial for matmul, only work
  /// with fp16.
  ///
  void SetIpuConfig(bool ipu_enable_fp16 = false, int ipu_replica_num = 1,
                    float ipu_available_memory_proportion = 1.0,
                    bool ipu_enable_half_partial = false);

  ///
  /// \brief Set XPU device id.
  ///
  /// \param device_id the XPU card to use (default is 0).
  ///
  void SetXpuDeviceId(int device_id = 0);
  ///
  /// \brief Turn on NPU.
  ///
  /// \param device_id device_id the NPU card to use (default is 0).
  ///
  void EnableNpu(int device_id = 0);
  ///
  /// \brief A boolean state telling whether the GPU is turned on.
  ///
  /// \return bool Whether the GPU is turned on.
  ///
  bool use_gpu() const { return use_gpu_; }
  ///
  /// \brief A boolean state telling whether the XPU is turned on.
  ///
  /// \return bool Whether the XPU is turned on.
  ///
  bool use_xpu() const { return use_xpu_; }
  ///
  /// \brief A boolean state telling whether the NPU is turned on.
  ///
  /// \return bool Whether the NPU is turned on.
  ///
  bool use_npu() const { return use_npu_; }
  /// \brief A boolean state telling whether the IPU is turned on.
  ///
  /// \return bool Whether the IPU is turned on.
  ///
  bool use_ipu() const { return use_ipu_; }
  ///
  /// \brief Get the GPU device id.
  ///
  /// \return int The GPU device id.
  ///
  int gpu_device_id() const { return gpu_device_id_; }
  ///
  /// \brief Get the XPU device id.
  ///
  /// \return int The XPU device id.
  ///
  int xpu_device_id() const { return xpu_device_id_; }
  ///
  /// \brief Get the NPU device id.
  ///
  /// \return int The NPU device id.
  ///
  int npu_device_id() const { return npu_device_id_; }
  /// \brief Get the the number of IPU device .
  ///
  /// \return int The number of IPU device.
  ///
  int ipu_device_num() const { return ipu_device_num_; }
  ///
  /// \brief Get the initial size in MB of the GPU memory pool.
  ///
  /// \return int The initial size in MB of the GPU memory pool.
  ///
  int memory_pool_init_size_mb() const { return memory_pool_init_size_mb_; }
  ///
  /// \brief Get the proportion of the initial memory pool size compared to the
  /// device.
  ///
  /// \return float The proportion of the initial memory pool size.
  ///
  float fraction_of_gpu_memory_for_pool() const;

  // CUDNN related.
  ///
  /// \brief Turn on CUDNN.
  ///
  ///
  void EnableCUDNN();
  ///
  /// \brief A boolean state telling whether to use CUDNN.
  ///
  /// \return bool Whether to use CUDNN.
  ///
  bool cudnn_enabled() const { return use_cudnn_; }

  ///
  /// \brief Control whether to perform IR graph optimization.
  /// If turned off, the AnalysisConfig will act just like a NativeConfig.
  ///
  /// \param x Whether the ir graph optimization is actived.
  ///
  void SwitchIrOptim(int x = true) { enable_ir_optim_ = x; }
  ///
  /// \brief A boolean state telling whether the ir graph optimization is
  /// actived.
  ///
  /// \return bool Whether to use ir graph optimization.
  ///
  bool ir_optim() const { return enable_ir_optim_; }

  ///
  /// \brief INTERNAL Determine whether to use the feed and fetch operators.
  /// Just for internal development, not stable yet.
  /// When ZeroCopyTensor is used, this should be turned off.
  ///
  /// \param x Whether to use the feed and fetch operators.
  ///
  void SwitchUseFeedFetchOps(int x = true) { use_feed_fetch_ops_ = x; }
  ///
  /// \brief A boolean state telling whether to use the feed and fetch
  /// operators.
  ///
  /// \return bool Whether to use the feed and fetch operators.
  ///
  bool use_feed_fetch_ops_enabled() const { return use_feed_fetch_ops_; }

  ///
  /// \brief Control whether to specify the inputs' names.
  /// The ZeroCopyTensor type has a name member, assign it with the
  /// corresponding
  /// variable name. This is used only when the input ZeroCopyTensors passed to
  /// the
  /// AnalysisPredictor.ZeroCopyRun() cannot follow the order in the training
  /// phase.
  ///
  /// \param x Whether to specify the inputs' names.
  ///
  void SwitchSpecifyInputNames(bool x = true) { specify_input_name_ = x; }
  ///
  /// \brief A boolean state tell whether the input ZeroCopyTensor names
  /// specified should
  /// be used to reorder the inputs in AnalysisPredictor.ZeroCopyRun().
  ///
  /// \return bool Whether to specify the inputs' names.
  ///
  bool specify_input_name() const { return specify_input_name_; }

  ///
  /// \brief Turn on the TensorRT engine.
  /// The TensorRT engine will accelerate some subgraphes in the original Fluid
  /// computation graph. In some models such as resnet50, GoogleNet and so on,
  /// it gains significant performance acceleration.
  ///
  /// \param workspace_size The memory size(in byte) used for TensorRT
  /// workspace.
  /// \param max_batch_size The maximum batch size of this prediction task,
  /// better set as small as possible for less performance loss.
  /// \param min_subgraph_size The minimum TensorRT subgraph size needed, if a
  /// subgraph is smaller than this, it will not be transferred to TensorRT
  /// engine.
  /// \param precision The precision used in TensorRT.
  /// \param use_static Serialize optimization information to disk for reusing.
  /// \param use_calib_mode Use TRT int8 calibration(post training
  /// quantization).
  ///
  ///
  void EnableTensorRtEngine(int workspace_size = 1 << 20,
                            int max_batch_size = 1, int min_subgraph_size = 3,
                            Precision precision = Precision::kFloat32,
                            bool use_static = false,
                            bool use_calib_mode = true);
  ///
  /// \brief A boolean state telling whether the TensorRT engine is used.
  ///
  /// \return bool Whether the TensorRT engine is used.
  ///
  bool tensorrt_engine_enabled() const { return use_tensorrt_; }
  ///
  /// \brief  Get the TensorRT engine precision.
  ///
  /// \return Precision Get the TensorRT engine precision.
  ///
  Precision tensorrt_precision_mode() const { return tensorrt_precision_mode_; }
  ///
  /// \brief Set min, max, opt shape for TensorRT Dynamic shape mode.
  /// \param min_input_shape The min input shape of the subgraph input.
  /// \param max_input_shape The max input shape of the subgraph input.
  /// \param opt_input_shape The opt input shape of the subgraph input.
  /// \param disable_trt_plugin_fp16 Setting this parameter to true means that
  /// TRT plugin will not run fp16.
  ///
  void SetTRTDynamicShapeInfo(
      std::map<std::string, std::vector<int>> min_input_shape,
      std::map<std::string, std::vector<int>> max_input_shape,
      std::map<std::string, std::vector<int>> optim_input_shape,
      bool disable_trt_plugin_fp16 = false);
  ///
  /// \brief A boolean state telling whether the trt dynamic_shape is used.
  ///
  /// \return bool Whether the trt dynamic_shape is used.
  ///
  bool tensorrt_dynamic_shape_enabled() const {
    return !min_input_shape_.empty();
  }
  ///
  /// \brief Enable tuned tensorrt dynamic shape.
  ///
  /// \param shape_range_info_path the path to shape_info file got in
  /// CollectShapeInfo
  /// mode.
  /// \param allow_build_at_runtime allow build trt engine at runtime.
  ///
  void EnableTunedTensorRtDynamicShape(const std::string& shape_range_info_path,
                                       bool allow_build_at_runtime = true);

  ///
  /// \brief A boolean state telling whether to use tuned tensorrt dynamic
  /// shape.
  ///
  bool tuned_tensorrt_dynamic_shape();

  ///
  /// \brief A boolean state telling whether to allow building trt engine at
  /// runtime.
  ///
  bool trt_allow_build_at_runtime();

  ///
  /// \brief Collect shape info of all tensors in compute graph.
  ///
  /// \param shape_range_info_path the path to save shape info.
  ///
  void CollectShapeRangeInfo(const std::string& shape_range_info_path);

  ///
  /// \brief the shape info path in CollectShapeInfo mode.
  ///
  /// \return the shape info path.
  ///
  const std::string& shape_range_info_path();

  ///
  /// \brief A boolean state telling whether to collect shape info.
  ///
  /// \return bool Whether to collect shape info.
  ///
  bool shape_range_info_collected();

  ///
  /// \brief Prevent ops running in Paddle-TRT
  /// NOTE: just experimental, not an official stable API, easy to be broken.
  ///
  void Exp_DisableTensorRtOPs(const std::vector<std::string>& ops);

  ///
  /// \brief Replace some TensorRT plugins to TensorRT OSS(
  /// https://github.com/NVIDIA/TensorRT), with which some models's inference
  /// may be more high-performance. Libnvinfer_plugin.so greater than
  /// V7.2.1 is needed.
  ///
  void EnableTensorRtOSS();

  ///
  /// \brief A boolean state telling whether to use the TensorRT OSS.
  ///
  /// \return bool Whether to use the TensorRT OSS.
  ///
  bool tensorrt_oss_enabled() { return trt_use_oss_; }

  ///
  /// \brief Enable TensorRT DLA
  /// \param dla_core ID of DLACore, which should be 0, 1,
  ///        ..., IBuilder.getNbDLACores() - 1
  ///
  void EnableTensorRtDLA(int dla_core = 0);

  ///
  /// \brief A boolean state telling whether to use the TensorRT DLA.
  ///
  /// \return bool Whether to use the TensorRT DLA.
  ///
  bool tensorrt_dla_enabled() { return trt_use_dla_; }

  void EnableTensorRtInspector();
  bool tensorrt_inspector_enabled() { return trt_use_inspector_; }

  void EnableDlnne(int min_subgraph_size = 3);
  bool dlnne_enabled() const { return use_dlnne_; }

  ///
  /// \brief Turn on the usage of Lite sub-graph engine.
  ///
  /// \param precision_mode Precion used in Lite sub-graph engine.
  /// \param passes_filter Set the passes used in Lite sub-graph engine.
  /// \param ops_filter Operators not supported by Lite.
  ///
  void EnableLiteEngine(
      AnalysisConfig::Precision precision_mode = Precision::kFloat32,
      bool zero_copy = false,
      const std::vector<std::string>& passes_filter = {},
      const std::vector<std::string>& ops_filter = {});

  ///
  /// \brief A boolean state indicating whether the Lite sub-graph engine is
  /// used.
  ///
  /// \return bool whether the Lite sub-graph engine is used.
  ///
  bool lite_engine_enabled() const { return use_lite_; }

  ///
  /// \brief Control whether to debug IR graph analysis phase.
  /// This will generate DOT files for visualizing the computation graph after
  /// each analysis pass applied.
  ///
  /// \param x whether to debug IR graph analysis phase.
  ///
  void SwitchIrDebug(int x = true);

  ///
  /// \brief Turn on MKLDNN.
  ///
  ///
  void EnableMKLDNN();
  ///
  /// \brief Set the cache capacity of different input shapes for MKLDNN.
  /// Default value 0 means not caching any shape.
  /// Please see MKL-DNN Data Caching Design Document:
  /// https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md
  ///
  /// \param capacity The cache capacity.
  ///
  void SetMkldnnCacheCapacity(int capacity);
  ///
  /// \brief A boolean state telling whether to use the MKLDNN.
  ///
  /// \return bool Whether to use the MKLDNN.
  ///
  bool mkldnn_enabled() const { return use_mkldnn_; }

  ///
  /// \brief Set the number of cpu math library threads.
  ///
  /// \param cpu_math_library_num_threads The number of cpu math library
  /// threads.
  ///
  void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);
  ///
  /// \brief An int state telling how many threads are used in the CPU math
  /// library.
  ///
  /// \return int The number of threads used in the CPU math library.
  ///
  int cpu_math_library_num_threads() const {
    return cpu_math_library_num_threads_;
  }

  ///
  /// \brief Transform the AnalysisConfig to NativeConfig.
  ///
  /// \return NativeConfig The NativeConfig transformed.
  ///
  NativeConfig ToNativeConfig() const;
  ///
  /// \brief Specify the operator type list to use MKLDNN acceleration.
  ///
  /// \param op_list The operator type list.
  ///
  void SetMKLDNNOp(std::unordered_set<std::string> op_list) {
    mkldnn_enabled_op_types_ = op_list;
  }

  ///
  /// \brief Turn on MKLDNN quantization.
  ///
  ///
  void EnableMkldnnQuantizer();

  ///
  /// \brief Turn on MKLDNN bfloat16.
  ///
  ///
  void EnableMkldnnBfloat16();

  ///
  /// \brief A boolean state telling whether to use the MKLDNN Bfloat16.
  ///
  /// \return bool Whether to use the MKLDNN Bfloat16.
  ///
  bool mkldnn_bfloat16_enabled() const { return use_mkldnn_bfloat16_; }

  /// \brief Specify the operator type list to use Bfloat16 acceleration.
  ///
  /// \param op_list The operator type list.
  ///
  void SetBfloat16Op(std::unordered_set<std::string> op_list) {
    bfloat16_enabled_op_types_ = op_list;
  }

  ///
  /// \brief A boolean state telling whether the thread local CUDA stream is
  /// enabled.
  ///
  /// \return bool Whether the thread local CUDA stream is enabled.
  ///
  bool thread_local_stream_enabled() const { return thread_local_stream_; }

  ///
  /// \brief A boolean state telling whether the MKLDNN quantization is enabled.
  ///
  /// \return bool Whether the MKLDNN quantization is enabled.
  ///
  bool mkldnn_quantizer_enabled() const { return use_mkldnn_quantizer_; }

  ///
  /// \brief Get MKLDNN quantizer config.
  ///
  /// \return MkldnnQuantizerConfig* MKLDNN quantizer config.
  ///
  MkldnnQuantizerConfig* mkldnn_quantizer_config() const;

  ///
  /// \brief Specify the memory buffer of program and parameter.
  /// Used when model and params are loaded directly from memory.
  ///
  /// \param prog_buffer The memory buffer of program.
  /// \param prog_buffer_size The size of the model data.
  /// \param params_buffer The memory buffer of the combined parameters file.
  /// \param params_buffer_size The size of the combined parameters data.
  ///
  void SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size,
                      const char* params_buffer, size_t params_buffer_size);
  ///
  /// \brief A boolean state telling whether the model is set from the CPU
  /// memory.
  ///
  /// \return bool Whether model and params are loaded directly from memory.
  ///
  bool model_from_memory() const { return model_from_memory_; }

  ///
  /// \brief Turn on memory optimize
  /// NOTE still in development.
  ///
  /// \param x Whether to enable memory optimize.
  ///
  void EnableMemoryOptim(bool x = true);
  ///
  /// \brief A boolean state telling whether the memory optimization is
  /// activated.
  ///
  /// \return bool Whether the memory optimization is activated.
  ///
  bool enable_memory_optim() const;

  ///
  /// \brief Turn on profiling report.
  /// If not turned on, no profiling report will be generated.
  ///
  void EnableProfile();
  ///
  /// \brief A boolean state telling whether the profiler is activated.
  ///
  /// \return bool Whether the profiler is activated.
  ///
  bool profile_enabled() const { return with_profile_; }

  ///
  /// \brief Mute all logs in Paddle inference.
  ///
  void DisableGlogInfo();
  ///
  /// \brief A boolean state telling whether logs in Paddle inference are muted.
  ///
  /// \return bool Whether logs in Paddle inference are muted.
  ///
  bool glog_info_disabled() const { return !with_glog_info_; }

  ///
  /// \brief Set the AnalysisConfig to be invalid.
  /// This is to ensure that an AnalysisConfig can only be used in one
  /// AnalysisPredictor.
  ///
  void SetInValid() const { is_valid_ = false; }
  ///
  /// \brief A boolean state telling whether the AnalysisConfig is valid.
  ///
  /// \return bool Whether the AnalysisConfig is valid.
  ///
  bool is_valid() const { return is_valid_; }

  friend class ::paddle::AnalysisPredictor;

  ///
  /// \brief Get a pass builder for customize the passes in IR analysis phase.
  /// NOTE: Just for developer, not an official API, easy to be broken.
  ///
  ///
  PassStrategy* pass_builder() const;

  ///
  /// \brief Enable the GPU multi-computing stream feature.
  /// NOTE: The current behavior of this interface is to bind the computation
  /// stream to the thread, and this behavior may be changed in the future.
  ///
  void EnableGpuMultiStream();
  void PartiallyRelease();

  ///
  /// \brief Print the summary of config.
  ///
  std::string Summary();

  LiteNNAdapterConfig& NNAdapter() { return nnadapter_config_; }

  void SetDistConfig(const DistConfig& dist_config) {
    dist_config_ = dist_config;
  }

  const DistConfig& dist_config() const { return dist_config_; }

 protected:
  // Update the config.
  void Update();

  std::string SerializeInfoCache();

 protected:
  // Model pathes.
  std::string model_dir_;
  mutable std::string prog_file_;
  mutable std::string params_file_;

  // GPU related.
  bool use_gpu_{false};
  int gpu_device_id_{0};
  uint64_t memory_pool_init_size_mb_{100};  // initial size is 100MB.
  bool thread_local_stream_{false};

  bool use_cudnn_{false};

  // NPU related
  bool use_npu_{false};
  int npu_device_id_{0};

  // Padding related
  bool use_fc_padding_{true};

  // TensorRT related.
  bool use_tensorrt_{false};
  // For workspace_size, refer it from here:
  // https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#troubleshooting
  int tensorrt_workspace_size_{1 << 30};
  // While TensorRT allows an engine optimized for a given max batch size
  // to run at any smaller size, the performance for those smaller
  // sizes may not be as well-optimized. Therefore, Max batch is best
  // equivalent to the runtime batch size.
  int tensorrt_max_batchsize_{1};
  //  We transform the Ops that can be converted into TRT layer in the model,
  //  and aggregate these Ops into subgraphs for TRT execution.
  //  We set this variable to control the minimum number of nodes in the
  //  subgraph, 3 as default value.
  int tensorrt_min_subgraph_size_{3};
  Precision tensorrt_precision_mode_{Precision::kFloat32};
  bool trt_use_static_engine_{false};
  bool trt_use_calib_mode_{true};
  bool trt_use_oss_{false};
  bool trt_with_interleaved_{false};
  bool trt_use_dla_{false};
  int trt_dla_core_{0};
  std::map<std::string, std::vector<int>> min_input_shape_{};
  std::map<std::string, std::vector<int>> max_input_shape_{};
  std::map<std::string, std::vector<int>> optim_input_shape_{};
  std::vector<std::string> trt_disabled_ops_{};
  bool disable_trt_plugin_fp16_{false};
  bool trt_allow_build_at_runtime_{false};
  // tune to get dynamic_shape info.
  bool trt_tuned_dynamic_shape_{false};
  bool trt_use_inspector_{false};

  // In CollectShapeInfo mode, we will collect the shape information of
  // all intermediate tensors in the compute graph and calculate the
  // min_shape, max_shape and opt_shape and save in shape_range_info_path_;
  bool collect_shape_range_info_{false};
  std::string shape_range_info_path_;

  // dlnne related.
  bool use_dlnne_{false};
  int dlnne_min_subgraph_size_{3};

  // memory reuse related.
  bool enable_memory_optim_{false};

  bool use_mkldnn_{false};
  std::unordered_set<std::string> mkldnn_enabled_op_types_;

  bool model_from_memory_{false};

  bool enable_ir_optim_{true};
  bool use_feed_fetch_ops_{true};
  bool ir_debug_{false};

  bool specify_input_name_{false};

  int cpu_math_library_num_threads_{1};

  bool with_profile_{false};

  bool with_glog_info_{true};

  // A runtime cache, shouldn't be transferred to others.
  std::string serialized_info_cache_;

  mutable std::unique_ptr<PassStrategy> pass_builder_;

  bool use_lite_{false};
  std::vector<std::string> lite_passes_filter_;
  std::vector<std::string> lite_ops_filter_;
  Precision lite_precision_mode_;
  bool lite_zero_copy_;

  // XPU related.
  bool use_xpu_{false};
  int xpu_device_id_{0};
  int xpu_l3_workspace_size_{0};
  bool xpu_locked_;
  bool xpu_autotune_;
  std::string xpu_autotune_file_;
  std::string xpu_precision_;
  bool xpu_adaptive_seqlen_;

  // NNAdapter related
  LiteNNAdapterConfig nnadapter_config_;

  // mkldnn related.
  int mkldnn_cache_capacity_{10};
  bool use_mkldnn_quantizer_{false};
  std::shared_ptr<MkldnnQuantizerConfig> mkldnn_quantizer_config_;
  bool use_mkldnn_bfloat16_{false};
  std::unordered_set<std::string> bfloat16_enabled_op_types_;

  // ipu related.
  bool use_ipu_{false};
  int ipu_device_num_{1};
  int ipu_micro_batch_size_{1};
  bool ipu_enable_pipelining_{false};
  int ipu_batches_per_step_{1};

  bool ipu_enable_fp16_{false};
  int ipu_replica_num_{1};
  float ipu_available_memory_proportion_{1.0};
  bool ipu_enable_half_partial_{false};

  // If the config is already used on a predictor, it becomes invalid.
  // Any config can only be used with one predictor.
  // Variables held by config can take up a lot of memory in some cases.
  // So we release the memory when the predictor is set up.
  mutable bool is_valid_{true};
  std::string opt_cache_dir_;
  friend class paddle_infer::experimental::InternalUtils;

  // fleet exe related
  DistConfig dist_config_{};
};

}  // namespace paddle
