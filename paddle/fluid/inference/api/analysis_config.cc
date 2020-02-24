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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
extern const std::vector<std::string> kTRTSubgraphPasses;
extern const std::vector<std::string> kLiteSubgraphPasses;

PassStrategy *AnalysisConfig::pass_builder() const {
  if (!pass_builder_.get()) {
    if (use_gpu_) {
      LOG(INFO) << "Create GPU IR passes";
      pass_builder_.reset(new GpuPassStrategy);
    } else {
      LOG(INFO) << "Create CPU IR passes";
      pass_builder_.reset(new CpuPassStrategy);
    }
  } else if (pass_builder_->use_gpu() ^ use_gpu()) {
    LOG(WARNING) << "The use_gpu flag is not compatible between Config and "
                    "PassBuilder, the flags are "
                 << use_gpu() << " " << pass_builder_->use_gpu();
    LOG(WARNING) << "Please make them compatible, still use the existing "
                    "PassBuilder.";
  }

  return pass_builder_.get();
}

AnalysisConfig::AnalysisConfig(const std::string &model_dir) {
  model_dir_ = model_dir;

  Update();
}
AnalysisConfig::AnalysisConfig(const std::string &prog_file,
                               const std::string &params_file) {
  prog_file_ = prog_file;
  params_file_ = params_file;

  Update();
}
void AnalysisConfig::SetModel(const std::string &prog_file_path,
                              const std::string &params_file_path) {
  prog_file_ = prog_file_path;
  params_file_ = params_file_path;

  Update();
}
void AnalysisConfig::EnableUseGpu(uint64_t memory_pool_init_size_mb,
                                  int device_id) {
#ifdef PADDLE_WITH_CUDA
  use_gpu_ = true;
  memory_pool_init_size_mb_ = memory_pool_init_size_mb;
  device_id_ = device_id;
#else
  LOG(ERROR) << "Please compile with gpu to EnableGpu()";
  use_gpu_ = false;
#endif

  Update();
}
void AnalysisConfig::DisableGpu() {
  use_gpu_ = false;

  Update();
}

void AnalysisConfig::DisableFCPadding() {
  use_fc_padding_ = false;

  Update();
}

AnalysisConfig::AnalysisConfig(const AnalysisConfig &other) {
#define CP_MEMBER(member__) member__ = other.member__;

  // Model related.
  CP_MEMBER(model_dir_);
  CP_MEMBER(model_from_memory_);  // the memory model reuses prog_file_ and
                                  // params_file_ fields.

  CP_MEMBER(opt_cache_dir_);
  prog_file_ = std::move(other.prog_file_);
  params_file_ = std::move(other.params_file_);

  CP_MEMBER(use_fc_padding_);
  // GPU related.
  CP_MEMBER(use_gpu_);
  CP_MEMBER(use_cudnn_);
  CP_MEMBER(device_id_);
  CP_MEMBER(memory_pool_init_size_mb_);

  CP_MEMBER(enable_memory_optim_);
  // TensorRT related.
  CP_MEMBER(use_tensorrt_);
  CP_MEMBER(tensorrt_workspace_size_);
  CP_MEMBER(tensorrt_max_batchsize_);
  CP_MEMBER(tensorrt_min_subgraph_size_);
  CP_MEMBER(tensorrt_precision_mode_);
  CP_MEMBER(trt_use_static_engine_);
  CP_MEMBER(trt_use_calib_mode_);
  // NGRAPH related.
  CP_MEMBER(use_ngraph_);
  // MKLDNN related.
  CP_MEMBER(use_mkldnn_);
  CP_MEMBER(mkldnn_enabled_op_types_);
  CP_MEMBER(mkldnn_cache_capacity_);
  // Quantization related.
  CP_MEMBER(use_mkldnn_quantizer_);
  CP_MEMBER(mkldnn_quantizer_config_);

  CP_MEMBER(use_lite_);
  CP_MEMBER(lite_precision_mode_);
  CP_MEMBER(lite_passes_filter_);
  CP_MEMBER(lite_ops_filter_);

  // profile related.
  CP_MEMBER(with_profile_);

  // glog related.
  CP_MEMBER(with_glog_info_);

  // Ir related.
  CP_MEMBER(enable_ir_optim_);
  CP_MEMBER(use_feed_fetch_ops_);
  CP_MEMBER(ir_debug_);
  CP_MEMBER(specify_input_name_);

  CP_MEMBER(cpu_math_library_num_threads_);

  CP_MEMBER(serialized_info_cache_);

  if (use_gpu_) {
    pass_builder_.reset(new GpuPassStrategy(
        *static_cast<GpuPassStrategy *>(other.pass_builder())));
  } else {
    pass_builder_.reset(new CpuPassStrategy(
        *static_cast<CpuPassStrategy *>(other.pass_builder())));
  }

#undef CP_MEMBER

  Update();
}

void AnalysisConfig::EnableCUDNN() {
#ifdef PADDLE_WITH_CUDA
  use_cudnn_ = use_gpu_;
#else
  LOG(ERROR) << "Please compile with CUDA first to use cuDNN";
  use_cudnn_ = false;
#endif

  Update();
}

void AnalysisConfig::EnableMKLDNN() {
#ifdef PADDLE_WITH_MKLDNN
  use_mkldnn_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MKLDNN";
  use_mkldnn_ = false;
#endif

  Update();
}

void AnalysisConfig::SetMkldnnCacheCapacity(int capacity) {
#ifdef PADDLE_WITH_MKLDNN
  mkldnn_cache_capacity_ = capacity;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to set MKLDNN Thread Id";
  mkldnn_cache_capacity_ = 0;
#endif
}

void AnalysisConfig::EnableMkldnnQuantizer() {
#ifdef PADDLE_WITH_MKLDNN
  if (!mkldnn_quantizer_config_)
    mkldnn_quantizer_config_.reset(new MkldnnQuantizerConfig());
  use_mkldnn_quantizer_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MkldnnQuantizer";
  use_mkldnn_quantizer_ = false;
#endif

  Update();
}

void AnalysisConfig::EnableNgraph() {
#ifdef PADDLE_WITH_NGRAPH
  pass_builder()->EnableNgraph();
  use_ngraph_ = true;
#else
  LOG(ERROR) << "Please compile with NGRAPH first to use NGRAPH";
  use_ngraph_ = false;
#endif
}

MkldnnQuantizerConfig *AnalysisConfig::mkldnn_quantizer_config() const {
  PADDLE_ENFORCE_NOT_NULL(mkldnn_quantizer_config_,
                          "MkldnnQuantizer was not enabled yet.");
  return mkldnn_quantizer_config_.get();
}

void AnalysisConfig::EnableTensorRtEngine(
    int workspace_size, int max_batch_size, int min_subgraph_size,
    AnalysisConfig::Precision precision_mode, bool use_static,
    bool use_calib_mode) {
#ifdef PADDLE_WITH_CUDA
  if (!use_gpu()) {
    LOG(ERROR) << "To use TensorRT engine, please call EnableGpu() first";
    return;
  }

  use_tensorrt_ = true;
  tensorrt_workspace_size_ = workspace_size;
  tensorrt_max_batchsize_ = max_batch_size;
  tensorrt_min_subgraph_size_ = min_subgraph_size;
  tensorrt_precision_mode_ = precision_mode;
  trt_use_static_engine_ = use_static;
  trt_use_calib_mode_ = use_calib_mode;

  Update();
#else
  LOG(ERROR)
      << "To use TensorRT engine, please compile inference lib with GPU first.";
#endif
}

// TODO(Superjomn) refactor this, buggy.
void AnalysisConfig::Update() {
  auto info = SerializeInfoCache();
  if (info == serialized_info_cache_) return;

  // Transfer pass_builder and copy the existing compatible passes.
  if (!pass_builder_ || ((use_gpu() ^ pass_builder_->use_gpu()))) {
    if (use_gpu()) {
      pass_builder_.reset(new GpuPassStrategy);

      if (use_tensorrt_) {
        // Append after the Affine_channel_conv_fuse pass.
        pass_builder()->InsertPass(3, "tensorrt_subgraph_pass");
      }
    } else {
      pass_builder_.reset(new CpuPassStrategy);
    }

  } else {
    if (use_gpu()) {
      pass_builder_.reset(new GpuPassStrategy(
          *static_cast<GpuPassStrategy *>(pass_builder_.get())));

    } else {
      pass_builder_.reset(new CpuPassStrategy(
          *static_cast<CpuPassStrategy *>(pass_builder_.get())));
    }
  }

  if (use_tensorrt_) {
    pass_builder()->ClearPasses();
    for (const auto &pass : kTRTSubgraphPasses) {
      pass_builder()->AppendPass(pass);
    }
  }
  if (use_gpu() && use_cudnn_) {
#ifdef PADDLE_WITH_CUDA
    if (!enable_ir_optim_) {
      LOG(ERROR) << "EnableCUDNN() only works when IR optimization is enabled.";
    } else {
      pass_builder()->EnableCUDNN();
    }
#endif
  }

  if (use_ngraph_) {
    if (!enable_ir_optim_) {
      LOG(ERROR)
          << "EnableNgraph() only works when IR optimization is enabled.";
    }
#ifdef PADDLE_WITH_NGRAPH
    pass_builder()->EnableNgraph();
    use_ngraph_ = true;
#else
    LOG(ERROR) << "Please compile with NGRAPH first to use NGRAPH";
    use_ngraph_ = false;
#endif
  }

  if (use_mkldnn_) {
#ifdef PADDLE_WITH_MKLDNN
    if (!enable_ir_optim_) {
      LOG(ERROR)
          << "EnableMKLDNN() only works when IR optimization is enabled.";
    } else {
      pass_builder()->EnableMKLDNN();
    }
#endif
  }

  // Quantization passes must come after all other optimization passes
  if (use_mkldnn_quantizer_) {
    if (!enable_ir_optim_) {
      LOG(ERROR) << "EnableMkldnnQuantizer() only works when IR optimization "
                    "is enabled.";
    }
#ifdef PADDLE_WITH_MKLDNN
    pass_builder()->EnableMkldnnQuantizer();
#endif
  }

#ifdef PADDLE_WITH_MKLDNN
  // Do not optimize when mkldnn is on
  if (enable_memory_optim_ && !use_mkldnn_) {
#else
  if (enable_memory_optim_) {
#endif
    pass_builder()->AppendAnalysisPass("memory_optimize_pass");
  }

  if (use_lite_) {
#ifndef PADDLE_WITH_LITE
    LOG(WARNING) << "You tried to enable the lite subgraph "
                    "but did not have the option -DWITH_LITE compiled.";
#endif
    pass_builder()->ClearPasses();
    for (const auto &pass : kLiteSubgraphPasses) {
      if (std::find(lite_passes_filter_.begin(), lite_passes_filter_.end(),
                    pass) == lite_passes_filter_.end()) {
        pass_builder()->AppendPass(pass);
      }
    }
  }

  if (ir_debug_) {
    pass_builder()->TurnOnDebug();
  }
}

std::string AnalysisConfig::SerializeInfoCache() {
  std::stringstream ss;
  ss << model_dir_;
  ss << prog_file_;
  ss << params_file_;

  ss << use_gpu_;
  ss << use_fc_padding_;
  ss << device_id_;
  ss << memory_pool_init_size_mb_;

  ss << use_tensorrt_;
  ss << tensorrt_workspace_size_;
  ss << tensorrt_max_batchsize_;
  ss << tensorrt_min_subgraph_size_;

  ss << enable_memory_optim_;

  ss << use_ngraph_;

  ss << use_mkldnn_;
  ss << mkldnn_cache_capacity_;
  for (auto &item : mkldnn_enabled_op_types_) ss << item;
  ss << ";";

  ss << use_mkldnn_quantizer_;
  ss << model_from_memory_;

  ss << with_profile_;

  ss << with_glog_info_;

  ss << enable_ir_optim_;
  ss << use_feed_fetch_ops_;
  ss << ir_debug_;

  ss << specify_input_name_;
  ss << cpu_math_library_num_threads_;

  ss << use_lite_;

  return ss.str();
}

void AnalysisConfig::SetCpuMathLibraryNumThreads(
    int cpu_math_library_num_threads) {
  cpu_math_library_num_threads_ = cpu_math_library_num_threads;

  Update();
}

float AnalysisConfig::fraction_of_gpu_memory_for_pool() const {
#ifdef PADDLE_WITH_CUDA
  // Get the GPU memory details and calculate the fraction of memory for the
  // GPU memory pool.
  size_t gpu_used, gpu_available;
  platform::SetDeviceId(device_id_);
  platform::GpuMemoryUsage(&gpu_used, &gpu_available);
  double total_gpu_memory = (gpu_used + gpu_available) / 1024. / 1024.;
  float fraction_of_gpu_memory =
      static_cast<double>(memory_pool_init_size_mb()) / total_gpu_memory;
  return fraction_of_gpu_memory;
#else
  return 0.;
#endif
}

void AnalysisConfig::EnableMemoryOptim() {
  enable_memory_optim_ = true;
  Update();
}

bool AnalysisConfig::enable_memory_optim() const {
  return enable_memory_optim_;
}

void AnalysisConfig::SetModelBuffer(const char *prog_buffer,
                                    size_t prog_buffer_size,
                                    const char *param_buffer,
                                    size_t param_buffer_size) {
  prog_file_ = std::string(prog_buffer, prog_buffer + prog_buffer_size);
  params_file_ = std::string(param_buffer, param_buffer + param_buffer_size);
  model_from_memory_ = true;

  Update();
}

NativeConfig AnalysisConfig::ToNativeConfig() const {
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

void AnalysisConfig::SwitchIrDebug(int x) {
  ir_debug_ = x;
  Update();
}

void AnalysisConfig::EnableProfile() {
  with_profile_ = true;
  Update();
}

void AnalysisConfig::DisableGlogInfo() {
  with_glog_info_ = false;
  Update();
}

void AnalysisConfig::EnableLiteEngine(
    AnalysisConfig::Precision precision_mode,
    const std::vector<std::string> &passes_filter,
    const std::vector<std::string> &ops_filter) {
  use_lite_ = true;
  lite_precision_mode_ = precision_mode;
  lite_passes_filter_ = passes_filter;
  lite_ops_filter_ = ops_filter;
  Update();
}

void AnalysisConfig::PartiallyRelease() {
  prog_file_.clear();
  prog_file_.shrink_to_fit();
  params_file_.clear();
  params_file_.shrink_to_fit();
}

}  // namespace paddle
