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

PassStrategy *contrib::AnalysisConfig::pass_builder() const {
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

contrib::AnalysisConfig::AnalysisConfig(const std::string &model_dir) {
  model_dir_ = model_dir;
}
contrib::AnalysisConfig::AnalysisConfig(const std::string &prog_file,
                                        const std::string &params_file) {
  prog_file_ = prog_file;
  params_file_ = params_file;
}
void contrib::AnalysisConfig::SetModel(const std::string &prog_file_path,
                                       const std::string &params_file_path) {
  prog_file_ = prog_file_path;
  params_file_ = params_file_path;
}
void contrib::AnalysisConfig::EnableUseGpu(uint64_t memory_pool_init_size_mb,
                                           int device_id) {
#ifdef PADDLE_WITH_CUDA
  use_gpu_ = true;
  memory_pool_init_size_mb_ = memory_pool_init_size_mb;
  device_id_ = device_id;
#else
  LOG(ERROR) << "Please compile with gpu to EnableGpu";
  use_gpu_ = false;
#endif
}
void contrib::AnalysisConfig::DisableGpu() { use_gpu_ = false; }

contrib::AnalysisConfig::AnalysisConfig(const contrib::AnalysisConfig &other) {
#define CP_MEMBER(member__) member__ = other.member__;

  // Model related.
  CP_MEMBER(model_dir_);
  CP_MEMBER(prog_file_);
  CP_MEMBER(params_file_);
  CP_MEMBER(model_from_memory_);  // the memory model reuses prog_file_ and
                                  // params_file_ fields.
  // Gpu releated.
  CP_MEMBER(use_gpu_);
  CP_MEMBER(device_id_);
  CP_MEMBER(memory_pool_init_size_mb_);
  // TensorRT releated.
  CP_MEMBER(use_tensorrt_);
  CP_MEMBER(tensorrt_workspace_size_);
  CP_MEMBER(tensorrt_max_batchsize_);
  CP_MEMBER(tensorrt_min_subgraph_size_);
  // MKLDNN releated.
  CP_MEMBER(use_mkldnn_);
  CP_MEMBER(mkldnn_enabled_op_types_);

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
}

void contrib::AnalysisConfig::EnableMKLDNN() {
#ifdef PADDLE_WITH_MKLDNN
  pass_builder()->EnableMKLDNN();
  use_mkldnn_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MKLDNN";
  use_mkldnn_ = false;
#endif
}

void contrib::AnalysisConfig::EnableTensorRtEngine(int workspace_size,
                                                   int max_batch_size,
                                                   int min_subgraph_size) {
  use_tensorrt_ = true;
  tensorrt_workspace_size_ = workspace_size;
  tensorrt_max_batchsize_ = max_batch_size;
  Update();
}

void contrib::AnalysisConfig::Update() {
  auto info = SerializeInfoCache();
  if (info == serialized_info_cache_) return;

  if (use_gpu_) {
    pass_builder_.reset(new GpuPassStrategy);
  } else {
    pass_builder_.reset(new CpuPassStrategy);
  }

  if (use_tensorrt_) {
    if (!use_gpu_) {
      LOG(ERROR)
          << "TensorRT engine is not available when EnableGpu() not actived.";
    } else {
      // Append after the infer_clean pass.
      pass_builder()->InsertPass(1, "tensorrt_subgraph_pass");
    }
  }

  if (use_mkldnn_) {
    if (!enable_ir_optim_) {
      LOG(ERROR)
          << "EnableMKLDNN() only works when IR optimization is enabled.";
    }
#ifdef PADDLE_WITH_MKLDNN
    pass_builder()->EnableMKLDNN();
    use_mkldnn_ = true;
#else
    LOG(ERROR) << "Please compile with MKLDNN first to use MKLDNN";
    use_mkldnn_ = false;
#endif
  }

  if (ir_debug_) {
    pass_builder()->TurnOnDebug();
  }
}

std::string contrib::AnalysisConfig::SerializeInfoCache() {
  std::stringstream ss;
  ss << use_gpu_;
  ss << memory_pool_init_size_mb_;

  ss << use_tensorrt_;
  ss << tensorrt_workspace_size_;
  ss << tensorrt_max_batchsize_;

  ss << use_mkldnn_;
  ss << enable_ir_optim_;
  ss << use_feed_fetch_ops_;
  ss << ir_debug_;

  return ss.str();
}

void contrib::AnalysisConfig::SetCpuMathLibraryNumThreads(
    int cpu_math_library_num_threads) {
  cpu_math_library_num_threads_ = cpu_math_library_num_threads;
}

float contrib::AnalysisConfig::fraction_of_gpu_memory_for_pool() const {
#ifdef PADDLE_WITH_CUDA
  // Get the GPU memory details and calculate the fraction of memory for the
  // GPU memory pool.
  size_t gpu_used, gpu_available;
  platform::GpuMemoryUsage(&gpu_used, &gpu_available);
  double total_gpu_memory = (gpu_used + gpu_available) / 1024. / 1024.;
  float fraction_of_gpu_memory =
      static_cast<double>(memory_pool_init_size_mb()) / total_gpu_memory;
  return fraction_of_gpu_memory;
#else
  return 0.;
#endif
}

void contrib::AnalysisConfig::SetModelBuffer(const char *prog_buffer,
                                             size_t prog_buffer_size,
                                             const char *param_buffer,
                                             size_t param_buffer_size) {
  prog_file_ = std::string(prog_buffer, prog_buffer + prog_buffer_size);
  params_file_ = std::string(param_buffer, param_buffer + param_buffer_size);
  model_from_memory_ = true;
}

}  // namespace paddle
