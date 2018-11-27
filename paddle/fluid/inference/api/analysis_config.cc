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
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle_pass_builder.h"  // NOLINT

namespace paddle {

PassStrategy *contrib::AnalysisConfig::pass_builder() const {
  PADDLE_ENFORCE(
      pass_builder_.get(),
      "Should call constructor first, that will init the pass_builder_.");
  return pass_builder_.get();
}

contrib::AnalysisConfig::AnalysisConfig(bool use_gpu) {
  this->use_gpu = use_gpu;
  if (use_gpu) {
    pass_builder_.reset(new GpuPassStrategy);
  } else {
    pass_builder_.reset(new CpuPassStrategy);
  }
}

contrib::AnalysisConfig::AnalysisConfig(const contrib::AnalysisConfig &other) {
  // fields from Config
  model_dir = other.model_dir;
  // fields from NativeConfig
  use_gpu = other.use_gpu;
  device = other.device;
  fraction_of_gpu_memory = other.fraction_of_gpu_memory;
  prog_file = other.prog_file;
  param_file = other.param_file;
  specify_input_name = other.specify_input_name;
  cpu_math_library_num_threads_ = other.cpu_math_library_num_threads_;
  // fields from this.
  enable_ir_optim = other.enable_ir_optim;
  use_feed_fetch_ops = other.use_feed_fetch_ops;
  use_tensorrt_ = other.use_tensorrt_;
  tensorrt_max_batchsize_ = other.tensorrt_max_batchsize_;
  tensorrt_workspace_size_ = other.tensorrt_workspace_size_;

  if (use_gpu) {
    pass_builder_.reset(new GpuPassStrategy(
        *static_cast<GpuPassStrategy *>(other.pass_builder())));
  } else {
    pass_builder_.reset(new CpuPassStrategy(
        *static_cast<CpuPassStrategy *>(other.pass_builder())));
  }
}

contrib::AnalysisConfig::AnalysisConfig(contrib::AnalysisConfig &&other) {
  // fields from Config
  model_dir = other.model_dir;
  // fields from NativeConfig
  use_gpu = other.use_gpu;
  device = other.device;
  fraction_of_gpu_memory = other.fraction_of_gpu_memory;
  prog_file = other.prog_file;
  param_file = other.param_file;
  specify_input_name = other.specify_input_name;
  cpu_math_library_num_threads_ = other.cpu_math_library_num_threads_;
  // fields from this.
  enable_ir_optim = other.enable_ir_optim;
  use_feed_fetch_ops = other.use_feed_fetch_ops;
  use_tensorrt_ = other.use_tensorrt_;
  tensorrt_max_batchsize_ = other.tensorrt_max_batchsize_;
  tensorrt_workspace_size_ = other.tensorrt_workspace_size_;
  pass_builder_ = std::move(other.pass_builder_);
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
                                                   int max_batch_size) {
  use_tensorrt_ = true;
  tensorrt_workspace_size_ = workspace_size;
  tensorrt_max_batchsize_ = max_batch_size;
  // Append after the infer_clean pass.
  pass_builder()->InsertPass(1, "tensorrt_subgraph_pass");
}

}  // namespace paddle
