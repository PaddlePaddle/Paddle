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
struct AnalysisConfig : public NativeConfig {
  explicit AnalysisConfig(bool use_gpu = false);
  explicit AnalysisConfig(const AnalysisConfig& other);
  explicit AnalysisConfig(AnalysisConfig&& other);

  // Determine whether to perform graph optimization.
  bool enable_ir_optim = true;

  // Get a pass builder for customize the passes in IR analysis phase.
  PassStrategy* pass_builder() const;

  // NOT stable yet.
  bool use_feed_fetch_ops{true};

  void EnableTensorRtEngine(int workspace_size = 1 << 20,
                            int max_batch_size = 1);
  bool use_tensorrt() const { return use_tensorrt_; }

  void EnableMKLDNN();
  // NOTE this is just for internal development, please not use it.
  // NOT stable yet.
  bool use_mkldnn() const { return use_mkldnn_; }

  friend class ::paddle::AnalysisPredictor;

 protected:
  bool use_tensorrt_{false};
  bool use_mkldnn_{false};
  int tensorrt_workspace_size_;
  int tensorrt_max_batchsize_;
  std::unique_ptr<PassStrategy> pass_builder_;
};

// Configurations for Anakin engine.
struct AnakinConfig : public PaddlePredictor::Config {
  enum TargetType { NVGPU = 0, X86 };
  int device;
  std::string model_file;
  int max_batch_size{-1};
  TargetType target_type;
};

}  // namespace contrib
}  // namespace paddle
