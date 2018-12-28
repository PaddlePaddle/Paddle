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
                            int max_batch_size = 1, int min_subgraph_size = 3);
  bool use_tensorrt() const { return use_tensorrt_; }

  void EnableMKLDNN();
  bool use_mkldnn() const { return use_mkldnn_; }
  void SetMKLDNNOp(std::unordered_set<std::string> op_list) {
    mkldnn_enabled_op_types_ = op_list;
  }

  // Specify the memory buffer of program and parameter
  void SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size,
                      const char* program_buffer, size_t program_buffer_size);
  bool model_from_memory() const { return model_from_memory_; }

  friend class ::paddle::AnalysisPredictor;

 protected:
  bool use_tensorrt_{false};
  bool use_mkldnn_{false};
  std::unordered_set<std::string> mkldnn_enabled_op_types_;
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
  std::unique_ptr<PassStrategy> pass_builder_;
  bool model_from_memory_{false};
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
