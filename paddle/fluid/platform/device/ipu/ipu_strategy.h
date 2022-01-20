/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <popart/op.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensorlocation.hpp>
#include "popart/patterns/patterns.hpp"

namespace paddle {
namespace platform {
namespace ipu {

using VirtualGraphMode = popart::VirtualGraphMode;
using RecomputationType = popart::RecomputationType;

struct IpuStrategy {
  IpuStrategy() {
    // we always save optimizer state to OffChip and enable rts for saving
    // memory
    auto storage = popart::TensorLocation(popart::TensorStorage::OffChip,
                                          popart::ReplicatedTensorSharding::On);
    popart_options.optimizerStateTensorLocationSettings =
        popart::TensorLocationSettings(storage);

    // We divide the accumulationFactor and replicatedGraphCount after all
    // reduce
    popart_options.accumulationAndReplicationReductionType =
        popart::ReductionType::Mean;
    popart_options.meanAccumulationAndReplicationReductionStrategy =
        popart::MeanReductionStrategy::Post;

    popart_options.enableFloatingPointChecks = false;

    // A directory for log traces to be written into.
    popart_options.logDir = "popart_log";
  }
  ~IpuStrategy() {}

  // Number ipus total needed, replica * ipu_per_replica
  int num_ipus = 1;

  // batches per step
  int batches_per_step = 1;

  // micro batch-size
  int micro_batch_size = 1;

  // training flag, true for training
  bool is_training = true;

  // save the onnx model lowered by paddle program description
  bool save_init_onnx = false;

  // save the trained model
  bool save_onnx_checkpoint = false;

  // save paddle model per n steps
  int save_per_n_step = 1;

  // average sharding, debugging used
  bool need_avg_shard = false;

  // flag for fp16, true for pure fp16
  bool enable_fp16 = false;

  // available memory proportion, 0.0f for disable
  float available_memory_proportion = 0.0f;

  // loss scaling, currently we can't get loss scaling from
  // optimizer_extract_pass, so we have to set it here
  float loss_scaling = 1.0f;

  // defaultMaxWeightNorm for adam optimizer
  float max_weight_norm = 65504.0f;

  // popart session option
  popart::SessionOptions popart_options;
  popart::Patterns popart_patterns;

 public:
  void enablePattern(const std::string& t);
  void disablePattern(const std::string& t);
  const bool isPatternEnabled(const std::string& t);
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
