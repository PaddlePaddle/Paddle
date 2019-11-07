// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle_api.h"  // NOLINT

namespace paddle {

// Algorithms for finding scale of quantized Tensors.
enum class ScaleAlgo {
  NONE,      // Do not compute scale
  MAX,       // Find scale based on the max absolute value
  MAX_CH,    // Find scale based on the max absolute value per output channel
  MAX_CH_T,  // Find scale based on the max absolute value per output channel
             // of a transposed tensor
  KL,        // Find scale based on KL Divergence
};

struct MkldnnQuantizerConfig {
  MkldnnQuantizerConfig();

  /** Specify a quantization algorithm for a connection (input/output) of the
   * operator type.
   * @param op_type_name the operator's name.
   * @param conn_name name of the connection (input/output) of the operator.
   * @param algo the algorithm for computing scale.
   */
  void SetScaleAlgo(std::string op_type_name, std::string conn_name,
                    ScaleAlgo algo) {
    rules_[op_type_name][conn_name] = algo;
  }

  /** Get the quantization algorithm for a connection (input/output) of the
   * operator type.
   * @param op_type_name the operator's name.
   * @param conn_name name of the connection (input/output) of the operator.
   * @return the algorithm for computing scale.
   */
  ScaleAlgo scale_algo(const std::string& op_type_name,
                       const std::string& conn_name) const;

  /** Set the batch of data to be used for warm-up iteration.
   * @param data batch of data.
   */
  void SetWarmupData(std::shared_ptr<std::vector<PaddleTensor>> data) {
    warmup_data_ = data;
  }

  /** Get the batch of data used for warm-up iteration.
   * @return batch of data.
   */
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data() const {
    return warmup_data_;
  }

  void SetWarmupBatchSize(int batch_size) { warmup_bs_ = batch_size; }

  int warmup_batch_size() const { return warmup_bs_; }

  void SetEnabledOpTypes(std::unordered_set<std::string> op_list) {
    enabled_op_types_ = op_list;
  }

  const std::unordered_set<std::string>& enabled_op_types() const {
    return enabled_op_types_;
  }

  void SetExcludedOpIds(std::unordered_set<int> op_ids_list) {
    excluded_op_ids_ = op_ids_list;
  }

  const std::unordered_set<int>& excluded_op_ids() const {
    return excluded_op_ids_;
  }

  void SetDefaultScaleAlgo(ScaleAlgo algo) { default_scale_algo_ = algo; }

  ScaleAlgo default_scale_algo() const { return default_scale_algo_; }

 protected:
  std::map<std::string, std::map<std::string, ScaleAlgo>> rules_;
  std::unordered_set<std::string> enabled_op_types_;
  std::unordered_set<int> excluded_op_ids_;
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data_;
  int warmup_bs_{1};
  ScaleAlgo default_scale_algo_{ScaleAlgo::MAX};
};

}  // namespace paddle
