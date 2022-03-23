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

#include <popart/patterns/patterns.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensorlocation.hpp>
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {

struct RuntimeOptions {
  // enable the eval mode in training by switching optimizers.
  bool enable_eval = false;
};

class IpuStrategy {
 public:
  IpuStrategy();

  // TODO(alleng) create PaddleOptions
  // training flag, true for training
  bool is_training = true;

  // average sharding, debugging used
  bool need_avg_shard = false;

  // flag for fp16, true for pure fp16
  bool enable_fp16 = false;

  // enable transfer cast Op target from fp32 to fp16 in fp16 mode
  bool transfer_cast_op = true;

  // The mode of Adam/Lamb optimizer
  // false: The standard Adam/Lamb optimizer
  // true: The Adam_No_Bias/Lamb_No_Bias optimizer from PopART
  bool use_no_bias_optimizer = false;

  // enable distributed computing for POD128 or POD256
  bool enable_distribution = false;

  // Number ipus total needed, local_replica * ipu_per_replica
  int num_ipus = 1;

  // batches per step
  int batches_per_step = 1;

  // micro batch-size
  int micro_batch_size = 1;

  // random seed
  std::uint64_t random_seed = std::numeric_limits<std::uint64_t>::max();

  // TODO(alleng) remove this param
  // available memory proportion, 0.0f for disable
  float available_memory_proportion = 0.0f;

  // loss scaling, currently we can't get loss scaling from
  // optimizer_extract_pass, so we have to set it here
  float loss_scaling = 1.0f;

  // defaultMaxWeightNorm for adam optimizer
  float max_weight_norm = 65504.0f;

  // file path for dumping compiled model in onnx format
  std::string onnx_dump_path;

  // Data type to use for tensor that stores first-order momentum optimizer
  // state. FLOAT or FLOAT16
  std::string accl1_type = "FLOAT";

  // Data type to use for tensor that stores second-order momentum optimizer
  // state. FLOAT or FLOAT16
  std::string accl2_type = "FLOAT";

  // Data type to use for tensor that stores third-order momentum optimizer
  // state. FLOAT or FLOAT16
  std::string accl3_type = "FLOAT";

  // WeightDecayMode for setting the optimizer
  // if set, it will override other settings
  // value must be one of "decay" or "l2_regularization" or not set
  std::string weight_decay_mode = "";

  // Runtime Options
  RuntimeOptions runtime_options;

  // popart session option
  popart::SessionOptions popart_options;

  // popart pattern manager
  popart::Patterns popart_patterns;

  // custom ops
  std::vector<IpuCustomOpIdentifier> custom_ops;

 public:
  void AddBoolOption(const std::string &option, bool value);
  void AddUint64Option(const std::string &option, std::uint64_t value);
  void AddDoubleOption(const std::string &option, double value);
  void AddStringOption(const std::string &option, const std::string &value);
  void InsertStringOption(const std::string &option, const std::string &value);
  void InsertStringPairOption(const std::string &option, const std::string &key,
                              const std::string &value);
  void SetTensorLocation(const std::string &tensor, const std::string &option,
                         std::uint64_t value);
  void SetAccumulateOuterFragmentSettings(const std::uint64_t &schedule,
                                          const std::vector<int> &values);
  void AddCustomOp(const std::string &paddle_op, const std::string &popart_op,
                   const std::string &domain, int version);

  std::string GetOption(const std::string &);
  std::vector<std::string> GetVectorOption(const std::string &);
  std::map<std::string, std::string> GetMapOption(const std::string &);
  std::string GetOptionType(const std::string &);
  std::vector<std::string> GetAllOptionNames();

  void EnablePattern(const std::string &t);
  void DisablePattern(const std::string &t);
  const bool IsPatternEnabled(const std::string &t);

 private:
  template <typename ValueType>
  void set(
      const std::string &key, ValueType value,
      std::map<std::string, std::function<void(ValueType)>> &options,  // NOLINT
      const std::string &type_str) {
    auto it = options.find(key);
    PADDLE_ENFORCE_NE(it, options.end(), platform::errors::InvalidArgument(
                                             "Cannot find option: %s, type: %s "
                                             "when setting IpuStrategy options",
                                             key, type_str));
    it->second(value);
  }

  template <typename ValueType>
  ValueType get(
      const std::string &key,
      std::map<std::string, std::function<ValueType()>> &options) {  // NOLINT
    auto it = options.find(key);
    PADDLE_ENFORCE_NE(
        it, options.end(),
        platform::errors::InvalidArgument(
            "Cannot find option name: %s when trying to get IpuStrategy option",
            key));
    return it->second();
  }

  std::map<std::string, std::function<void(bool)>> bool_options;
  std::map<std::string, std::function<void(std::uint64_t)>> uint64_options;
  std::map<std::string, std::function<void(double)>> double_options;
  std::map<std::string, std::function<void(std::string)>> string_options;
  std::map<std::string,
           std::function<void(std::pair<std::string, std::string>)>>
      container_options;

  std::map<std::string, std::function<std::string()>> options_getter;
  std::map<std::string, std::function<std::vector<std::string>()>>
      vector_options_getter;
  std::map<std::string, std::function<std::map<std::string, std::string>()>>
      map_options_getter;
  std::map<std::string, std::string> options_type;
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
