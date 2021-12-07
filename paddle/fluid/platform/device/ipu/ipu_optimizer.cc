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

#include "paddle/fluid/platform/device/ipu/ipu_optimizer.h"

namespace paddle {
namespace platform {
namespace ipu {

OptmizerMetaInfo::OptmizerMetaInfo() {}

OptmizerMetaInfo::~OptmizerMetaInfo() {}

void OptmizerMetaInfo::SetType(const std::string &type) {
  type_ = OptTypeStr2Enum(type);
}

float OptmizerMetaInfo::GetAttr(const std::string &attr,
                                float default_value) const {
  if (attrs_.count(attr) == 0) {
    return default_value;
  }
  return attrs_.at(attr);
}

void OptmizerMetaInfo::SetAttr(const std::string &attr, float value) {
  attrs_[attr] = value;
}

OptimizerType OptTypeStr2Enum(const std::string type) {
  if (type == "sgd") {
    return OptimizerType::SGD;
  } else if (type == "adam") {
    return OptimizerType::Adam;
  } else if (type == "lamb") {
    return OptimizerType::Lamb;
  } else {
    return OptimizerType::Undefined;
  }
}

std::unique_ptr<popart::Optimizer> GetPopartOptimizer(
    const OptmizerMetaInfo &opt_meta_info) {
  auto opt_type = opt_meta_info.GetType();
  PADDLE_ENFORCE_NE(
      opt_type, OptimizerType::Undefined,
      platform::errors::InvalidArgument("Optimizer type have not been set."));

  if (opt_type == OptimizerType::SGD) {
    auto optimizer = std::make_unique<popart::SGD>(
        popart::OptimizerValue(opt_meta_info.GetLR(), false),
        popart::OptimizerValue(popart::SGD::getUnsetWeightDecay()),
        popart::OptimizerValue(popart::SGD::getUnsetMomentum()),
        popart::OptimizerValue(popart::SGD::getUnsetDampening()),
        popart::OptimizerValue(popart::SGD::getUnsetVelocityScaling()),
        popart::OptimizerValue(popart::SGD::getUnsetLossScaling()));
    return optimizer;
  } else if (opt_type == OptimizerType::Adam) {
    auto optimizer = std::make_unique<popart::Adam>(
        popart::OptimizerValue(opt_meta_info.GetLR(), false),
        popart::OptimizerValue(popart::Adam::getUnsetWeightDecay()),
        popart::OptimizerValue(opt_meta_info.GetAttr("beta1"), false),
        popart::OptimizerValue(opt_meta_info.GetAttr("beta2"), false),
        popart::OptimizerValue(opt_meta_info.GetAttr("epsilon"), false),
        popart::OptimizerValue(popart::Adam::getUnsetLossScaling()),
        popart::AdamMode::Adam, popart::WeightDecayMode::Decay,
        popart::DataType::FLOAT, popart::DataType::FLOAT,
        popart::DataType::FLOAT);
    return optimizer;
  } else if (opt_type == OptimizerType::Lamb) {
    auto optimizer = std::make_unique<popart::Adam>(
        popart::OptimizerValue(opt_meta_info.GetLR(), false),
        popart::OptimizerValue(opt_meta_info.GetAttr("weight_decay"), false),
        popart::OptimizerValue(opt_meta_info.GetAttr("beta1"), false),
        popart::OptimizerValue(opt_meta_info.GetAttr("beta2"), false),
        popart::OptimizerValue(opt_meta_info.GetAttr("epsilon"), false),
        popart::OptimizerValue(popart::Adam::getUnsetLossScaling()),
        popart::AdamMode::Lamb, popart::WeightDecayMode::Decay,
        popart::DataType::FLOAT, popart::DataType::FLOAT,
        popart::DataType::FLOAT);
    return optimizer;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Optimizer %d is not implemented now.", static_cast<int>(opt_type)));
  }
}

bool IsOptimizerSupported(OptimizerType type) {
  switch (type) {
    case OptimizerType::SGD:
    case OptimizerType::Adam:
    case OptimizerType::Lamb:
      return true;
    default:
      return false;
  }
}

std::vector<std::pair<std::string, std::string>> GetOptPrePostfix(
    OptimizerType opt_type) {
  // format: {popart_tensor_id, paddle_tensor_id}, ...
  std::vector<std::pair<std::string, std::string>> pre_post_fix;

  switch (opt_type) {
    case OptimizerType::SGD:
      pre_post_fix.push_back(std::make_pair("", ""));
      break;
    case OptimizerType::Adam:
    case OptimizerType::Lamb:
      pre_post_fix.push_back(std::make_pair("", ""));
      pre_post_fix.push_back(std::make_pair("Accl1___", "_moment1_0"));
      pre_post_fix.push_back(std::make_pair("Accl2___", "_moment2_0"));
      pre_post_fix.push_back(std::make_pair("Step___", "_beta1_pow_acc_0"));
      break;
    default:
      pre_post_fix.push_back(std::make_pair("", ""));
      break;
  }

  return pre_post_fix;
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
