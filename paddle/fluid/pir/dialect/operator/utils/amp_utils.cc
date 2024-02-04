// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/utils/amp_utils.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"

namespace paddle {
namespace dialect {

phi::DataType GetPromoteType(
    const std::string& op_name,
    const std::vector<std::vector<pir::Value>>& amp_values_vector,
    const phi::DataType& amp_dtype) {
  auto dst_type = amp_dtype;
  // only consider the dtype of input(X).
  if (op_name == "batch_norm" || op_name == "layer_norm" ||
      op_name == "sync_batch_norm" ||
      op_name == "moving_average_abs_max_scale") {
    if (GetValueDataType(amp_values_vector[0][0]) == phi::DataType::FLOAT32) {
      dst_type = phi::DataType::FLOAT32;
    }
    return dst_type;
  }

  if (egr::Controller::Instance().GetCurrentAMPState()->GetAmpDtype() ==
      "float16") {
    if (op_name == "fused_attention") {
      for (size_t i = 0; i < amp_values_vector.size(); i++) {
        if (i != 3 || i != 4 || i != 9 || i != 10) {
          if (GetValueDataType(amp_values_vector[i][0]) ==
              phi::DataType::FLOAT32) {
            dst_type = phi::DataType::FLOAT32;
            return dst_type;
          }
        }
      }
    } else if (op_name == "fused_feedforward") {
      for (size_t i = 0; i < amp_values_vector.size(); i++) {
        if (i != 7 || i != 8 || i != 9 || i != 10) {
          if (GetValueDataType(amp_values_vector[i][0]) ==
              phi::DataType::FLOAT32) {
            dst_type = phi::DataType::FLOAT32;
            return dst_type;
          }
        }
      }
    }
  }

  for (const auto& values : amp_values_vector) {
    for (const auto& value : values) {
      if (GetValueDataType(value) == phi::DataType::FLOAT32) {
        dst_type = GetValueDataType(value);
        break;
      }
    }
  }

  return dst_type;
}

pir::Value Cast(const pir::Value& input, const phi::DataType& dst_dtype) {
  paddle::imperative::AutoCastGuard guard(
      egr::Controller::Instance().GetCurrentAMPState(),
      paddle::imperative::AmpLevel::O0);
  return paddle::dialect::cast(input, dst_dtype);
}

bool NeedCast(const pir::Value& value, const phi::DataType& dst_dtype) {
  auto data_type = GetValueDataType(value);
  if ((data_type == phi::DataType::FLOAT32 ||
       data_type == phi::DataType::FLOAT16 ||
       data_type == phi::DataType::BFLOAT16) &&
      (data_type != dst_dtype)) {
    return true;
  }
  return false;
}

pir::Value PirAmpAutoCast(const std::string& input_name,
                          const pir::Value& input,
                          const phi::DataType& dst_dtype,
                          const std::string& op_name) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " input(" << input_name << " to dst_dtype("
          << phi::DataTypeToString(dst_dtype) << ").";
  if ((op_name == "batch_norm" || op_name == "layer_norm" ||
       op_name == "sync_batch_norm" || op_name == "weight_only_linear") &&
      input_name != "x") {
    return input;
  }

  if (dst_dtype == phi::DataType::FLOAT16) {
    if (op_name == "run_program") {
      return input;
    }
    if ((op_name == "fused_attention" || op_name == "fused_feedforward")) {
      if (input_name == "LnScale" || input_name == "LnBias" ||
          input_name == "Ln2Scale" || input_name == "Ln2Bias" ||
          input_name == "Ln1Scale" || input_name == "Ln1Bias") {
        return input;
      }
    }
  }
  if (NeedCast(input, dst_dtype)) {
    VLOG(6) << "Input : " << input_name << "NeedCast";
    return Cast(input, dst_dtype);
  }
  return input;
}

paddle::optional<pir::Value> PirAmpAutoCast(
    const std::string& input_name,
    const paddle::optional<pir::Value>& input,
    const phi::DataType& dst_dtype,
    const std::string& op_name) {
  if (input) {
    return PirAmpAutoCast(input_name, *input, dst_dtype, op_name);
  }
  return paddle::none;
}

std::vector<pir::Value> PirAmpAutoCast(const std::string& inputs_name,
                                       const std::vector<pir::Value>& inputs,
                                       const phi::DataType& dst_dtype,
                                       const std::string& op_name) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " inputs(" << inputs_name << ") dst_dtype("
          << phi::DataTypeToString(dst_dtype) << ").";
  std::vector<pir::Value> inputs_casted;
  for (auto& input : inputs) {
    if (NeedCast(input, dst_dtype)) {
      inputs_casted.emplace_back(std::move(Cast(input, dst_dtype)));
    } else {
      inputs_casted.emplace_back(input);
    }
  }
  return inputs_casted;
}

paddle::optional<std::vector<pir::Value>> PirAmpAutoCast(
    const std::string& inputs_name,
    const paddle::optional<std::vector<pir::Value>>& inputs,
    const phi::DataType& dst_dtype,
    const std::string& op_name) {
  if (inputs) {
    return PirAmpAutoCast(inputs_name, *inputs, dst_dtype, op_name);
  }
  return paddle::optional<std::vector<pir::Value>>();
}

phi::DataType GetAmpDestDtype(
    const std::string& op_name,
    const std::vector<std::vector<pir::Value>>& amp_values_vector) {
  auto amp_level = egr::Controller::Instance().GetAMPLevel();
  auto amp_setting_dtype =
      egr::Controller::Instance().GetCurrentAMPState()->GetAmpPhiDtype();
  auto dst_type = amp_setting_dtype;

  bool use_promote = true;
  if (amp_level == paddle::imperative::AmpLevel::O2) {
    use_promote =
        egr::Controller::Instance().GetCurrentAMPState()->GetUsePromote();
  }

  if (use_promote) {
    if (paddle::imperative::AmpOperators::Instance()
            .GetMutableAllowOps()
            ->count(op_name)) {
      dst_type = amp_setting_dtype;
    } else if (paddle::imperative::AmpOperators::Instance()
                   .GetMutableBlockOps()
                   ->count(op_name)) {
      dst_type = phi::DataType::FLOAT32;
    } else {
      if (amp_level == paddle::imperative::AmpLevel::OD) {
        dst_type = phi::DataType::FLOAT32;
      } else {
        dst_type =
            GetPromoteType(op_name, amp_values_vector, amp_setting_dtype);
      }
    }
  } else {
    // use_promote can be set to false only for O2 training.
    if (paddle::imperative::AmpOperators::Instance()
            .GetMutableBlockOps()
            ->count(op_name)) {
      dst_type = phi::DataType::FLOAT32;
    }
  }

  if (dst_type == amp_setting_dtype &&
      (paddle::imperative::AmpOperators::Instance()
           .GetMutableUnsupportedOps(amp_setting_dtype)
           ->count(op_name))) {
    dst_type = phi::DataType::FLOAT32;
  }

  VLOG(6) << "AMP GetAmpDestDtype:"
          << " op(" << op_name << ") amp_dtype(" << dst_type << ") amp_level("
          << static_cast<int>(amp_level) << ").";
  return dst_type;
}
}  // namespace dialect
}  // namespace paddle
