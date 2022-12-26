// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/paddlelite_predictor.h"
#include <memory>
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/variable_helper.h"

namespace paddle {

void PaddleLitePredictor::Init() {
  place_ = paddle::platform::CPUPlace();
  scope_.reset(new paddle::framework::Scope());
  // Create paddlelite_config
  CreatePaddleLiteConfigFromAnalysisConfig();
  // Create paddlelite_predictor
  paddlelite_predictor_ = lite_api::CreatePaddlePredictor(paddlelite_config_);
  framework::proto::VarType::Type proto_type =
      framework::proto::VarType::LOD_TENSOR;
  // Create input variables
  for (auto &input_name : GetInputNames()) {
    auto *ptr = scope_->Var(input_name);
    framework::InitializeVariable(ptr, proto_type);
  }
  // Create output variables
  for (auto &output_name : GetOutputNames()) {
    LOG(INFO) << "CREATE VARIABLES Output name: " << output_name;
    auto *ptr = scope_->Var(output_name);
    framework::InitializeVariable(ptr, proto_type);
  }
}

void PaddleLitePredictor::CreatePaddleLiteConfigFromAnalysisConfig() {
  // Set model path.
  if (!analysis_config_.prog_file().empty() &&
      !analysis_config_.params_file().empty()) {
    paddlelite_config_.set_model_file(analysis_config_.prog_file());
    paddlelite_config_.set_param_file(analysis_config_.params_file());
  } else if (!analysis_config_.model_dir().empty()) {
    paddlelite_config_.set_model_dir(analysis_config_.model_dir());
  } else {
    LOG(FATAL) << "Please check pro_file and params_file path or model_dir is "
                  "not empty!";
  }
  // Set places.
  auto ConvertToLitePrecision =
      [](const AnalysisConfig::Precision &precision_type)
      -> paddle::lite_api::PrecisionType {
    if (precision_type == paddle_infer::PrecisionType::kFloat32) {
      return paddle::lite_api::PrecisionType::kFloat;
    } else if (precision_type == paddle_infer::PrecisionType::kInt8) {
      return paddle::lite_api::PrecisionType::kInt8;
    } else if (precision_type == paddle_infer::PrecisionType::kHalf) {
      return paddle::lite_api::PrecisionType::kFP16;
    } else {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Not support precision. We now only support Float32, Half, and "
          "Int8."));
      return paddle::lite_api::PrecisionType::kFloat;
    }
  };
  std::vector<paddle::lite_api::Place> valid_places;
  if (analysis_config_.use_xpu()) {
    valid_places.push_back(paddle::lite_api::Place{
        TARGET(kXPU),
        ConvertToLitePrecision(analysis_config_.GetLitePrecisionMode())});
  }
  valid_places.push_back(paddle::lite_api::Place{
      TARGET(kHost),
      ConvertToLitePrecision(analysis_config_.GetLitePrecisionMode())});
  paddlelite_config_.set_valid_places(valid_places);
}

std::vector<std::string> PaddleLitePredictor::GetInputNames() {
  return paddlelite_predictor()->GetInputNames();
}

std::vector<std::string> PaddleLitePredictor::GetOutputNames() {
  return paddlelite_predictor()->GetOutputNames();
}

std::map<std::string, std::vector<int64_t>>
PaddleLitePredictor::GetInputTensorShape() {
  std::map<std::string, std::vector<int64_t>> input_shapes;
  std::vector<std::string> names = GetInputNames();
  for (std::string name : names) {
    auto tensor = paddlelite_predictor()->GetInputByName(name);
    input_shapes[name] = tensor->shape();
  }
  return input_shapes;
}

std::unique_ptr<ZeroCopyTensor> PaddleLitePredictor::GetInputTensor(
    const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(scope_->FindVar(name),
                          platform::errors::PreconditionNotMet(
                              "The input variable named %s is not found in the "
                              "PaddleLitePredictor.",
                              name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(scope_.get()), this));
  res->input_or_output_ = true;
  res->SetName(name);
  res->SetPlace(PaddlePlace::kCPU);
  return res;
}

std::unique_ptr<ZeroCopyTensor> PaddleLitePredictor::GetOutputTensor(
    const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(
      scope_->FindVar(name),
      platform::errors::PreconditionNotMet(
          "The output variable named %s is not found in the "
          "PaddleLitePredictor.",
          name));
  std::unique_ptr<ZeroCopyTensor> res(
      new ZeroCopyTensor(static_cast<void *>(scope_.get()), this));
  res->input_or_output_ = true;
  res->SetName(name);
  res->SetPlace(PaddlePlace::kCPU);
  return res;
}

std::unique_ptr<PaddlePredictor> PaddleLitePredictor::Clone(void *stream) {
  LOG(ERROR) << "PaddleLitePredictor does not implemented Clone method.";
  return nullptr;
}

bool PaddleLitePredictor::Run(const std::vector<PaddleTensor> &inputs,
                              std::vector<PaddleTensor> *output_data,
                              int batch_size) {
  LOG(ERROR) << "PaddleLite predictor not support Run(), please use "
                "ZeroCopyRun() instead.";
  return false;
}

bool PaddleLitePredictor::ZeroCopyRun() {
  // Share zero_copy_tensor data to lite tensor, this function should be called
  // after CopyFromCpu.
  for (auto &input_name : GetInputNames()) {
    auto zero_copy_tensor = GetInputTensor(input_name);
    std::unique_ptr<lite_api::Tensor> lite_tensor =
        paddlelite_predictor()->GetInputByName(input_name);
    auto tensor_shape = zero_copy_tensor->shape();
    lite_tensor->Resize(
        std::vector<int64_t>(tensor_shape.begin(), tensor_shape.end()));
    auto numel = std::accumulate(tensor_shape.begin(),
                                 tensor_shape.end(),
                                 1,
                                 std::multiplies<int64_t>());
    size_t tensor_mermory_size =
        numel * lite_api::PrecisionTypeLength(lite_tensor->precision());
#define SHARE_ZERO_COPY_TENSOR_MEMORY_TO_LITE_TENSOR(paddle_infer_data_type, \
                                                     pod_data_type)          \
  case paddle_infer::DataType::paddle_infer_data_type: {                     \
    lite_tensor->ShareExternalMemory(                                        \
        static_cast<void *>(zero_copy_tensor->mutable_data<pod_data_type>(   \
            zero_copy_tensor->place())),                                     \
        tensor_mermory_size,                                                 \
        lite_api::TargetType::kHost);                                        \
    break;                                                                   \
  };

    switch (zero_copy_tensor->type()) {
      SHARE_ZERO_COPY_TENSOR_MEMORY_TO_LITE_TENSOR(FLOAT32, float)
      SHARE_ZERO_COPY_TENSOR_MEMORY_TO_LITE_TENSOR(INT64, int64_t)
      SHARE_ZERO_COPY_TENSOR_MEMORY_TO_LITE_TENSOR(INT32, int)
      SHARE_ZERO_COPY_TENSOR_MEMORY_TO_LITE_TENSOR(UINT8, uint8_t)
      SHARE_ZERO_COPY_TENSOR_MEMORY_TO_LITE_TENSOR(INT8, int8_t)
      default:
        LOG(ERROR) << "Share zero_copy_tensor data memory to lite tensor only "
                      "support [FLOAT32, INT64, INT32, UINT8, INT8]";
    }
#undef SHARE_ZERO_COPY_TENSOR_MEMORY_TO_LITE_TENSOR
  }
  // Run
  paddlelite_predictor()->Run();
  // Share lite tensor data to zero_copy_tensor
  for (auto &output_name : GetOutputNames()) {
    auto zero_copy_tensor = GetOutputTensor(output_name);
    auto lite_output_tensor = paddlelite_predictor()->GetTensor(output_name);
#define SHARE_LITE_TENSOR_MEMORY_TO_ZERO_COPY_TENSOR(lite_tensor_data_type,    \
                                                     pod_data_type)            \
  case paddle::lite_api::PrecisionType::lite_tensor_data_type: {               \
    auto lite_tensor_data = lite_output_tensor->mutable_data<pod_data_type>(); \
    auto lite_tensor_shape = lite_output_tensor->shape();                      \
    zero_copy_tensor->ShareExternalData<pod_data_type>(                        \
        static_cast<pod_data_type *>(lite_tensor_data),                        \
        std::vector<int>(lite_tensor_shape.begin(), lite_tensor_shape.end()),  \
        zero_copy_tensor->place());                                            \
    break;                                                                     \
  }
    switch (lite_output_tensor->precision()) {
      SHARE_LITE_TENSOR_MEMORY_TO_ZERO_COPY_TENSOR(kFloat, float)
      SHARE_LITE_TENSOR_MEMORY_TO_ZERO_COPY_TENSOR(kInt8, int8_t)
      SHARE_LITE_TENSOR_MEMORY_TO_ZERO_COPY_TENSOR(kInt32, int)
      SHARE_LITE_TENSOR_MEMORY_TO_ZERO_COPY_TENSOR(kInt64, int64_t)
      default: {
        LOG(ERROR) << "Share lite tensor data memory to zero_copy_tensor only "
                      "support lite precision [kFloat, kInt8, kInt32, kInt64].";
      }
    }
#undef SHARE_LITE_TENSOR_MEMORY_TO_ZERO_COPY_TENSOR
  }
  return true;
}

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kPaddleLite>(
    const AnalysisConfig &config) {
  PADDLE_ENFORCE_EQ(
      config.is_valid(),
      true,
      platform::errors::InvalidArgument(
          "Note: Each config can only be used for one predictor."));
  VLOG(3) << "Create paddlelite predictor with analysisConfig";
  std::unique_ptr<PaddlePredictor> predictor(new PaddleLitePredictor(config));
  auto predictor_p = dynamic_cast<PaddleLitePredictor *>(predictor.get());
  predictor_p->Init();
  // Each config can only be used for one predictor.
  config.SetInValid();
  return predictor;
}

}  // namespace paddle
