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

#include "paddle/fluid/inference/api/distmodel_predictor.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle_infer {
template <typename T>
void DistModelDataBuf::CopyFromCpu(T* data) {
  size_ = len_ * sizeof(T);
  data_.resize(size_);
  std::memcpy(data_.data(), reinterpret_cast<char*>(data), size_);
  if (std::is_same<T, phi::dtype::float16>::value)
    dtype_ = DistModelDataType::DIST_FLOAT16;
  if (std::is_same<T, float>::value) dtype_ = DistModelDataType::DIST_FLOAT32;
  if (std::is_same<T, int64_t>::value) dtype_ = DistModelDataType::DIST_INT64;
  if (std::is_same<T, int32_t>::value) dtype_ = DistModelDataType::DIST_INT32;
  if (std::is_same<T, int8_t>::value) dtype_ = DistModelDataType::DIST_INT8;
}

template PD_INFER_DECL void DistModelDataBuf::CopyFromCpu<float>(float* data);
template PD_INFER_DECL void DistModelDataBuf::CopyFromCpu<int64_t>(
    int64_t* data);
template PD_INFER_DECL void DistModelDataBuf::CopyFromCpu<int32_t>(
    int32_t* data);
template PD_INFER_DECL void DistModelDataBuf::CopyFromCpu<uint8_t>(
    uint8_t* data);
template PD_INFER_DECL void DistModelDataBuf::CopyFromCpu<int8_t>(int8_t* data);
template PD_INFER_DECL void DistModelDataBuf::CopyFromCpu<phi::dtype::float16>(
    phi::dtype::float16* data);

DistModelPredictor::DistModelPredictor(const DistModelPredictorConfig& config) {
  paddle::distributed::DistModelConfig dist_config;
  // Copy config
  dist_config.model_dir = config.model_dir;
  dist_config.place = config.place;
  // Create distmodel
  predictor_ = std::make_unique<paddle::distributed::DistModel>(dist_config);
  // Init
  paddle::framework::InitDevices();
  paddle::framework::InitDefaultKernelSignatureMap();
  predictor_->Init();
  // Create Input And Output Tensors
  const std::map<int64_t, std::string>& idx_to_feeds =
      predictor_->GetIdxToFeeds();
  for (const std::pair<int64_t, std::string>& idx_to_feed : idx_to_feeds) {
    paddle::distributed::DistModelTensor input_tensor;
    input_tensor.name = idx_to_feed.second;
    input_tensors_[input_tensor.name] = input_tensor;
  }

  const std::map<int64_t, std::string>& idx_to_fetches =
      predictor_->GetIdxToFetches();
  for (const std::pair<int64_t, std::string>& idx_to_fetch : idx_to_fetches) {
    paddle::distributed::DistModelTensor output_tensor;
    output_tensor.name = idx_to_fetch.second;
    output_tensors_[output_tensor.name] = output_tensor;
  }
}

bool DistModelPredictor::Run() {
  std::vector<paddle::distributed::DistModelTensor> inputs;
  for (const auto& input_tensor : input_tensors_) {
    inputs.push_back(input_tensor.second);
  }
  std::vector<paddle::distributed::DistModelTensor> outputs;
  bool res = predictor_->Run(inputs, &outputs);
  for (auto& output : outputs) {
    output_tensors_[output.name] = std::move(output);
  }
  run_flag_ = true;
  return res;
}

std::vector<std::string> DistModelPredictor::GetInputNames() {
  std::vector<std::string> res;
  for (const auto& input_tensor : input_tensors_) {
    res.push_back(input_tensor.first);
  }
  return res;
}

void DistModelPredictor::SetInput(const std::string& name,
                                  DistModelDataBuf* data_buf,
                                  std::vector<std::vector<size_t>> lod) {
  if (input_tensors_.count(name) == 0) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Cannot find input %s in model.", name));
  }
  paddle::distributed::DistModelTensor& input_tensor = input_tensors_.at(name);

  paddle::distributed::DistModelDataBuf dist_data_buf(
      reinterpret_cast<void*>(data_buf->data()), data_buf->size());
  input_tensor.data = std::move(dist_data_buf);
  input_tensor.dtype = static_cast<paddle::distributed::DistModelDataType>(
      static_cast<int>(data_buf->dtype()));
  input_tensor.shape = data_buf->shape();
  input_tensor.lod = lod;

  run_flag_ = false;  // reset
}

std::vector<std::string> DistModelPredictor::GetOutputNames() {
  std::vector<std::string> res;
  for (const auto& output_tensor : output_tensors_) {
    res.push_back(output_tensor.first);
  }
  return res;
}

std::vector<int> DistModelPredictor::GetOutputShape(const std::string& name) {
  if (!run_flag_) {
    LOG(ERROR) << "Please invoke Run method firstly.";
    return {};
  }
  if (output_tensors_.count(name) == 0) {
    LOG(ERROR) << "Cannot find input " << name << " in model.";
    return {};
  }
  return output_tensors_.at(name).shape;
}

DistModelDataBuf DistModelPredictor::GetOutputData(const std::string& name) {
  if (!run_flag_) {
    LOG(ERROR) << "Please invoke Run method firstly.";
    return {};
  }
  if (output_tensors_.count(name) == 0) {
    LOG(ERROR) << "Cannot find input " << name << " in model.";
    return {};
  }
  DistModelDataBuf buf;
  buf.Reshape(output_tensors_.at(name).shape);
  buf.SetDtype(static_cast<int>(output_tensors_.at(name).dtype));
  switch (buf.dtype()) {
    case DistModelDataType::DIST_FLOAT16:
      buf.CopyFromCpu(reinterpret_cast<phi::dtype::float16*>(
          output_tensors_.at(name).data.data()));
      break;
    case DistModelDataType::DIST_FLOAT32:
      buf.CopyFromCpu(
          reinterpret_cast<float*>(output_tensors_.at(name).data.data()));
      break;
    case DistModelDataType::DIST_INT64:
      buf.CopyFromCpu(
          reinterpret_cast<int64_t*>(output_tensors_.at(name).data.data()));
      break;
    case DistModelDataType::DIST_INT32:
      buf.CopyFromCpu(
          reinterpret_cast<int32_t*>(output_tensors_.at(name).data.data()));
      break;
    case DistModelDataType::DIST_INT8:
      buf.CopyFromCpu(
          reinterpret_cast<int8_t*>(output_tensors_.at(name).data.data()));
      break;
    default:
      PADDLE_THROW(
          paddle::platform::errors::Unimplemented("Not supported datatype."));
      break;
  }

  return buf;
}

std::shared_ptr<DistModelPredictorBase> CreateDistModelPredictor(
    const DistModelPredictorConfig& config) {
  std::shared_ptr<DistModelPredictorBase> pred(new DistModelPredictor(config));
  return pred;
}

}  // namespace paddle_infer
