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

#include "paddle/contrib/inference/paddle_inference_api_anakin_engine.h"
#include <cuda.h>

namespace paddle {

PaddleInferenceAnakinPredictor::PaddleInferenceAnakinPredictor(
    const AnakinConfig &config) {
  CHECK(Init(config));
}

bool PaddleInferenceAnakinPredictor::Init(const AnakinConfig &config) {
  if (!(graph_.load(config.model_file))) {
    return false;
  }
  graph_.ResetBatchSize("input_0", config.max_batch_size);
  // optimization for graph
  if (!(graph_.Optimize())) {
    return false;
  }
  // construct executer
  if (executor_p_ != nullptr) {
    executor_p_ = new anakin::Net<anakin::NV,
                                  anakin::saber::AK_FLOAT,
                                  anakin::Precision::FP32>(graph_, true);
    ;
  }
  return true;
}

bool PaddleInferenceAnakinPredictor::Run(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data) {
  for (const auto &input : inputs) {
    if (input.dtype != PaddleDType::FLOAT32) {
      LOG(ERROR) << "Only support float type inputs. " << input.name
                 << "'s type is not float";
      return false;
    }
    auto d_tensor_in_p = executor_p_->get_in(input.name);
    auto net_shape = d_tensor_in_p->valid_shape();
    if (net_shape.size() != input.shape.size()) {
      LOG(ERROR) << " input  " << input.name
                 << "'s shape size should be equal to that of net";
      return false;
    }
    int sum = 1;
    for_each(input.shape.begin(), input.shape.end(), [&](int n) { sum *= n; });
    if (sum > net_shape.count()) {
      graph_.Reshape(input.name, input.shape);
      delete executor_p_;
      executor_p_ = new anakin::Net<anakin::NV,
                                    anakin::saber::AK_FLOAT,
                                    anakin::Precision::FP32>(graph_, true);
      d_tensor_in_p = executor_p_->get_in(input.name);
    }

    anakin::saber::Shape tmp_shape;
    for (auto s : input.shape) {
      tmp_shape.push_back(s);
    }
    d_tensor_in_p->reshape(tmp_shape);

    float *d_data_p = d_tensor_in_p->mutable_data();
    if (cudaMemcpy(d_data_p,
                   static_cast<float *>(input.data.data()),
                   d_tensor_in_p->valid_size() * sizeof(float),
                   cudaMemcpyHostToDevice) != 0) {
      LOG(ERROR) << "copy data from CPU to GPU error";
      return false;
    }
  }

  executor_p_->prediction();
  cudaDeviceSynchronize();

  if (output_data->empty()) {
    LOG(ERROR) << "At least one output should be set with tensors' names.";
    return false;
  }
  for (auto &output : *output_data) {
    auto *tensor = executor_p_->get_out(output.name);
    output.shape = tensor->valid_shape();
    if (output.data.length() < tensor->valid_size() * sizeof(float)) {
      output.data.Resize(tensor->valid_size() * sizeof(float));
    }
    // Copy data from GPU -> CPU
    if (cudaMemcpy(output.data.data(),
                   tensor->mutable_data(),
                   tensor->valid_size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != 0) {
      LOG(ERROR) << "copy data from GPU to CPU error";
      return false;
    }
  }
  return true;
}

anakin::Net<anakin::NV, anakin::saber::AK_FLOAT, anakin::Precision::FP32>
    &PaddleInferenceAnakinPredictor::get_executer() {
  return *executor_p_;
}

// the cloned new Predictor of anakin share the same net weights from original
// Predictor
std::unique_ptr<PaddlePredictor> PaddleInferenceAnakinPredictor::Clone() {
  VLOG(3) << "Anakin Predictor::clone";
  std::unique_ptr<PaddlePredictor> cls(new PaddleInferenceAnakinPredictor());
  // construct executer from other graph
  auto anakin_predictor_p =
      dynamic_cast<PaddleInferenceAnakinPredictor *>(cls.get());
  if (!anakin_predictor_p) {
    LOG(ERROR) << "fail to call Init";
    return nullptr;
  }
  anakin_predictor_p->get_executer().init(graph_);

  return std::move(cls);
}

// A factory to help create difference predictor.
template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(
    const AnakinConfig &config) {
  VLOG(3) << "Anakin Predictor create.";
  std::unique_ptr<PaddlePredictor> x(
      new PaddleInferenceAnakinPredictor(config));
  return x;
};

}  // namespace paddle
