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

#include <cuda.h>

#include "paddle/contrib/inference/paddle_inference_api_anakin_engine.h"

namespace paddle {

PaddleInferenceAnakinPredictor::PaddleInferenceAnakinPredictor(
    const AnakinConfig &config) {
  CHECK(Init(config));
}

bool PaddleInferenceAnakinPredictor::Init(const AnakinConfig &config) {
  // TODO(Superjomn) Tell anakin to support return code.
  engine_.Build(config.model_file, config.max_batch_size);
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
    engine_.SetInputFromCPU(
        input.name, static_cast<float *>(input.data.data), input.data.length);
  }

  // TODO(Superjomn) Tell anakin to support return code.
  engine_.Execute();

  if (output_data->empty()) {
    LOG(ERROR) << "At least one output should be set with tensors' names.";
    return false;
  }
  for (auto &output : *output_data) {
    auto *tensor = engine_.GetOutputInGPU(output.name);
    output.shape = tensor->shape();
    // Copy data from GPU -> CPU
    if (cudaMemcpy(output.data.data,
                   tensor->data(),
                   tensor->size(),
                   cudaMemcpyDeviceToHost) != 0) {
      LOG(ERROR) << "copy data from GPU to CPU error";
      return false;
    }
  }
  return true;
}

// TODO(Superjomn) To implement latter.
std::unique_ptr<PaddlePredictor> PaddleInferenceAnakinPredictor::Clone() {
  return nullptr;
}

// A factory to help create difference predictor.
template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnakinConfig, PaddleEngineKind::kAnakin>(
    const AnakinConfig &config) {
  std::unique_ptr<PaddlePredictor> x(
      new PaddleInferenceAnakinPredictor(config));
  return x;
};

}  // namespace paddle
