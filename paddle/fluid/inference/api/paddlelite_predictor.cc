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

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace paddle {

template <>
std::unique_ptr<PaddlePredictor>
CreatePaddlePredictor<AnalysisConfig, PaddleEngineKind::kPaddleLite>(
    const AnalysisConfig &config) {
  if (config.glog_info_disabled()) {
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 2;  // GLOG_ERROR
  }

  PADDLE_ENFORCE_EQ(
      config.is_valid(),
      true,
      platform::errors::InvalidArgument(
          "Note: Each config can only be used for one predictor."));

  VLOG(3) << "create PaddleLitePredictor";

  std::unique_ptr<PaddlePredictor> predictor(new PaddleLitePredictor(config));
  // Each config can only be used for one predictor.
  config.SetInValid();
  auto predictor_p = dynamic_cast<PaddleLitePredictor *>(predictor.get());

  if (!predictor_p->Init()) {
    return nullptr;
  }

  return predictor;
}

bool PaddleLitePredictor::Init() {
  VLOG(3) << "PaddleLite Predictor::init()";
  return true;
}

std::vector<std::string> PaddleLitePredictor::GetInputNames() {
  std::vector<std::string> input_names;
  // TODO: Use lite api: GetInputNames()
  return input_names;
}

std::map<std::string, std::vector<int64_t>>
PaddleLitePredictor::GetInputTensorShape() {
  std::map<std::string, std::vector<int64_t>> input_shapes;
  // TODO: Use lite api: output_tensor->shape() e.g.:
  // auto output_tensor = predictor->GetOutput(0);
  // auto output_data = output_tensor->data<float>();
  // auto output_size = shape_production(output_tensor->shape());
  return input_shapes;
}

std::vector<std::string> PaddleLitePredictor::GetOutputNames() {
  std::vector<std::string> output_names;
  // TODO: Use lite api: GetOutputNames()
  return output_names;
}

std::unique_ptr<ZeroCopyTensor> PaddleLitePredictor::GetInputTensor(
    const std::string &name) {
  std::unique_ptr<ZeroCopyTensor> res(new ZeroCopyTensor(nullptr, this));
  res->SetName(name);
  // TODO: Add lite api(GetMutableTensor) to ZeroCopyTensor
  return res;
}

std::unique_ptr<ZeroCopyTensor> PaddleLitePredictor::GetOutputTensor(
    const std::string &name) {
  std::unique_ptr<ZeroCopyTensor> res(new ZeroCopyTensor(nullptr, this));
  res->SetName(name);
  // TODO: Add lite api(GetMutableTensor) to ZeroCopyTensor
  return res;
}

bool PaddleLitePredictor::Run(const std::vector<PaddleTensor> &inputs,
                               std::vector<PaddleTensor> *output_data,
                               int batch_size) {
  // TODO: Use lite run()
  return true;
}

bool PaddleLitePredictor::ZeroCopyRun() {
  // TODO: Use lite run()
  return true;
}

std::unique_ptr<PaddlePredictor> PaddleLitePredictor::Clone(void *stream) {
  std::lock_guard<std::mutex> lk(clone_mutex_);
  auto *x = new PaddleLitePredictor(config_,);
  x->Init();
  return std::unique_ptr<PaddlePredictor>(x);
}

uint64_t PaddleLitePredictor::TryShrinkMemory() {
  // TODO: use lite TryShrinkMemory api
  return 0;
}

PaddleLitePredictor::~PaddleLitePredictor() {}

}  // namespace paddle
