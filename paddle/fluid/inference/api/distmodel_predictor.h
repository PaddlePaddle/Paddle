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

#pragma once

#include "paddle/fluid/distributed/fleet_executor/dist_model.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"

namespace paddle_infer {

class DistModelPredictor : public DistModelPredictorBase {
 public:
  explicit DistModelPredictor(const DistModelPredictorConfig& config);

  bool Run() override;

  std::vector<std::string> GetInputNames() override;

  void SetInput(const std::string& name,
                DistModelDataBuf* data_buf,
                std::vector<std::vector<size_t>> lod = {}) override;

  std::vector<std::string> GetOutputNames() override;

  std::vector<int> GetOutputShape(const std::string& name) override;

  DistModelDataBuf GetOutputData(const std::string& name) override;

 private:
  bool run_flag_{false};
  std::unique_ptr<paddle::distributed::DistModel> predictor_;
  std::map<std::string, paddle::distributed::DistModelTensor> input_tensors_;
  std::map<std::string, paddle::distributed::DistModelTensor> output_tensors_;
};

}  // namespace paddle_infer
