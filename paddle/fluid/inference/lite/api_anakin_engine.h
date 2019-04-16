/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains the implementation of inference API with Anakin engine
 * embeded, this API can only support Anakin models.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "framework/core/net/net.h"
#include "framework/graph/graph.h"
#include "paddle/fluid/inference/lite/anakin_config.h"
#include "saber/core/shape.h"
#include "saber/saber_types.h"

namespace paddle {

using contrib::AnakinConfig;

template <typename Target, anakin::Precision Precision>
class PaddleInferenceAnakinPredictor : public PaddlePredictor {
 public:
  PaddleInferenceAnakinPredictor() {}

  explicit PaddleInferenceAnakinPredictor(const AnakinConfig& config);

  // NOTE Unlike the native engine, the buffers of anakin engine's output_data
  // should be allocated first.
  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data,
           int batch_size = -1) override;

  std::vector<std::string> GetInputNames() override { return graph_.get_ins(); }

  std::vector<std::string> GetOutputNames() override {
    return graph_.get_outs();
  }

  std::unique_ptr<PaddlePredictor> Clone() override;

  anakin::Net<Target, Precision>& get_executer();

  ~PaddleInferenceAnakinPredictor() override;

 private:
  bool Init(const AnakinConfig& config);

  anakin::graph::Graph<Target, Precision> graph_;
  anakin::Net<Target, Precision>* executor_p_{nullptr};
  AnakinConfig config_;
  int max_batch_size_{0};
};

}  // namespace paddle
