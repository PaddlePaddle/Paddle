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
#include <vector>

#include "framework/core/net/net.h"
#include "framework/graph/graph.h"
#include "paddle/fluid/inference/api/paddle_anakin_config.h"
#include "saber/core/shape.h"
#include "saber/saber_types.h"

namespace paddle {

using contrib::AnakinConfig;

template <typename Target>
class PaddleInferenceAnakinPredictor : public PaddlePredictor {
 public:
  PaddleInferenceAnakinPredictor() : config_() {}

  explicit PaddleInferenceAnakinPredictor(const AnakinConfig& config);

  // NOTE Unlike the native engine, the buffers of anakin engine's output_data
  // should be allocated first.
  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data,
           int batch_size = -1) override;

  std::unique_ptr<PaddlePredictor> Clone() override;
  bool ResetConfig(const AnakinConfig& config);
  anakin::Net<Target, anakin::Precision::FP32, ::anakin::OpRunType::ASYNC>&
  ResetExecuter(
      std::shared_ptr<anakin::graph::Graph<Target, anakin::Precision::FP32>>
          graph_p);

  ~PaddleInferenceAnakinPredictor() override;

 private:
  bool Init();
  bool RunImpl(const std::vector<PaddleTensor>& inputs,
               std::vector<PaddleTensor>* output_data);
  std::mutex mutex_;
  AnakinConfig config_;
  std::shared_ptr<anakin::Context<Target>> ctx_p_;
  std::shared_ptr<anakin::graph::Graph<Target, anakin::Precision::FP32>>
      graph_p_;
  anakin::Net<Target, anakin::Precision::FP32, ::anakin::OpRunType::ASYNC>*
      executor_p_{nullptr};
};

}  // namespace paddle
