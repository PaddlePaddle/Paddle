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
using anakin::Precision;
using anakin::OpRunType;

template <typename T, Precision P, OpRunType R>
class PaddleInferenceAnakinPredictor : public PaddlePredictor {
 public:
  PaddleInferenceAnakinPredictor() : config_() { InitEnv(); }

  explicit PaddleInferenceAnakinPredictor(const AnakinConfig& config);

  // NOTE Unlike the native engine, the buffers of anakin engine's output_data
  // should be allocated first.
  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data,
           int batch_size = -1) override;

  std::unique_ptr<PaddlePredictor> Clone() override;
  bool ResetConfig(const AnakinConfig& config);
  anakin::Net<T, P, R>& ResetExecuter(
      std::shared_ptr<anakin::graph::Graph<T, P>> graph_p);

  ~PaddleInferenceAnakinPredictor() override;

 private:
  void InitPredictor();
  void InitEnv();
  void InitGraph();
  void OptimizeGraph();
  void InitNet();
  void SetContext();
  bool RunImpl(const std::vector<PaddleTensor>& inputs,
               std::vector<PaddleTensor>* output_data);
  void Predict();
  static std::mutex mutex_;
  static std::once_flag init_anakin_;
  AnakinConfig config_;
  std::shared_ptr<anakin::Context<T>> ctx_p_;
  std::shared_ptr<anakin::graph::Graph<T, P>> graph_p_;
  anakin::Net<T, P, R>* executor_p_{nullptr};
};

}  // namespace paddle
