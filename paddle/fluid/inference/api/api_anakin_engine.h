/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
  PaddleInferenceAnakinPredictor() = default;

  explicit PaddleInferenceAnakinPredictor(const AnakinConfig& config)
      : config_(config) {
    this->InitPredictor();
  }

  // NOTE Unlike the native engine, the buffers of anakin engine's output_data
  // should be allocated first.
  bool Run(const std::vector<PaddleTensor>& inputs,
           std::vector<PaddleTensor>* output_data,
           int batch_size = -1) override;

  std::unique_ptr<PaddlePredictor> Clone() override;
  bool Reset(PaddleInferenceAnakinPredictor<T, P, R>* predictor);
  void InitPredictor();
  std::shared_ptr<anakin::graph::Graph<T, P>> GetGraph() {
    return this->graph_p_;
  }
  std::vector<std::string> GetInputNames() override {
    return this->input_names_;
  }
  std::vector<std::string> GetOutputNames() override {
    return this->output_names_;
  }
  const AnakinConfig& GetConfig() const { return this->config_; }

  ~PaddleInferenceAnakinPredictor() override;

 protected:
  void InitEnv();
  void InitGraph();
  virtual void OptimizeGraph();
  virtual void InitNet();
  virtual void SetContext();
  virtual void Predict();
  virtual std::unique_ptr<PaddlePredictor> New();
  static std::mutex mutex_;
  AnakinConfig config_;
  std::shared_ptr<anakin::Context<T>> ctx_p_;
  std::shared_ptr<anakin::graph::Graph<T, P>> graph_p_;
  anakin::Net<T, P, R>* executor_p_{nullptr};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

 private:
  bool RunImpl(const std::vector<PaddleTensor>& inputs,
               std::vector<PaddleTensor>* output_data);
  static std::once_flag init_anakin_;
};

#ifdef ANAKIN_MLU_PLACE
template <Precision P, OpRunType R>
class PaddleInferenceAnakinMLUPredictor final
    : public PaddleInferenceAnakinPredictor<anakin::MLU, P, R> {
 public:
  PaddleInferenceAnakinMLUPredictor() = default;
  explicit PaddleInferenceAnakinMLUPredictor(const AnakinConfig& config) {
    this->config_ = config;
    this->InitPredictor();
  }
  std::unique_ptr<PaddlePredictor> New() override;
  void SetContext() override;
  void OptimizeGraph() override;
  void InitNet() override;
  void Predict() override;
};
#endif

#ifdef ANAKIN_BM_PLACE
template <Precision P, OpRunType R>
class PaddleInferenceAnakinBMPredictor final
    : public PaddleInferenceAnakinPredictor<anakin::BM, P, R> {
 public:
  PaddleInferenceAnakinBMPredictor() = default;
  explicit PaddleInferenceAnakinBMPredictor(const AnakinConfig& config) {
    this->config_ = config;
    this->InitPredictor();
  }
  std::unique_ptr<PaddlePredictor> New() override;
  void OptimizeGraph() override;
  void InitNet() override;
  void Predict() override;
};
#endif
}  // namespace paddle
