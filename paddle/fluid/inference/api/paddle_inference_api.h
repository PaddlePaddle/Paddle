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
 * This file contains the definition of a simple Inference API for Paddle.
 *
 * ATTENTION: It requires some C++11 features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle_analysis_config.h"  // NOLINT
#include "paddle_api.h"              // NOLINT

namespace paddle_infer {
using DataType = paddle::PaddleDType;
using PlaceType = paddle::PaddlePlace;
using PrecisionType = paddle::AnalysisConfig::Precision;
using Config = paddle::AnalysisConfig;

class PD_INFER_DECL Tensor {
 public:
  // Can only be created by predictor->GetInputHandle(cosnt std::string& name)
  // or predictor->GetOutputHandle(cosnt std::string& name)
  Tensor() = delete;
  explicit Tensor(std::unique_ptr<paddle::ZeroCopyTensor>&& tensor)
      : tensor_(std::move(tensor)) {}
  void Reshape(const std::vector<int>& shape);

  template <typename T>
  void CopyFromCpu(const T* data);

  // should add the place
  template <typename T>
  T* mutable_data(PlaceType place);

  template <typename T>
  void CopyToCpu(T* data);

  template <typename T>
  T* data(PlaceType* place, int* size) const;

  void SetLoD(const std::vector<std::vector<size_t>>& x);
  std::vector<std::vector<size_t>> lod() const;

  DataType type() const;

  std::vector<int> shape() const;
  const std::string& name() const;

 private:
  std::unique_ptr<paddle::ZeroCopyTensor> tensor_;
};

class PD_INFER_DECL Predictor {
 public:
  Predictor() = delete;
  ~Predictor() {}
  // Use for clone
  explicit Predictor(std::unique_ptr<paddle::PaddlePredictor>&& pred)
      : predictor_(std::move(pred)) {}

  explicit Predictor(const Config& config);

  std::vector<std::string> GetInputNames();
  std::unique_ptr<Tensor> GetInputHandle(const std::string& name);

  bool Run();

  std::vector<std::string> GetOutputNames();
  std::unique_ptr<Tensor> GetOutputHandle(const std::string& name);

  std::unique_ptr<Predictor> Clone();
  void ClearIntermediateTensor();

 private:
  std::unique_ptr<paddle::PaddlePredictor> predictor_;
};

PD_INFER_DECL std::shared_ptr<Predictor> CreatePredictor(
    const Config& config);  // NOLINT
PD_INFER_DECL int GetNumBytesOfDataType(DataType dtype);

PD_INFER_DECL std::string GetVersion();
PD_INFER_DECL std::string UpdateDllFlag(const char* name, const char* value);

template <typename T>
void Tensor::CopyFromCpu(const T* data) {
  tensor_->copy_from_cpu<T>(data);
}

template <typename T>
void Tensor::CopyToCpu(T* data) {
  return tensor_->copy_to_cpu<T>(data);
}

template <typename T>
T* Tensor::mutable_data(PlaceType place) {
  return tensor_->mutable_data<T>(place);
}

template <typename T>
T* Tensor::data(PlaceType* place, int* size) const {
  return tensor_->data<T>(place, size);
}

}  // namespace paddle_infer

namespace paddle_infer {
namespace services {

class PD_INFER_DECL PredictorPool {
 public:
  PredictorPool() = delete;
  PredictorPool(const PredictorPool&) = delete;
  PredictorPool& operator=(const PredictorPool&) = delete;

  explicit PredictorPool(const Config& config, size_t size = 1);
  Predictor* Retrive(size_t idx);

 private:
  std::shared_ptr<Predictor> main_pred_;
  std::vector<std::unique_ptr<Predictor>> preds_;
};
}  // namespace services
}  // namespace paddle_infer
