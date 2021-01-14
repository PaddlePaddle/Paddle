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

///
/// \file paddle_inference_api.h
///
/// \brief Paddle Inference API
///
/// \author paddle-infer@baidu.com
/// \date 2020-09-01
/// \since 2.0.0-beta
///

namespace paddle_infer {
using DataType = paddle::PaddleDType;
using PlaceType = paddle::PaddlePlace;
using PrecisionType = paddle::AnalysisConfig::Precision;
using Config = paddle::AnalysisConfig;

///
/// \class Tensor
///
/// \brief Represents an n-dimensional array of values.
/// The Tensor is used to store the input or output of the network.
/// It is obtained through Predictor::GetinputHandle()
/// and Predictor::GetOutputHandle() interface.
///
class PD_INFER_DECL Tensor {
 public:
  // Can only be created by predictor->GetInputHandle(cosnt std::string& name)
  // or predictor->GetOutputHandle(cosnt std::string& name)
  Tensor() = delete;
  explicit Tensor(std::unique_ptr<paddle::ZeroCopyTensor>&& tensor)
      : tensor_(std::move(tensor)) {}

  ///
  /// \brief Reset the shape of the tensor.
  /// Generally it's only used for the input tensor.
  /// Reshape must be called before calling mutable_data() or CopyFromCpu()
  /// \param shape The shape to set.
  ///
  void Reshape(const std::vector<int>& shape);

  ///
  /// \brief Copy the host memory to tensor data.
  /// It's usually used to set the input tensor data.
  /// \param data The pointer of the data, from which the tensor will copy.
  ///
  template <typename T>
  void CopyFromCpu(const T* data);

  ///
  /// \brief Get the memory pointer in CPU or GPU with specific data type.
  /// Please Reshape the tensor first before call this.
  /// It's usually used to get input data pointer.
  /// \param place The place of the tensor.
  /// \return The tensor data buffer pointer.
  ///
  template <typename T>
  T* mutable_data(PlaceType place);

  ///
  /// \brief Copy the tensor data to the host memory.
  /// It's usually used to get the output tensor data.
  /// \param[out] data The tensor will copy the data to the address.
  ///
  template <typename T>
  void CopyToCpu(T* data);

  ///
  /// \brief Get the memory pointer directly.
  /// It's usually used to get the output data pointer.
  /// \param[out] place To get the device type of the tensor.
  /// \param[out] size To get the data size of the tensor.
  /// \return The tensor data buffer pointer.
  ///
  template <typename T>
  T* data(PlaceType* place, int* size) const;

  ///
  /// \brief Set lod info of the tensor.
  /// More about LOD can be seen here:
  ///  https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/lod_tensor.html#lodtensor
  /// \param x the lod info.
  ///
  void SetLoD(const std::vector<std::vector<size_t>>& x);

  /// \brief Return the lod info of the tensor.
  std::vector<std::vector<size_t>> lod() const;

  /// \brief Return the data type of the tensor.
  /// It's usually used to get the output tensor data type.
  /// \return The data type of the tensor.
  DataType type() const;

  /// \brief Return the shape of the Tensor.
  std::vector<int> shape() const;

  /// \brief Return the name of the tensor.
  const std::string& name() const;

 private:
  std::unique_ptr<paddle::ZeroCopyTensor> tensor_;
};

///
/// \class Predictor
///
/// \brief Predictor is the interface for model prediction.
///
/// The predictor has the following typical uses:
///
/// Get predictor
/// \code{cpp}
///   auto predictor = CreatePredictor(config);
/// \endcode
///
/// Get input or output names
/// \code{cpp}
///   auto input_names = predictor->GetInputNames();
///   auto output_names = predictor->GetOutputNames();
/// \endcode
///
/// Get input or output handle
/// \code{cpp}
///   auto input_t = predictor->GetInputHandle(input_names[0]);
///   auto output_t = predictor->GetOutputHandle(output_names[0]);
/// \endcode
///
/// Run predictor
/// \code{cpp}
///   predictor->Run();
/// \endcode
///
class PD_INFER_DECL Predictor {
 public:
  Predictor() = delete;
  ~Predictor() {}
  // Use for clone
  explicit Predictor(std::unique_ptr<paddle::PaddlePredictor>&& pred)
      : predictor_(std::move(pred)) {}

  ///
  /// \brief Construct a new Predictor object
  ///
  /// \param[in] Config config
  ///
  explicit Predictor(const Config& config);

  ///
  /// \brief Get the input names
  ///
  /// \return input names
  ///
  std::vector<std::string> GetInputNames();

  ///
  /// \brief Get the Input Tensor object
  ///
  /// \param[in] name input name
  /// \return input tensor
  ///
  std::unique_ptr<Tensor> GetInputHandle(const std::string& name);

  ///
  /// \brief Run the prediction engine
  ///
  /// \return Whether the function executed successfully
  ///
  bool Run();

  ///
  /// \brief Get the output names
  ///
  /// \return output names
  ///
  std::vector<std::string> GetOutputNames();

  ///
  /// \brief Get the Output Tensor object
  ///
  /// \param[in] name otuput name
  /// \return output tensor
  ///
  std::unique_ptr<Tensor> GetOutputHandle(const std::string& name);

  ///
  /// \brief Clone to get the new predictor. thread safe.
  ///
  /// \return get a new predictor
  ///
  std::unique_ptr<Predictor> Clone();

  /// \brief Clear the intermediate tensors of the predictor
  void ClearIntermediateTensor();

  ///
  /// \brief Release all tmp tensor to compress the size of the memory pool.
  /// The memory pool is considered to be composed of a list of chunks, if
  /// the chunk is not occupied, it can be released.
  ///
  /// \return Number of bytes released. It may be smaller than the actual
  /// released memory, because part of the memory is not managed by the
  /// MemoryPool.
  ///
  uint64_t TryShrinkMemory();

 private:
  std::unique_ptr<paddle::PaddlePredictor> predictor_;
};

///
/// \brief A factory to help create predictors.
///
/// Usage:
///
/// \code{.cpp}
/// Config config;
/// ... // change the configs.
/// auto predictor = CreatePredictor(config);
/// \endcode
///
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

///
/// \class PredictorPool
///
/// \brief PredictorPool is a simple encapsulation of Predictor, suitable for
/// use in multi-threaded situations. According to the thread id, the
/// corresponding Predictor is taken out from PredictorPool to complete the
/// prediction.
///
class PD_INFER_DECL PredictorPool {
 public:
  PredictorPool() = delete;
  PredictorPool(const PredictorPool&) = delete;
  PredictorPool& operator=(const PredictorPool&) = delete;

  /// \brief Construct the predictor pool with \param size predictor instances.
  explicit PredictorPool(const Config& config, size_t size = 1);

  /// \brief Get \param id-th predictor.
  Predictor* Retrive(size_t idx);

 private:
  std::shared_ptr<Predictor> main_pred_;
  std::vector<std::unique_ptr<Predictor>> preds_;
};
}  // namespace services
}  // namespace paddle_infer
