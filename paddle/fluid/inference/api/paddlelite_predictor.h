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
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

#include <paddle_api.h>

namespace paddle {

///
/// \class PaddleLitePredictor
///
/// \brief The PaddleLitePredictor using Paddle Lite for inference
///
/// The predictor has the following typical uses:
///
/// Get predictor
/// \code{cpp}
///   auto predictor = CreatePaddlePredictor(config);
/// \endcode
///
/// Get input or output names
/// \code{cpp}
///   auto input_names = predictor->GetInputNames();
///   auto output_names = predictor->GetOutputNames();
/// \endcode
///
/// Get input or output tensors
/// \code{cpp}
///   auto input_t = predictor->GetInputTensor(input_names[0]);
///   auto output_t = predictor->GetOutputTensor(output_names[0]);
/// \endcode
///
/// Run predictor
/// \code{cpp}
///   predictor->ZeroCopyRun();
/// \endcode
///
class PaddleLitePredictor : public PaddlePredictor {
 public:
  ///
  /// \brief Construct a new PaddleLite Predictor object
  ///
  /// \param[in] AnalysisConfig config
  ///
  explicit PaddleLitePredictor(const AnalysisConfig &config)
      : config_(config) {
    predictor_id_ = inference::GetUniqueId();
  }

  ///
  /// \brief Destroy the PaddleLite Predictor object
  ///
  ~PaddleLitePredictor();


  ///
  /// \brief Initialize predictor
  ///
  /// \return Whether the init function executed successfully
  ///
  bool Init();

  ///
  /// \brief Run the prediction engine. Deprecated. Please refer to ZeroCopyRun
  ///
  /// \param[in] inputs input tensors
  /// \param[out] output_data output tensors
  /// \param[in] batch_size data's batch size
  /// \return Whether the function executed successfully
  ///
  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

  ///
  /// \brief Get the input names
  ///
  /// \return input names
  ///
  std::vector<std::string> GetInputNames() override;

  ///
  /// \brief Get the output names
  ///
  /// \return output names
  ///
  std::vector<std::string> GetOutputNames() override;

  ///
  /// \brief Get the Input Tensor object
  ///
  /// \param[in] name input name
  /// \return input tensor
  ///
  std::unique_ptr<ZeroCopyTensor> GetInputTensor(const std::string &name) override;

  ///
  /// \brief Get the Output Tensor object
  ///
  /// \param[in] name otuput name
  /// \return output tensor
  ///
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(const std::string &name) override;

  ///
  /// \brief Get all input names and their corresponding shapes
  ///
  /// \return the map of input names and shapes
  ///
  std::map<std::string, std::vector<int64_t>> GetInputTensorShape() override;

  ///
  /// \brief Run the prediction engine
  ///
  /// \return Whether the function executed successfully
  ///
  bool ZeroCopyRun() override;

  ///
  /// \brief Release all tmp tensor to compress the size of the memory pool.
  /// The memory pool is considered to be composed of a list of chunks, if
  /// the chunk is not occupied, it can be released.
  ///
  /// \return Number of bytes released. It may be smaller than the actual
  /// released memory, because part of the memory is not managed by the
  /// MemoryPool.
  ///
  uint64_t TryShrinkMemory() override;

  ///
  /// \brief Clone to get the new predictor. thread safe.
  ///
  /// \return get a new predictor
  ///
  std::unique_ptr<PaddlePredictor> Clone(void *stream = nullptr) override;

 private:
  AnalysisConfig config_;
  int predictor_id_;
  // A mutex help to make Clone thread safe.
  std::mutex clone_mutex_;
  // PaddleLite predictor.
  std::shared_ptr<paddle::lite_api::PaddlePredictor> lite_predictor_ = nullptr;
};

}  // namespace paddle
