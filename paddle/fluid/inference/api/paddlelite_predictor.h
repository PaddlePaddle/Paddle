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

#include "lite/api/paddle_api.h"
#include "lite/api/paddle_place.h"
#include "lite/api/paddle_use_passes.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_api.h"

namespace paddle {

///
/// \class PaddleLitePredictor
///
/// \brief The paddlelite predictor is designed to directly call the lite
/// engine, without accessing the paddlelite through the subgraph method.
///
/// The predictor has the following typical uses:
///
/// Get predictor.
/// \code{cpp}
///   auto predictor = CreatePaddlePredictor(config);
/// \endcode
///
/// Get input or output names.
/// \code{cpp}
///   auto input_names = predictor->GetInputNames();
///   auto output_names = predictor->GetOutputNames();
/// \endcode
///
/// Get input or output tensors.
/// \code{cpp}
///   auto input_t = predictor->GetInputTensor(input_names[0]);
///   auto output_t = predictor->GetOutputTensor(output_names[0]);
/// \endcode
///
/// Run predictor.
/// \code{cpp}
///   predictor->ZeroCopyRun();
/// \endcode
///
class PaddleLitePredictor : public PaddlePredictor {
 public:
  ///
  /// \brief Construct a new PaddleLite Predictor object.
  ///
  /// \param[in] AnalysisConfig config
  ///
  explicit PaddleLitePredictor(const AnalysisConfig &config)
      : analysis_config_(config) {
    predictor_id_ = inference::GetUniqueId();
  }

  ///
  /// \brief Initialize the paddlelite predictor.
  ///
  void Init();

  ///
  /// \brief Destroy the PaddleLite Predictor object.
  ///
  ~PaddleLitePredictor() = default;

  ///
  /// \brief Get the input names.
  ///
  /// \return input names.
  ///
  std::vector<std::string> GetInputNames() override;

  ///
  /// \brief Get the output names.
  ///
  /// \return output names.
  ///
  std::vector<std::string> GetOutputNames() override;

  ///
  /// \brief Get all input names and their corresponding shapes.
  ///
  /// \return the map of input names and shapes.
  ///
  std::map<std::string, std::vector<int64_t>> GetInputTensorShape() override;

  ///
  /// \brief Get the Input Tensor object.
  ///
  /// \param[in] name input name.
  /// \return input tensor.
  ///
  std::unique_ptr<ZeroCopyTensor> GetInputTensor(const std::string &name);

  ///
  /// \brief Get the Output Tensor object.
  ///
  /// \param[in] name otuput name.
  /// \return output tensor.
  ///
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(const std::string &name);

  ///
  /// \brief Get the native paddlelite predictor created by cxx config by
  /// parsing analysis config.
  ///
  std::shared_ptr<lite_api::PaddlePredictor> paddlelite_predictor() {
    return paddlelite_predictor_;
  }

  ///
  /// \brief Clone to get the new predictor. thread safe.
  ///
  /// \return get a new predictor
  ///
  std::unique_ptr<PaddlePredictor> Clone(void *stream = nullptr) override;

  ///
  /// \brief Run the paddlelite engine, and call the paddlelite interface
  /// internally
  ///
  /// \return Whether the function executed successfully
  ///
  bool ZeroCopyRun() override;

  /// Not support, please use ZeroCopyRun().
  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

 private:
  ///
  /// \brief Create the paddlelite cxx config by parsing the analysis config.
  ///
  void CreatePaddleLiteConfigFromAnalysisConfig();

 private:
  AnalysisConfig analysis_config_;
  lite_api::CxxConfig paddlelite_config_;
  std::shared_ptr<lite_api::PaddlePredictor> paddlelite_predictor_;
  platform::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  int predictor_id_;
};

}  // namespace paddle
