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
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_compatible_info.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/platform/device/gpu/gpu_types.h"
#include "paddle/fluid/string/printf.h"

#include "onnxruntime_c_api.h"    // NOLINT
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "paddle2onnx/converter.h"

#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#endif

///
/// \file onnxruntime_predictor.h
///
/// \brief A predictor using ONNXRuntime
///
/// \author heliqi@baidu.com
/// \date 2022-02-14
/// \since 2.3.0
///

namespace paddle {

bool CheckConvertToONNX(const AnalysisConfig &config);

struct ONNXDesc {
  std::string name;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType dtype;
};

///
/// \class ONNXRuntimePredictor
///
/// \brief The ONNXRuntimePredictor using ONNXRuntime for inference
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
class ONNXRuntimePredictor : public PaddlePredictor {
 public:
  ///
  /// \brief Construct a new ONNXRuntime Predictor object
  ///
  /// \param[in] AnalysisConfig config
  ///
  explicit ONNXRuntimePredictor(const AnalysisConfig &config)
      : config_(config), env_(ORT_LOGGING_LEVEL_WARNING, "onnx") {
    predictor_id_ = inference::GetUniqueId();
  }
  ///
  /// \brief Destroy the ONNXRuntime Predictor object
  ///
  ~ONNXRuntimePredictor();

  ///
  /// \brief Initialize predictor
  ///
  /// \return Whether the init function executed successfully
  ///
  bool Init();

  ///
  /// \brief Get the input names
  ///
  /// \return input names
  ///
  std::vector<std::string> GetInputNames();

  ///
  /// \brief Get the output names
  ///
  /// \return output names
  ///
  std::vector<std::string> GetOutputNames();

  ///
  /// \brief Get the Input Tensor object
  ///
  /// \param[in] name input name
  /// \return input tensor
  ///
  std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string &name) override;

  ///
  /// \brief Get the Output Tensor object
  ///
  /// \param[in] name otuput name
  /// \return output tensor
  ///
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string &name) override;
  ///
  /// \brief Get all input names and their corresponding shapes
  ///
  /// \return the map of input names and shapes
  ///
  std::map<std::string, std::vector<int64_t>> GetInputTensorShape() override;

  /// Not supoort
  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

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
  std::unique_ptr<PaddlePredictor> Clone() override;

 private:
  ///
  /// \brief Whether to find in/out by name.
  ///
  /// \param[in] name input or output name
  ///
  /// \param[in] is_input input(true) or output(false)
  ///
  /// \return Whether to find by name
  ///
  bool FindONNXDesc(const std::string &name, bool is_input);

 private:
  AnalysisConfig config_;

  // ONNXRuntime
  Ort::Env env_;
  Ort::Session session_{nullptr};
  std::shared_ptr<Ort::IoBinding> binding_;

  platform::Place place_;
  std::vector<ONNXDesc> input_desc_;
  std::vector<ONNXDesc> output_desc_;
  int predictor_id_;

// Some more detailed tests, they are made the friends of the predictor, so that
// the all the details can be tested.
#if PADDLE_WITH_TESTING
  FRIEND_TEST(ONNXRuntimePredictor, onnxruntime_on);
#endif
};

}  // namespace paddle
