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
 * ATTENTION: It requires some C++ features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace paddle {

struct PaddleTensor {
  std::string name;  // variable name.
  std::vector<int> shape;
  std::vector<unsigned char> data;         // bytes of data.
  size_t type{typeid(float).hash_code()};  // hash of type
};

/*
 * A simple Inference API for Paddle. Currently this API might just be used by
 * non-sequence scenerios.
 * TODO(Superjomn) Prepare another API for NLP-related usages.
 */
class PaddlePredictor {
public:
  struct Config;
  PaddlePredictor() = default;
  PaddlePredictor(const PaddlePredictor&) = delete;

  // One drived class should has such a constructor
  // PaddlePredictor(const XConfig& config);
  // The XConfig is a derived class of Config.

  // Predict an record.
  virtual bool Run(const std::vector<PaddleTensor>& inputs,
                   std::vector<PaddleTensor>* output_data) = 0;

  // Clone a predictor that share the model weights, the Cloned predictor should
  // be thread-safe.
  virtual std::unique_ptr<PaddlePredictor> Clone() = 0;

  // Destroy the Predictor.
  virtual ~PaddlePredictor() {}

  friend std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(
      const PaddlePredictor::Config& config);

  // The common configs for all the predictors.
  struct Config {
    enum class EngineKind;

    std::string model_dir;      // path to the model directory.
    bool enable_engine{false};  // Enable to execute (part of) the model on
    // third-party engines.
    EngineKind engine_kind{Config::EngineKind::kNone};

    enum class EngineKind {
      kNone = -1,          // Use the native Fluid facility.
      kAnakin,             // Use Anakin for inference.
      kTensorRT,           // Use TensorRT for inference.
      kAutoMixedAnakin,    // Automatically mix Fluid with Anakin.
      kAutoMixedTensorRT,  // Automatically mix Fluid with TensorRT.
    };
  };
};

// A factory to help create difference predictor.
template <typename ConfigT>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);

}  // namespace paddle
