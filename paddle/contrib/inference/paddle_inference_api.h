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

enum PaddleDType {
  FLOAT32,
  INT64,
};

struct PaddleBuf {
  void* data;     // pointer to the data memory.
  size_t length;  // number of memory bytes.
};

struct PaddleTensor {
  std::string name;  // variable name.
  std::vector<int> shape;
  PaddleBuf data;  // blob of data.
  PaddleDType dtype;
};

/*
 * A simple Inference API for Paddle. Currently this API can be used by
 * non-sequence scenerios.
 * TODO(Superjomn) Support another API for NLP-related usages.
 */
class PaddlePredictor {
 public:
  struct Config;
  PaddlePredictor() = default;
  PaddlePredictor(const PaddlePredictor&) = delete;

  // Predict an record.
  // The caller should be responsible for allocating and releasing the memory of
  // `inputs`. `inputs` should be alive until Run returns. caller should be
  // responsible for releasing the memory of `output_data`.
  virtual bool Run(const std::vector<PaddleTensor>& inputs,
                   std::vector<PaddleTensor>* output_data) = 0;

  // Clone a predictor that share the model weights, the Cloned predictor should
  // be thread-safe.
  virtual std::unique_ptr<PaddlePredictor> Clone() = 0;

  // Destroy the Predictor.
  virtual ~PaddlePredictor() {}

  enum class EngineKind {
    kNative = -1,  // Use the native Fluid facility.
    // TODO(Superjomn) support latter.
    // kAnakin,             // Use Anakin for inference.
    // kTensorRT,           // Use TensorRT for inference.
    // kAutoMixedAnakin,    // Automatically mix Fluid with Anakin.
    // kAutoMixedTensorRT,  // Automatically mix Fluid with TensorRT.
  };

  // The common configs for all the predictors.
  struct Config {
    std::string model_dir;      // path to the model directory.
    bool enable_engine{false};  // Enable to execute (part of) the model on
  };
};

struct NativeConfig : public PaddlePredictor::Config {
  bool use_gpu{false};
  int device;
  float fraction_of_gpu_memory;
  std::string prog_file;
  std::string param_file;
  bool share_variables;
};

// A factory to help create difference predictor.
template <
    typename ConfigT,
    PaddlePredictor::EngineKind engine = PaddlePredictor::EngineKind::kNative>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);

}  // namespace paddle
