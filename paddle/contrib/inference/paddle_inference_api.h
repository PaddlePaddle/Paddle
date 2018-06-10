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
  // TODO(Superjomn) for LoD support, add a vector<vector<int>> field if needed.
  PaddleBuf data;  // blob of data.
  PaddleDType dtype;
};

enum class PaddleEngineKind {
  kNative = 0,  // Use the native Fluid facility.
  kAnakin,      // Use Anakin for inference.
  // TODO(Superjomn) support following engines latter.
  // kTensorRT,           // Use TensorRT for inference.
  // kAutoMixedAnakin,    // Automatically mix Fluid with Anakin.
  // kAutoMixedTensorRT,  // Automatically mix Fluid with TensorRT.
};

/*
 * A simple Inference API for Paddle. Currently this API can be used by
 * non-sequence scenerios.
 */
class PaddlePredictor {
 public:
  struct Config;
  PaddlePredictor() = default;
  PaddlePredictor(const PaddlePredictor&) = delete;
  PaddlePredictor& operator=(const PaddlePredictor&) = delete;

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
  virtual ~PaddlePredictor() = default;

  // The common configs for all the predictors.
  struct Config {
    std::string model_dir;      // path to the model directory.
  };
};

struct NativeConfig : public PaddlePredictor::Config {
  // GPU related fields.
  bool use_gpu{false};
  int device{0};
  float fraction_of_gpu_memory{-1.f};  // Negative to notify initialization.

  std::string prog_file;
  std::string param_file;
};

// Configurations for Anakin engine.
struct AnakinConfig : public PaddlePredictor::Config {
  int device;
  std::string model_file;
  int max_batch_size{-1};
};

// A factory to help create different predictors.
//
// FOR EXTENSION DEVELOPER:
// Different predictors are designated by config type and engine kind. Similar
// configs can be merged, but there shouldn't be a huge config containing
// different fields for more than one kind of predictors.
//
// Similarly, each engine kind should map to a unique predictor implementation.
template <typename ConfigT, PaddleEngineKind engine = PaddleEngineKind::kNative>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT& config);
}  // namespace paddle
