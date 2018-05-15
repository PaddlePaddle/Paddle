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

#pragma once

#include <string>
#include <vector>

namespace paddle {

class Predictor {
public:
  struct Attr;
  Predictor() = default;

  // Build the network before inference.
  bool Init(const Attr& attr);

  // Predict an record.
  // Arguments:
  //   inputs: the name of the input variables.
  //   outputs: the name of the output varaibles.
  //   input_shapes: the shape of the input variables.
  //   output_shapes: the shape of the output variables.
  //   input_data: the data of the input variables.
  //   output_data: the data of the output variables.
  bool Run(const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           const std::vector<std::vector<int>>& input_shapes,
           const std::vector<std::vector<int>>& output_shapes,
           const std::vector<std::vector<float>>& input_data,
           std::vector<std::vector<float>>* output_data);

  // Clone a predictor that share the model weights.
  Predictor* Clone();

  // Destroy the Predictor.
  ~Predictor();

  struct Attr {
    enum class EngineKind;

    std::string model_dir;      // path to the model directory.
    bool enable_engine{false};  // Enable to execute (part of) the model on
                                // third-party engines.
    EngineKind engine_kind{Attr::EngineKind::kNone};

    enum class EngineKind {
      kNone = -1,          // Use the native Fluid facility.
      kAnakin,             // Use Anakin for inference.
      kTensorRT,           // Use TensorRT for inference.
      kAutoMixedAnakin,    // Automatically mix Fluid with Anakin.
      kAutoMixedTensorRT,  // Automatically mix Fluid with TensorRT.
    };
  };
};

}  // namespace paddle
