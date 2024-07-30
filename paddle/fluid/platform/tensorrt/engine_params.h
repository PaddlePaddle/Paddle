/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include <map>
#include <string>
#include <vector>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace platform {
using ShapeMapType = std::map<std::string, std::vector<int>>;
// Use for construct tensorrt engine
struct EngineParams {
  // The max memory size the engine uses.
  int64_t max_workspace_size;

  // The precision of engine.
  phi::DataType precision{phi::DataType::FLOAT32};

  // Use for engine context memory sharing.
  bool context_memory_sharing{false};

  int device_id{0};

  bool use_dla{false};
  int dla_core{0};

  ShapeMapType min_input_shape;
  ShapeMapType max_input_shape;
  ShapeMapType optim_input_shape;
  ShapeMapType min_shape_tensor;
  ShapeMapType max_shape_tensor;
  ShapeMapType optim_shape_tensor;

  bool use_inspector{false};
  std::string engine_info_path{""};
  std::string engine_serialized_data{""};

  //
  // From tensorrt_subgraph_pass, only used for OpConverter.
  //
  bool use_varseqlen{false};
  bool with_interleaved{false};
  std::string tensorrt_transformer_posid;
  std::string tensorrt_transformer_maskid;
  bool enable_low_precision_io{false};
  // Setting the disable_trt_plugin_fp16 to true means that TRT plugin will
  // not run fp16. When running fp16, the output accuracy of the model will be
  // affected, closing the plugin fp16 may bring some improvement on accuracy.
  bool disable_trt_plugin_fp16{false};
  int optimization_level{3};
  bool use_explicit_quantization{false};
  bool allow_build_at_runtime{false};
};

}  // namespace platform
}  // namespace paddle
