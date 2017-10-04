/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <unordered_map>

#include "paddle/framework/op_info.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/shape_inference.h"

namespace paddle {
namespace framework {

class ShapeInferenceMap {
 public:
  static ShapeInferenceMap& Instance();

  void CreateOpWithKernel(const OpInfo& op_info, const std::string& op_type);

  OperatorWithKernel* GetOpWithKernel(const std::string& op_type) {
    auto it = op_shape_inference_map_.find(op_type);
    if (it == op_shape_inference_map_.end()) {
      return nullptr;
    }
    return it->second;
  }

 private:
  ShapeInferenceMap() = default;
  DISABLE_COPY_AND_ASSIGN(ShapeInferenceMap);

  std::unordered_map<std::string, OperatorWithKernel*> op_shape_inference_map_;
};

}  // namespace framework
}  // namespace paddle
