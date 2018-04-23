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

#include <NvInfer.h>
#include <functional>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class TensorRTConverter {
 public:
  explicit TensorRTConverter(const framework::Scope& scope) : scope_(scope) {
    this->RegisterOpConverters();
  }

  // convert fluid op to tensorrt layer
  void ConvertOp(const framework::OpDesc& op);

  // convert fluid block to tensorrt network
  void ConvertBlock(const framework::BlockDesc& block);

 private:
  // convert op registry, whose key is the fluid op type, and value is the
  // convert tensorrt function name
  std::unordered_map<std::string, std::function<void(const framework::OpDesc&)>>
      op_registry_;
  // fluid inference scope
  const framework::Scope& scope_;
  // tensorrt input/output tensor list, whose key is the fluid variable name,
  // and value is the pointer position of tensorrt tensor
  std::unordered_map<std::string, nvinfer1::ITensor*> tr_tensors_;

  // register different op converters
  void RegisterOpConverters();

  // convert a fluid Mul op to tensorrt fc layer without bias
  static void ConvertMul(const framework::OpDesc& op);

  // convert a fluid Conv2d op to tensorrt conv layer without bias
  static void ConvertConv2D(const framework::OpDesc& op);
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
