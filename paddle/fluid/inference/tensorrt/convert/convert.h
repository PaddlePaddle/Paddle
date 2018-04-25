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

class OpConverter {
 public:
  OpConverter() {}

  void Convert(const framework::OpDesc& op) {
    std::string type = op.Type();
    OpConverter& op_converter = this->register_op_converter_[type];
    op_converter.Convert(op);
  }

  template <typename T>
  static void Register(const std::string key) {
    register_op_converter_[key] = T();
  }
  static std::unordered_map<std::string, OpConverter> register_op_converter_;

  // fluid inference scope
  framework::Scope* scope_;
  // tensorrt input/output tensor list, whose key is the fluid variable name,
  // and value is the pointer position of tensorrt tensor
  std::unordered_map<std::string, nvinfer1::ITensor*> tr_tensors_;
};

#define REGISTER_TRT_OP_CONVETER(op_type, convert_class)                \
  class convert_class : public OpConverter {                            \
   public:                                                              \
    convert_class() { OpConverter::Register<convert_class>(#op_type); } \
    void Convert(const framework::OpDesc& op);                          \
  }

class TensorRTConverter {
 public:
  TensorRTConverter() {}

  // convert fluid block to tensorrt network
  void ConvertBlock(const framework::BlockDesc& block);
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
