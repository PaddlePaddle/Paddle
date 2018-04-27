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
#include <unordered_map>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Convert Op from Fluid to TensorRT Engine.
 */
class OpConverter {
 public:
  OpConverter() {}

  virtual void operator()(const framework::OpDesc& op) {}
  void Execute(const framework::OpDesc& op) {
    std::string type = op.Type();
    auto it = converters_.find(type);
    PADDLE_ENFORCE(it != converters_.end(), "no OpConverter for optype [%s]",
                   type);
    (*it->second)(op);
  }

  static OpConverter& Global() {
    static auto* x = new OpConverter;
    return *x;
  }

  template <typename T>
  void Register(const std::string& key) {
    converters_[key] = new T;
  }

  virtual ~OpConverter() {}

 private:
  // registered op converter map, whose key is the fluid op type, and value is
  // the pointer position of corresponding OpConverter class.
  std::unordered_map<std::string, OpConverter*> converters_;

  // fluid inference scope
  framework::Scope* scope_;
  // tensorrt input/output tensor map, whose key is the fluid variable name,
  // and value is the pointer position of tensorrt tensor
  std::unordered_map<std::string, nvinfer1::ITensor*> tr_tensors_;
};

#define REGISTER_TRT_OP_CONVERTER(op_type__, Converter__)      \
  struct trt_##op_type__##_converter {                         \
    trt_##op_type__##_converter() {                            \
      OpConverter::Global().Register<Converter__>(#op_type__); \
    }                                                          \
  };                                                           \
  trt_##op_type__##_converter trt_##op_type__##_converter__;

class BlockConverter {
 public:
  BlockConverter() {}

  // convert fluid block to tensorrt network
  void ConvertBlock(const framework::BlockDesc& block) {
    for (auto op : block.AllOps()) {
      OpConverter::Global().Execute(*op);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
