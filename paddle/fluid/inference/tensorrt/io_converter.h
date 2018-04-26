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

#include <unordered_map>
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace inference {
namespace tensorrt {

using framework::LoDTensor;

/*
 * Convert Input from Fluid to an Engine.
 * TensorRT's ITensor follows row major, NCHW. Fluid is also row major, so in
 * most cases just need to copy the data.
 */
class EngineInputConverter {
 public:
  virtual void operator()(LoDTensor& in, void* out, size_t max_size) = 0;
  static void Execute(const std::string& in_op_type, const LoDTensor& in,
                      void* out, size_t max_size) {
    conveters_[in_op_type](in, out, max_size);
  }

  static void Register(const std::string& key, const EngineInputConverter& v) {
    conveters_[key] = v;
  }

 private:
  static std::unordered_map<std::string, EngineInputConveter> conveters_;
};

#define REGISTER_TRT_INPUT_CONVERTER(in_op_type__, Conveter__)      \
  struct trt_input_##in_op_type__##_conveter {                      \
    trt_input_##in_op_type__##_conveter() {                         \
      ::paddle::inference::tensorrt::EngineInputConveter::Register( \
          #in_op_type__, Conveter__());                             \
    }                                                               \
  };                                                                \
  static trt_input_##in_op_type__##_conveter                        \
      trt_input_##in_op_type__##_conveter__;

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
