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
#include "paddle/fluid/inference/utils/singleton.h"

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
  EngineInputConverter() {}

  virtual void operator()(const LoDTensor& in, void* out, size_t max_size) {}

  void SetStream(cudaStream_t* stream) { stream_ = stream; }

  static void Run(const std::string& in_op_type, const LoDTensor& in, void* out,
                  size_t max_size, cudaStream_t* stream) {
    PADDLE_ENFORCE(stream != nullptr);
    auto* converter = Registry<EngineInputConverter>::Lookup(
        in_op_type, "default" /* default_type */);
    PADDLE_ENFORCE_NOT_NULL(converter);
    converter->SetStream(stream);
    (*converter)(in, out, max_size);
  }

  virtual ~EngineInputConverter() {}

 protected:
  cudaStream_t* stream_{nullptr};
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

#define REGISTER_TENSORRT_INPUT_CONVERTER(in_op_type__, Converter__) \
  struct trt_input_##in_op_type__##_converter {                      \
    trt_input_##in_op_type__##_converter() {                         \
      ::paddle::inference::Registry<EngineInputConverter>::Register< \
          Converter__>(#in_op_type__);                               \
    }                                                                \
  };                                                                 \
  trt_input_##in_op_type__##_converter trt_input_##in_op_type__##_converter__;
