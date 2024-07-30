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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Convert Input from Fluid to TensorRT Engine.
 * Convert Output from TensorRT Engine to Fluid.
 *
 * Note that TensorRT's ITensor follows row major, NCHW. Fluid is also row
 * major,
 * so in the default case just need to copy the data.
 */
class EngineIOConverter {
 public:
  EngineIOConverter() {}

  virtual void operator()(const phi::DenseTensor& in,
                          void* out,
                          size_t max_size) {}
  virtual void operator()(const void* in,
                          phi::DenseTensor* out,
                          size_t max_size) {}

  void SetStream(cudaStream_t* stream) { stream_ = stream; }

  static void ConvertInput(const std::string& op_type,
                           const phi::DenseTensor& in,
                           void* out,
                           size_t max_size,
                           cudaStream_t* stream) {
    PADDLE_ENFORCE_NOT_NULL(stream,
                            common::errors::InvalidArgument(
                                "The input stream must not be nullptr."));
    auto* converter = Registry<EngineIOConverter>::Global().Lookup(
        op_type, "default" /* default_type */);
    PADDLE_ENFORCE_NOT_NULL(
        converter,
        common::errors::Unimplemented("The %s in is not supported yet.",
                                      op_type.c_str()));
    converter->SetStream(stream);
    (*converter)(in, out, max_size);
  }

  static void ConvertOutput(const std::string& op_type,
                            const void* in,
                            phi::DenseTensor* out,
                            size_t max_size,
                            cudaStream_t* stream) {
    PADDLE_ENFORCE_NOT_NULL(stream,
                            common::errors::InvalidArgument(
                                "The input stream must not be nullptr."));
    auto* converter = Registry<EngineIOConverter>::Global().Lookup(
        op_type, "default" /* default_type */);
    PADDLE_ENFORCE_NOT_NULL(
        converter,
        common::errors::Unimplemented("The %s in not supported yet.",
                                      op_type.c_str()));
    converter->SetStream(stream);
    (*converter)(in, out, max_size);
  }

  virtual ~EngineIOConverter() {}

 protected:
  cudaStream_t* stream_{nullptr};
};

#define REGISTER_TENSORRT_IO_CONVERTER(op_type__, Converter__)                 \
  struct trt_io_##op_type__##_converter {                                      \
    trt_io_##op_type__##_converter() {                                         \
      Registry<EngineIOConverter>::Global().Register<Converter__>(#op_type__); \
    }                                                                          \
  };                                                                           \
  trt_io_##op_type__##_converter trt_io_##op_type__##_converter__;

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
