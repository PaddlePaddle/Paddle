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

#include "paddle/fluid/inference/tensorrt/io_converter.h"
#include <cuda.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

using platform::is_gpu_place;
using platform::is_cpu_place;

class DefaultInputConverter : public EngineInputConverter {
 public:
  DefaultInputConverter() {}
  // NOTE out is GPU memory.
  virtual void operator()(const LoDTensor& in, void* out,
                          size_t max_size) override {
    PADDLE_ENFORCE(out != nullptr);
    PADDLE_ENFORCE_LE(in.memory_size(), max_size);
    const auto& place = in.place();
    if (is_cpu_place(place)) {
      PADDLE_ENFORCE(stream_ != nullptr);
      PADDLE_ENFORCE_EQ(0,
                        cudaMemcpyAsync(out, in.data<float>(), in.memory_size(),
                                        cudaMemcpyHostToDevice, *stream_));

    } else if (is_gpu_place(place)) {
      PADDLE_ENFORCE_EQ(0,
                        cudaMemcpyAsync(out, in.data<float>(), in.memory_size(),
                                        cudaMemcpyHostToHost, *stream_));

    } else {
      PADDLE_THROW("Unknown device for converter");
    }
    cudaStreamSynchronize(*stream_);
  }
};

REGISTER_TENSORRT_INPUT_CONVERTER(mul, DefaultInputConverter);

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
