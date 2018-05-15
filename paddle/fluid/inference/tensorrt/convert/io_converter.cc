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

#include "paddle/fluid/inference/tensorrt/convert/io_converter.h"
#include <cuda.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

using platform::is_gpu_place;
using platform::is_cpu_place;

class DefaultIOConverter : public EngineIOConverter {
 public:
  DefaultIOConverter() {}
  // NOTE out is GPU memory.
  virtual void operator()(const LoDTensor& in, void* out,
                          size_t max_size) override {
    PADDLE_ENFORCE(out != nullptr);
    PADDLE_ENFORCE(stream_ != nullptr);
    const auto& place = in.place();
    size_t size = in.memory_size();
    PADDLE_ENFORCE_LE(size, max_size);
    if (is_cpu_place(place)) {
      PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(out, in.data<float>(), size,
                                           cudaMemcpyHostToDevice, *stream_));
    } else if (is_gpu_place(place)) {
      PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(out, in.data<float>(), size,
                                           cudaMemcpyDeviceToDevice, *stream_));
    } else {
      PADDLE_THROW("Unknown device for converter");
    }
    cudaStreamSynchronize(*stream_);
  }
  // NOTE in is GPU memory.
  virtual void operator()(const void* in, LoDTensor* out,
                          size_t max_size) override {
    PADDLE_ENFORCE(in != nullptr);
    PADDLE_ENFORCE(stream_ != nullptr);
    const auto& place = out->place();
    size_t size = out->memory_size();
    PADDLE_ENFORCE_LE(size, max_size);
    if (is_cpu_place(place)) {
      PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(out->data<float>(), in, size,
                                           cudaMemcpyDeviceToHost, *stream_));
    } else if (is_gpu_place(place)) {
      PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(out->data<float>(), in, size,
                                           cudaMemcpyDeviceToDevice, *stream_));
    } else {
      PADDLE_THROW("Unknown device for converter");
    }
    cudaStreamSynchronize(*stream_);
  }
};

// fluid LodTensor <-> tensorrt ITensor
REGISTER_TENSORRT_IO_CONVERTER(default, DefaultIOConverter);

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
