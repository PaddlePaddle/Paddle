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

#include "paddle/fluid/inference/tensorrt/io_converter.cc"

namespace paddle {
namespace inference {
namespace tensorrt {

class DefaultInputConverter : public EngineInputConveter {
 public:
  MulInputConverter(cudaStream_t* stream) : stream_(stream) {}
  // NOTE out is GPU memory.
  virtual void operator()(LoDTensor& in, void* out, size_t max_size) override {
    PADDLE_ENFORCE_LE(in.size(), max_size);
    auto& place = in.place();
    if (IsCPUPlace(place)) {
      PADDLE_ENFORCE(stream_ != nullptr);
      PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(out, in.data(), in.size(),
                                           cudaMemcpyHostToDevice, *stream_));
    } else if (IsCUDAPlace(place) || IsCUDAPinnedPlace(place)) {
      PADDLE_ENFORCE_EQ(0, cudaMemcpyAsync(out, in.data(), in.size(),
                                           cudaMemcpyHostToHost, *stream_));
    } else {
      PADDLE_THROW("Unknown device for converter");
    }
  }

 private:
  cudaStream_t* stream_;
};

REGISTER_TRT_INPUT_CONVERTER("mul", DefaultInputConverter);

std::unordered_map<std::string, EngineInputConveter>
    EngineInputConverter::converters_;

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
