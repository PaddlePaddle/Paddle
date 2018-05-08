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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/tensorrt/convert/io_converter.h"

#include <gtest/gtest.h>

namespace paddle {
namespace inference {
namespace tensorrt {

class EngineInputConverterTester : public ::testing::Test {
 public:
  void SetUp() override { tensor.Resize({10, 10}); }

  framework::LoDTensor tensor;
};

TEST_F(EngineInputConverterTester, DefaultCPU) {
  void* buffer;
  tensor.mutable_data<float>(platform::CPUPlace());
  ASSERT_EQ(cudaMalloc(&buffer, tensor.memory_size()), 0);

  cudaStream_t stream;
  EngineInputConverter::Run("test", tensor, buffer, tensor.memory_size(),
                            &stream);
}

TEST_F(EngineInputConverterTester, DefaultGPU) {
  void* buffer;
  tensor.mutable_data<float>(platform::CUDAPlace());
  ASSERT_EQ(cudaMalloc(&buffer, tensor.memory_size()), 0);

  cudaStream_t stream;
  EngineInputConverter::Run("test", tensor, buffer, tensor.memory_size(),
                            &stream);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
