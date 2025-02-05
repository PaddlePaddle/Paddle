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

#include <gtest/gtest.h>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/tensorrt/convert/io_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

void IOConverterTester(const phi::DeviceContext& ctx) {
  cudaStream_t stream;
  ASSERT_EQ(0, cudaStreamCreate(&stream));

  // init fluid in_tensor
  phi::DenseTensor in_tensor;
  in_tensor.Resize({10, 10});
  auto place = ctx.GetPlace();
  in_tensor.mutable_data<float>(place);
  std::vector<float> init;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init.push_back(i);
  }
  framework::TensorFromVector(init, ctx, &in_tensor);

  // init tensorrt buffer
  void* buffer;
  size_t size = in_tensor.memory_size();
  ASSERT_EQ(cudaMalloc(&buffer, size), 0);

  // convert fluid in_tensor to tensorrt buffer
  EngineIOConverter::ConvertInput("test", in_tensor, buffer, size, &stream);

  // convert tensorrt buffer to fluid out_tensor
  phi::DenseTensor out_tensor;
  out_tensor.Resize({10, 10});
  out_tensor.mutable_data<float>(place);
  EngineIOConverter::ConvertOutput("test", buffer, &out_tensor, size, &stream);

  // compare in_tensor and out_tensor
  std::vector<float> result;
  framework::TensorToVector(out_tensor, ctx, &result);
  EXPECT_EQ(init.size(), result.size());
  for (size_t i = 0; i < init.size(); i++) {
    EXPECT_EQ(init[i], result[i]);
  }
  cudaStreamDestroy(stream);
}

TEST(EngineIOConverterTester, DefaultCPU) {
  phi::CPUPlace place;
  phi::CPUContext ctx(place);
  IOConverterTester(ctx);
}

TEST(EngineIOConverterTester, DefaultGPU) {
  phi::GPUPlace place;
  phi::GPUContext ctx(place);
  IOConverterTester(ctx);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
