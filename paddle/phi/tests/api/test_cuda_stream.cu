/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/core/cuda_stream.h"

TEST(CUDAStream, GPU) {
  phi::GPUPlace gpu0(0);
  phi::CUDAStream* stream = paddle::getCurrentCUDAStream(gpu0);
  EXPECT_TRUE(stream != nullptr);
  gpuStream_t raw_stream = stream->raw_stream();
  EXPECT_TRUE(raw_stream != nullptr);
}
