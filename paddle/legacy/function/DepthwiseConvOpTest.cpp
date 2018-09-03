/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "ConvOpTest.h"

namespace paddle {

#ifdef PADDLE_WITH_CUDA
TEST(DepthwiseConv, Forward) {
  DepthwiseConvolution<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConv-CPU", "DepthwiseConv-GPU", forward);
}

TEST(DepthwiseConv, BackwardInput) {
  DepthwiseConvolution<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConvGradInput-CPU", "DepthwiseConvGradInput-GPU", backward_input);
}

TEST(DepthwiseConv, BackwardFilter) {
  DepthwiseConvolution<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConvGradFilter-CPU", "DepthwiseConvGradFilter-GPU", backward_filter);
}
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

TEST(DepthwiseConv, Forward) {
  DepthwiseConvolution<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU>(
      "GemmConv-CPU", "NeonDepthwiseConv-CPU", forward);
}

#endif

}  // namespace paddle
