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

TEST(GemmConv, NaiveConv) {
  Convolution<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU>(
      "NaiveConv-CPU", "GemmConv-CPU", forward);
  Convolution2<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU>(
      "NaiveConv-CPU", "GemmConv-CPU", forward);
}

#ifdef PADDLE_WITH_CUDA
TEST(GemmConv, Forward) {
  Convolution<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConv-CPU", "GemmConv-GPU", forward);
  Convolution2<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConv-CPU", "GemmConv-GPU", forward);
}

TEST(GemmConv, BackwardInput) {
  Convolution<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConvGradInput-CPU", "GemmConvGradInput-GPU", backward_input);
  Convolution2<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConvGradInput-CPU", "GemmConvGradInput-GPU", backward_input);
}

TEST(GemmConv, BackwardFilter) {
  Convolution<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConvGradFilter-CPU", "GemmConvGradFilter-GPU", backward_filter);
  Convolution2<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConvGradFilter-CPU", "GemmConvGradFilter-GPU", backward_filter);
}
#endif

}  // namespace paddle
