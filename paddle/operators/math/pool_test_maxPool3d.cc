/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include <gtest/gtest.h>
#include "paddle/operators/math/pooling.h"

#include "paddle/memory/memcpy.h"
#include "paddle/platform/enforce.h"

#include <stdlib.h>
#include <time.h>

#ifndef PADDLE_ONLY_CPU

template <typename PooType>
void testPool3d(paddle::platform::DeviceContext& context, PooType pool_process,
                paddle::framework::Tensor& input,
                paddle::framework::Tensor& input_grad,
                paddle::framework::Tensor& output,
                paddle::framework::Tensor& output_grad, std::vector<int>& ksize,
                std::vector<int>& strides, std::vector<int>& paddings) {
  paddle::operators::math::Pool3dForwardFunctor<paddle::platform::GPUPlace,
                                                PooType, float>
      pool3d_forward;
  pool3d_forward(context, input, output, ksize, strides, paddings,
                 pool_process);

  int times = 50;
  clock_t start, finish;
  double totaltime;

  // Pool3dBackwardFunctor
  start = clock();
  for (int i = 0; i < times; ++i) {
    paddle::operators::math::Pool3dBackwardFunctor<paddle::platform::GPUPlace,
                                                   PooType, float>
        pool3d_backward;
    pool3d_backward(context, input, input_grad, output, output_grad, ksize,
                    strides, paddings, pool_process);
    PADDLE_ENFORCE(cudaStreamSynchronize(0),
                   "cudaStreamSynchronize failed in pool3d_backward CopyFrom");
  }
  finish = clock();
  totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
  totaltime /= times;
  std::cout << "\nPool3dBackwardFunctor: " << totaltime << "s" << std::endl;

  // MaxPool3dBackwardFunctor
  start = clock();
  for (int j = 0; j < times; ++j) {
    paddle::operators::math::MaxPool3dBackwardFunctor<
        paddle::platform::GPUPlace, float>
        maxpool3d_backward;
    maxpool3d_backward(context, input, input_grad, output, output_grad, ksize,
                       strides, paddings);
    PADDLE_ENFORCE(
        cudaStreamSynchronize(0),
        "cudaStreamSynchronize failed in maxpool3d_backward CopyFrom");
  }
  finish = clock();
  totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
  totaltime /= times;
  std::cout << "\nMaxPool3dBackwardFunctor: " << totaltime << "s" << std::endl;
}

void test3dPool() {
  using paddle::platform::DeviceContext;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::GPUPlace;

  paddle::framework::Tensor input_tmp;
  paddle::framework::Tensor output_tmp;
  paddle::framework::Tensor input;
  paddle::framework::Tensor input_grad;
  paddle::framework::Tensor output;
  paddle::framework::Tensor output_grad;

  int batch = 32;
  int channel = 4;
  int input_depth = 4;
  int input_height = 128;
  int input_width = 128;
  int in_len = batch * channel * input_depth * input_height * input_width;
  std::vector<int> ksize({3, 3, 3});
  std::vector<int> strides({2, 2, 2});
  std::vector<int> paddings({1, 1, 1});

  int output_depth =
      (input_depth - ksize[0] + 2 * paddings[0]) / strides[0] + 1;
  int output_height =
      (input_height - ksize[1] + 2 * paddings[1]) / strides[1] + 1;
  int output_width =
      (input_width - ksize[2] + 2 * paddings[2]) / strides[2] + 1;

  int output_len = output_depth * output_height * output_width;

  input_tmp.mutable_data<float>(
      {batch, channel, input_depth, input_height, input_width},
      paddle::platform::CPUPlace());
  output_tmp.mutable_data<float>(
      {batch, channel, output_depth, output_height, output_width},
      paddle::platform::CPUPlace());

  float* arr = new float[in_len];
  auto* place = new paddle::platform::GPUPlace();

  // input
  float* input_ptr = input_tmp.data<float>();
  for (int i = 0; i < in_len; ++i) arr[i] = i;  // rand() / double(RAND_MAX/2);
  memcpy(input_ptr, arr, in_len * sizeof(float));
  input.CopyFrom<float>(input_tmp, *place);

  // input_grad
  input_ptr = input_tmp.data<float>();
  for (int i = 0; i < in_len; ++i) arr[i] = 0;
  memcpy(input_ptr, arr, in_len * sizeof(float));
  input_grad.CopyFrom<float>(input_tmp, *place);

  // output
  input_ptr = output_tmp.data<float>();
  for (int i = 0; i < output_len; ++i)
    arr[i] = 0;  // rand() / double(RAND_MAX/2);
  memcpy(input_ptr, arr, output_len * sizeof(float));
  output.CopyFrom<float>(input_tmp, *place);

  // output_grad
  input_ptr = output_tmp.data<float>();
  for (int i = 0; i < output_len; ++i)
    arr[i] = 1;  // rand() / double(RAND_MAX/2);
  memcpy(input_ptr, arr, output_len * sizeof(float));
  output_grad.CopyFrom<float>(input_tmp, *place);

  paddle::platform::DeviceContext* context =
      new paddle::platform::CUDADeviceContext(paddle::platform::GPUPlace());
  paddle::operators::math::pool::maxPool<float> pool_process;

  testPool3d<paddle::operators::math::pool::maxPool<float>>(
      *context, pool_process, input, input_grad, output, output_grad, ksize,
      strides, paddings);
}

int main() { test3dPool(); }
#endif