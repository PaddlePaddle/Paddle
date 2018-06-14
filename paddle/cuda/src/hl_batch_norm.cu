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

#include "hl_batch_norm.h"

__global__ void batchNormInference(real* output,
                                   const real* input,
                                   const real* scale,
                                   const real* bias,
                                   const real* estimatedMean,
                                   const real* estimatedVar,
                                   const double epsilon,
                                   size_t batchSize,
                                   size_t channel,
                                   size_t height,
                                   size_t width) {
  const int tid = threadIdx.x;
  const int num = channel * height * width;
  const int batch = blockIdx.x;
  for (int i = tid; i < num; i += blockDim.x) {
    const int c = i / (height * width);
    const int id = batch * num + i;
    real val = input[id] - estimatedMean[c];
    val /= sqrt(estimatedVar[c] + epsilon);
    val *= scale[c];
    val += bias[c];
    output[id] = val;
  }
}

void hl_batch_norm_cuda_inference(const real* input,
                                  real* output,
                                  const real* scale,
                                  const real* bias,
                                  const real* estimatedMean,
                                  const real* estimatedVar,
                                  const double epsilon,
                                  size_t batchSize,
                                  size_t channel,
                                  size_t height,
                                  size_t width) {
  batchNormInference<<<batchSize, 256, 0, STREAM_DEFAULT>>>(output,
                                                            input,
                                                            scale,
                                                            bias,
                                                            estimatedMean,
                                                            estimatedVar,
                                                            epsilon,
                                                            batchSize,
                                                            channel,
                                                            height,
                                                            width);

  CHECK_SYNC("hl_batch_norm_cuda_inference failed!");
}
