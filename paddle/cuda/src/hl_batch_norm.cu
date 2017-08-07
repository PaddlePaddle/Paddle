/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num = channel * height * width;
  const int batch = blockIdx.y;
  for (int i = tid; i < num; i += blockDim.x) {
    const int c = (i / (height * width)) % channel;
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
  dim3 block(256, 1);
  dim3 grid(1, batchSize);
  batchNormInference<<<grid, block, 0, STREAM_DEFAULT>>>(output,
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
