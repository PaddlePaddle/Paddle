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

#include "hl_base.h"
#include "CosSimOp.h"

namespace paddle {

template<int block_size>
__global__ void KeCosSim(real* output,
                         const real* input1,
                         const real* input2,
                         int width,
                         int input1_height,
                         int input2_height,
                         real scale) {
  const int ty = blockIdx.y;
  int tid = threadIdx.x;

  __shared__ real xx[block_size];
  __shared__ real yy[block_size];
  __shared__ real xy[block_size];

  xx[tid] = 0.0;
  yy[tid] = 0.0;
  xy[tid] = 0.0;
  __syncthreads();

  input1 += ty * width;
  if (input2_height > 1) {
    input2 += ty * width;
  }
  for (int index = tid; index < width; index += block_size) {
    real x = input1[index];
    real y = input2[index];
    xx[tid] += x * x;
    yy[tid] += y * y;
    xy[tid] += x * y;
  }
  __syncthreads();

  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      xx[tid] += xx[tid + s];
      yy[tid] += yy[tid + s];
      xy[tid] += xy[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[ty] = scale * xy[0] / (sqrt(xx[0]) * sqrt(yy[0]));
  }
}

void hlCossim(real* output,
               const real* input1,
               const real* input2,
               size_t width,
               size_t input1_height,
               size_t input2_height,
               real scale) {
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(input1);
  CHECK_NOTNULL(input2);
  const int block_size = 256;
  dim3 threads(block_size, 1);
  dim3 grid(1, input1_height);

  KeCosSim<block_size><<<grid, threads, 0, STREAM_DEFAULT>>>
    (output, input1, input2, width, input1_height, input2_height, scale);
  CHECK_SYNC("hl_cossim failed");
}

template <>
void CosSimForward<DEVICE_TYPE_GPU>(GpuMatrix* out_mat,
                                    const GpuMatrix* in1_mat,
                                    const GpuMatrix* in2_mat,
                                    real scale) {
  CHECK(out_mat && in1_mat && in2_mat);
  CHECK(in1_mat->useGpu_ == true && in2_mat->useGpu_ == true)
      << "Matrix type are not GPU";

  size_t numSamples = out_mat->getHeight();
  size_t dim = in1_mat->getWidth();
  real* out = out_mat->getData();
  const real* x = in1_mat->getData();
  const real* y = in2_mat->getData();
  hlCossim(out, x, y, dim, in1_mat->getHeight(), in2_mat->getHeight(), scale);
}

}  // namespace paddle
