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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void CopyForTensorKernel(FeatureItem* src, void** dest,
                                    size_t* offset, char* type,
                                    size_t total_size, size_t row_size,
                                    size_t col_size) {
  CUDA_KERNEL_LOOP(i, row_size * col_size) {
    int row_id = i / col_size;
    int col_id = i % col_size;
    size_t left, right;
    if (row_id == 0) {
      left = offset[row_id * (col_size + 1) + col_id];
      right = offset[row_id * (col_size + 1) + col_id + 1];
    } else {
      left = offset[row_id * (col_size + 1) + col_id] -
             offset[(row_id - 1) * (col_size + 1) + col_id];
      right = offset[row_id * (col_size + 1) + col_id + 1] -
              offset[(row_id - 1) * (col_size + 1) + col_id + 1];
    }
    uint64_t* up = NULL;
    float* fp = NULL;
    if (type[row_id] == 'f') {
      fp = reinterpret_cast<float*>(dest[row_id]);
    } else {
      up = reinterpret_cast<uint64_t*>(
          *(reinterpret_cast<uint64_t**>(dest) + row_id));
    }
    size_t begin = offset[row_id * (col_size + 1) + col_id + 1] +
                   offset[(row_size - 1) * (col_size + 1) + col_id] -
                   offset[row_id * (col_size + 1) + col_id] - (right - left);
    PADDLE_ENFORCE(begin >= 0, "begin must be ge 0.");
    PADDLE_ENFORCE(begin < total_size, "begin must be lt total_size.");
    for (size_t k = left; k < right; ++k) {
      PADDLE_ENFORCE((begin + k - left) >= 0 && (begin + k - left) < total_size,
                     "begin+k-left must be in [0, total_size)");
      if (type[row_id] == 'f') {
        *(fp + k) = src[begin + k - left].sign().float_feasign_;
      } else {
        *(up + k) = src[begin + k - left].sign().uint64_feasign_;
      }
    }
  }
}

void MultiSlotInMemoryDataFeed::CopyForTensor(
    const paddle::platform::Place& place, FeatureItem* src, void** dest,
    size_t* offset, char* type, size_t total_size, size_t row_size,
    size_t col_size) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        boost::get<platform::CUDAPlace>(place)))
                    ->stream();
  CopyForTensorKernel<<<((row_size * (col_size - 1)) + 511) / 512, 512, 0,
                        stream>>>(src, dest, offset, type, total_size, row_size,
                                  col_size - 1);
  cudaStreamSynchronize(stream);
}

}  // namespace framework
}  // namespace paddle
