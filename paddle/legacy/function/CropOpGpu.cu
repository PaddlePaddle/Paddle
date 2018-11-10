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

#include "CropOp.h"
#include "hl_base.h"

namespace paddle {

__global__ void KeCrop(real* outputs,
                       const real* inputs,
                       int inC,
                       int inH,
                       int inW,
                       int cropC,
                       int cropH,
                       int cropW,
                       int outC,
                       int outH,
                       int outW,
                       int nthreads) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nthreads) {
    const int w = idx % outW;
    const int h = (idx / outW) % outH;
    const int c = (idx / outW / outH) % outC;
    const int n = idx / outW / outH / outC;

    const int off = ((n * inC + c + cropC) * inH + h + cropH) * inW + cropW + w;
    outputs[idx] = inputs[off];
  }
}

template <>
void Crop<DEVICE_TYPE_GPU>(real* outputs,
                           const real* inputs,
                           const TensorShape inShape,
                           const TensorShape outShape,
                           const FuncConfig& conf) {
  std::vector<uint32_t> crop_corner =
      conf.get<std::vector<uint32_t>>("crop_corner");
  int cropC = crop_corner[1];
  int cropH = crop_corner[2];
  int cropW = crop_corner[3];

  int num = inShape[0];
  int inC = inShape[1];
  int inH = inShape[2];
  int inW = inShape[3];

  int outC = outShape[1];
  int outH = outShape[2];
  int outW = outShape[3];

  size_t nth = num * outC * outH * outW;
  int blockSize = 1024;
  int gridSize = (nth + blockSize - 1) / blockSize;

  KeCrop<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(outputs,
                                                     inputs,
                                                     inC,
                                                     inH,
                                                     inW,
                                                     cropC,
                                                     cropH,
                                                     cropW,
                                                     outC,
                                                     outH,
                                                     outW,
                                                     nth);
  CHECK_SYNC("Crop");
}

__global__ void KeCropDiff(const real* inGrad,
                           real* outGrad,
                           int inC,
                           int inH,
                           int inW,
                           int cropC,
                           int cropH,
                           int cropW,
                           int outC,
                           int outH,
                           int outW,
                           int nthreads) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nthreads) {
    const int w = idx % inW;
    const int h = (idx / inW) % inH;
    const int c = (idx / inW / inH) % inC;
    const int n = idx / inW / inH / inC;

    const int off =
        ((n * outC + c + cropC) * outH + h + cropH) * outW + cropW + w;

    outGrad[off] += inGrad[idx];
  }
}

template <>
void CropGrad<DEVICE_TYPE_GPU>(const real* inGrad,
                               real* outGrad,
                               const TensorShape inShape,
                               const TensorShape outShape,
                               const FuncConfig& conf) {
  std::vector<uint32_t> crop_corner =
      conf.get<std::vector<uint32_t>>("crop_corner");
  int cropC = crop_corner[1];
  int cropH = crop_corner[2];
  int cropW = crop_corner[3];

  int num = outShape[0];
  int outC = outShape[1];
  int outH = outShape[2];
  int outW = outShape[3];

  int inC = inShape[1];
  int inH = inShape[2];
  int inW = inShape[3];

  size_t nth = num * inC * inH * inW;
  int blockSize = 1024;
  int gridSize = (nth + blockSize - 1) / blockSize;

  KeCropDiff<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(inGrad,
                                                         outGrad,
                                                         inC,
                                                         inH,
                                                         inW,
                                                         cropC,
                                                         cropH,
                                                         cropW,
                                                         outC,
                                                         outH,
                                                         outW,
                                                         nth);
  CHECK_SYNC("CropGrad");
}

}  // namespace paddle
