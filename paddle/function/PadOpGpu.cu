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

#include "PadOp.h"
#include "hl_base.h"

namespace paddle {

__global__ void KePad(real* outputs,
                      const real* inputs,
                      int inC,
                      int inH,
                      int inW,
                      int padc,
                      int padh,
                      int padw,
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

    const int off = ((n * outC + c + padc) * outH + h + padh) * outW + padw + w;
    outputs[off] = inputs[idx];
  }
}

template <>
void Pad<DEVICE_TYPE_GPU>(real* outputs,
                          const real* inputs,
                          const int num,
                          const int inC,
                          const int inH,
                          const int inW,
                          const PadConf& pad) {
  size_t nth = num * inC * inH * inW;
  int blockSize = 1024;
  int gridSize = (nth + 1024 - 1) / 1024;
  int cstart = pad.channel[0], cend = pad.channel[1];
  int hstart = pad.height[0], hend = pad.height[1];
  int wstart = pad.width[0], wend = pad.width[1];
  int outC = inC + cstart + cend;
  int outH = inH + hstart + hend;
  int outW = inW + wstart + wend;
  KePad<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(outputs,
                                                    inputs,
                                                    inC,
                                                    inH,
                                                    inW,
                                                    cstart,
                                                    hstart,
                                                    wstart,
                                                    outC,
                                                    outH,
                                                    outW,
                                                    nth);
  CHECK_SYNC("Pad");
}

__global__ void KePadDiff(real* inGrad,
                          const real* outGrad,
                          int inC,
                          int inH,
                          int inW,
                          int padc,
                          int padh,
                          int padw,
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

    const int off = ((n * outC + c + padc) * outH + h + padh) * outW + padw + w;
    inGrad[idx] += outGrad[off];
  }
}

template <>
void PadGrad<DEVICE_TYPE_GPU>(real* inGrad,
                              const real* outGrad,
                              const int num,
                              const int inC,
                              const int inH,
                              const int inW,
                              const PadConf& pad) {
  int nth = num * inC * inH * inW;
  int blockSize = 1024;
  int gridSize = (nth + 1024 - 1) / 1024;
  int cstart = pad.channel[0], cend = pad.channel[1];
  int hstart = pad.height[0], hend = pad.height[1];
  int wstart = pad.width[0], wend = pad.width[1];
  int outC = inC + cstart + cend;
  int outH = inH + hstart + hend;
  int outW = inW + wstart + wend;
  KePadDiff<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(inGrad,
                                                        outGrad,
                                                        inC,
                                                        inH,
                                                        inW,
                                                        cstart,
                                                        hstart,
                                                        wstart,
                                                        outC,
                                                        outH,
                                                        outW,
                                                        nth);
  CHECK_SYNC("PadGrad");
}

}  // namespace paddle
