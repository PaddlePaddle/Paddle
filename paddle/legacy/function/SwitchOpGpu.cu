/* Copyright (c) 2016 Paddle

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "SwitchOp.h"
#include "hl_base.h"

namespace paddle {

__global__ void KeNCHW2NHWC(real* outputs,
                            const real* inputs,
                            int inC,
                            int inH,
                            int inW,
                            int nthreads,
                            int argType) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nthreads) {
    const int w = idx % inW;
    const int h = (idx / inW) % inH;
    const int c = (idx / inW / inH) % inC;
    const int n = idx / inW / inH / inC;

    const int off = ((n * inH + h) * inW + w) * inC + c;
    if (argType == ADD_TO) {
      outputs[off] += inputs[idx];
    } else {
      outputs[off] = inputs[idx];
    }
  }
}

template <>
void NCHW2NHWC<DEVICE_TYPE_GPU>(real* outputs,
                                const real* inputs,
                                const int num,
                                const int inC,
                                const int inH,
                                const int inW,
                                const int argType) {
  size_t nth = num * inC * inH * inW;
  int blockSize = 1024;
  int gridSize = (nth + 1024 - 1) / 1024;
  KeNCHW2NHWC<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
      outputs, inputs, inC, inH, inW, nth, argType);
  CHECK_SYNC("NCHW2NHWC");
}

__global__ void KeNHWC2NCHW(real* outputs,
                            const real* inputs,
                            int inH,
                            int inW,
                            int inC,
                            int nthreads,
                            int argType) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nthreads) {
    const int c = idx % inC;
    const int w = (idx / inC) % inW;
    const int h = (idx / inC / inW) % inH;
    const int n = idx / inW / inH / inC;

    const int off = ((n * inC + c) * inH + h) * inW + w;
    if (argType == ADD_TO) {
      outputs[off] += inputs[idx];
    } else {
      outputs[off] = inputs[idx];
    }
  }
}

template <>
void NHWC2NCHW<DEVICE_TYPE_GPU>(real* outputs,
                                const real* inputs,
                                const int num,
                                const int inH,
                                const int inW,
                                const int inC,
                                const int argType) {
  int nth = num * inC * inH * inW;
  int blockSize = 1024;
  int gridSize = (nth + 1024 - 1) / 1024;
  KeNHWC2NCHW<<<gridSize, blockSize, 0, STREAM_DEFAULT>>>(
      outputs, inputs, inH, inW, inC, nth, argType);
  CHECK_SYNC("NHWC2NCHW");
}

}  // namespace paddle
