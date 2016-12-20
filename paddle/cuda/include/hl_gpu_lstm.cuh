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


#ifndef HL_GPU_LSTM_CUH_
#define HL_GPU_LSTM_CUH_

#ifdef __NVCC__

#include "paddle/utils/Logging.h"
#include "hl_device_functions.cuh"

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template<class Op, bool isBatch>
__global__ void KeLstmForward(Op op,
                              hl_lstm_value value,
                              int frameSize,
                              int batchSize,
                              hl_activation_mode_t active_node,
                              hl_activation_mode_t active_gate,
                              hl_activation_mode_t active_state) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;

  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    value.gateValue += batchIdx * frameSize * 4;
    value.outputValue += batchIdx * frameSize;
    value.stateValue  += batchIdx * frameSize;
    value.stateActiveValue += batchIdx * frameSize;
  }

  real rState;
  real rPrevState = 0;
  real rStateAtv;
  real rOut;
  real rValueIn;
  real rValueIg;
  real rValueFg;
  real rValueOg;
  real rCheckI = value.checkIg[frameIdx];
  real rCheckF = value.checkFg[frameIdx];
  real rCheckO = value.checkOg[frameIdx];

  rValueIn = value.gateValue[frameIdx];
  rValueIg = value.gateValue[frameIdx + frameSize];
  rValueFg = value.gateValue[frameIdx + frameSize * 2];
  rValueOg = value.gateValue[frameIdx + frameSize * 3];

  if (value.prevStateValue) {
    if (isBatch) value.prevStateValue += batchIdx * frameSize;
    rPrevState = value.prevStateValue[frameIdx];
  }

  op(rValueIn,
     rValueIg,
     rValueFg,
     rValueOg,
     rPrevState,
     rState,
     rStateAtv,
     rOut,
     rCheckI,
     rCheckF,
     rCheckO,
     hppl::gpu::forward[active_node],
     hppl::gpu::forward[active_gate],
     hppl::gpu::forward[active_state]);

  value.gateValue[frameIdx] = rValueIn;
  value.gateValue[frameIdx + frameSize] = rValueIg;
  value.gateValue[frameIdx + frameSize * 2] = rValueFg;
  value.gateValue[frameIdx + frameSize * 3] = rValueOg;

  value.stateValue[frameIdx] = rState;
  value.stateActiveValue[frameIdx] = rStateAtv;
  value.outputValue[frameIdx] = rOut;
}

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template<class Op, bool isBatch>
__global__ void KeLstmBackward(Op op,
                               hl_lstm_value value,
                               hl_lstm_grad grad,
                               int frameSize,
                               int batchSize,
                               hl_activation_mode_t active_node,
                               hl_activation_mode_t active_gate,
                               hl_activation_mode_t active_state) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;

  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    value.gateValue += batchIdx * frameSize * 4;
    value.stateValue += batchIdx * frameSize;
    value.stateActiveValue += batchIdx * frameSize;
    grad.gateGrad += batchIdx * frameSize * 4;
    grad.stateGrad += batchIdx * frameSize;
    grad.outputGrad += batchIdx * frameSize;
  }

  real rValueIn;
  real rValueIg;
  real rValueFg;
  real rValueOg;
  real rGradIn;
  real rGradIg;
  real rGradFg;
  real rGradOg;
  real rPrevState = 0;
  real rPrevStateGrad;
  real rState;
  real rStateGrad;
  real rStateAtv;
  real rOutputGrad;
  real rCheckI = value.checkIg[frameIdx];
  real rCheckF = value.checkFg[frameIdx];
  real rCheckO = value.checkOg[frameIdx];
  real rCheckIGrad;
  real rCheckFGrad;
  real rCheckOGrad;

  rValueIn = value.gateValue[frameIdx];
  rValueIg = value.gateValue[frameIdx + frameSize];
  rValueFg = value.gateValue[frameIdx + frameSize * 2];
  rValueOg = value.gateValue[frameIdx + frameSize * 3];
  rState = value.stateValue[frameIdx];
  rStateAtv = value.stateActiveValue[frameIdx];
  rOutputGrad = grad.outputGrad[frameIdx];
  rStateGrad = grad.stateGrad[frameIdx];

  if (value.prevStateValue) {
    if (isBatch) value.prevStateValue += batchIdx * frameSize;
    rPrevState = value.prevStateValue[frameIdx];
  }

  op(rValueIn,
     rValueIg,
     rValueFg,
     rValueOg,
     rGradIn,
     rGradIg,
     rGradFg,
     rGradOg,
     rPrevState,
     rPrevStateGrad,
     rState,
     rStateGrad,
     rStateAtv,
     rOutputGrad,
     rCheckI,
     rCheckF,
     rCheckO,
     rCheckIGrad,
     rCheckFGrad,
     rCheckOGrad,
     hppl::gpu::backward[active_node],
     hppl::gpu::backward[active_gate],
     hppl::gpu::backward[active_state]);

  grad.gateGrad[frameIdx] = rGradIn;
  grad.gateGrad[frameIdx + frameSize    ] = rGradIg;
  grad.gateGrad[frameIdx + frameSize * 2] = rGradFg;
  grad.gateGrad[frameIdx + frameSize * 3] = rGradOg;
  grad.stateGrad[frameIdx] = rStateGrad;
  if (grad.prevStateGrad) {
    if (isBatch) grad.prevStateGrad += batchIdx * frameSize;
    grad.prevStateGrad[frameIdx] = rPrevStateGrad;
  }

  if (isBatch) {
    if (value.prevStateValue) {
      if (grad.checkIgGrad) paddle::paddleAtomicAdd(grad.checkIgGrad+frameIdx, rCheckIGrad);
      if (grad.checkFgGrad) paddle::paddleAtomicAdd(grad.checkFgGrad+frameIdx, rCheckFGrad);
    }
    if (grad.checkOgGrad) paddle::paddleAtomicAdd(grad.checkOgGrad+frameIdx, rCheckOGrad);
  } else {
    if (value.prevStateValue) {
      if (grad.checkIgGrad) grad.checkIgGrad[frameIdx] += rCheckIGrad;
      if (grad.checkFgGrad) grad.checkFgGrad[frameIdx] += rCheckFGrad;
    }
    if (grad.checkOgGrad) grad.checkOgGrad[frameIdx] += rCheckOGrad;
  }
}

template<class Op>
void hl_gpu_lstm_forward(Op op,
                         hl_lstm_value value,
                         int frameSize,
                         int batchSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate,
                         hl_activation_mode_t active_state) {
  dim3 threads;
  dim3 grid;
  if (batchSize == 1) {
    int framePerBlock = frameSize <= 1024 ? frameSize : 1024;
    int frameBlocks = (frameSize + 1024 - 1) / 1024;
    threads = dim3(framePerBlock, 1);
    grid = dim3(frameBlocks, 1);
  } else {
    /* framePerBlock = 32 batchPerBlock = 32 */
    threads = dim3(32, 32);
    grid = dim3((frameSize + 32 - 1) / 32, (batchSize + 32 - 1) / 32);
  }

  if (batchSize == 1) {
    KeLstmForward<Op, /* isBatch= */false>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(op, value,
      frameSize, batchSize, active_node, active_gate, active_state);
  } else {
    KeLstmForward<Op, /* isBatch= */true>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(op, value,
      frameSize, batchSize, active_node, active_gate, active_state);
  }

  CHECK_SYNC("hl_gpu_lstm_forward failed");
}

template<class Op>
void hl_gpu_lstm_backward(Op op,
                          hl_lstm_value value,
                          hl_lstm_grad grad,
                          int frameSize,
                          int batchSize,
                          hl_activation_mode_t active_node,
                          hl_activation_mode_t active_gate,
                          hl_activation_mode_t active_state) {
  dim3 threads;
  dim3 grid;
  if (batchSize == 1) {
    int framePerBlock = frameSize <= 1024 ? frameSize : 1024;
    int frameBlocks = (frameSize + 1024 - 1) / 1024;
    threads = dim3(framePerBlock, 1);
    grid = dim3(frameBlocks, 1);
  } else {
    /* framePerBlock = 32 batchPerBlock = 32 */
    threads = dim3(32, 32);
    grid = dim3((frameSize + 32 - 1) / 32, (batchSize + 32 - 1) / 32);
  }

  if (batchSize == 1) {
    KeLstmBackward<Op, /* isBatch= */false>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(op, value, grad,
      frameSize, batchSize, active_node, active_gate, active_state);
  } else {
    KeLstmBackward<Op, /* isBatch= */true>
      <<<grid, threads, 0, STREAM_DEFAULT>>>(op, value, grad,
      frameSize, batchSize, active_node, active_gate, active_state);
  }

  CHECK_SYNC("hl_gpu_lstm_backward failed");
}

#else

template<class Op>
void hl_gpu_lstm_forward(Op op,
                         hl_lstm_value value,
                         int frameSize,
                         int batchSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate,
                         hl_activation_mode_t active_state) {}

template<class Op>
void hl_gpu_lstm_backward(Op op,
                          hl_lstm_value value,
                          hl_lstm_grad grad,
                          int frameSize,
                          int batchSize,
                          hl_activation_mode_t active_node,
                          hl_activation_mode_t active_gate,
                          hl_activation_mode_t active_state) {}

#endif

#endif /* HL_GPU_LSTM_CUH_ */
