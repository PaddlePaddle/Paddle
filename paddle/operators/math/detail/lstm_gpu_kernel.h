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

#pragma once
#include "paddle/operators/math/detail/activation_functions.h"
#include "paddle/operators/math/lstm_compute.h"
#include "paddle/platform/cuda_helper.h"
#include "paddle/platform/device_context.h"

#include <type_traits>

namespace paddle {
namespace operators {
namespace math {
namespace detail {

/*
 * threads(framePerBlock, batchPerBlock)
 * grid(frameBlocks, batchBlocks)
 */
template <class T, class Op, bool isBatch>
__global__ void KeLstmForward(Op op, LstmMetaValue<T> value, int frameSize,
                              int batchSize, activation_mode_t active_node,
                              activation_mode_t active_gate,
                              activation_mode_t active_state) {
  const int frameIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frameIdx >= frameSize) return;

  int batchIdx = 0;
  if (isBatch) {
    batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize) return;
    value.gateValue += batchIdx * frameSize * 4;
    value.outputValue += batchIdx * frameSize;
    value.stateValue += batchIdx * frameSize;
    value.stateActiveValue += batchIdx * frameSize;
  }

  T rState;
  T rPrevState = 0;
  T rStateAtv;
  T rOut;
  T rValueIn;
  T rValueIg;
  T rValueFg;
  T rValueOg;

  T rCheckI = value.checkIg ? value.checkIg[frameIdx] : 0;
  T rCheckF = value.checkFg ? value.checkFg[frameIdx] : 0;
  T rCheckO = value.checkOg ? value.checkOg[frameIdx] : 0;

  rValueIn = value.gateValue[frameIdx];
  rValueIg = value.gateValue[frameIdx + frameSize];
  rValueFg = value.gateValue[frameIdx + frameSize * 2];
  rValueOg = value.gateValue[frameIdx + frameSize * 3];

  if (value.prevStateValue) {
    if (isBatch) value.prevStateValue += batchIdx * frameSize;
    rPrevState = value.prevStateValue[frameIdx];
  }

  op(rValueIn, rValueIg, rValueFg, rValueOg, rPrevState, rState, rStateAtv,
     rOut, rCheckI, rCheckF, rCheckO, active_node, active_gate, active_state);

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
template <class T, class Op, bool isBatch>
__global__ void KeLstmBackward(Op op, LstmMetaValue<T> value,
                               LstmMetaGrad<T> grad, int frameSize,
                               int batchSize, activation_mode_t active_node,
                               activation_mode_t active_gate,
                               activation_mode_t active_state) {
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

  T rValueIn;
  T rValueIg;
  T rValueFg;
  T rValueOg;
  T rGradIn;
  T rGradIg;
  T rGradFg;
  T rGradOg;
  T rPrevState = 0;
  T rPrevStateGrad;
  T rState;
  T rStateGrad;
  T rStateAtv;
  T rOutputGrad;
  T rCheckI = value.checkIg ? value.checkIg[frameIdx] : 0;
  T rCheckF = value.checkFg ? value.checkFg[frameIdx] : 0;
  T rCheckO = value.checkOg ? value.checkOg[frameIdx] : 0;

  T rCheckIGrad;
  T rCheckFGrad;
  T rCheckOGrad;

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

  op(rValueIn, rValueIg, rValueFg, rValueOg, rGradIn, rGradIg, rGradFg, rGradOg,
     rPrevState, rPrevStateGrad, rState, rStateGrad, rStateAtv, rOutputGrad,
     rCheckI, rCheckF, rCheckO, rCheckIGrad, rCheckFGrad, rCheckOGrad,
     active_node, active_gate, active_state);

  grad.gateGrad[frameIdx] = rGradIn;
  grad.gateGrad[frameIdx + frameSize] = rGradIg;
  grad.gateGrad[frameIdx + frameSize * 2] = rGradFg;
  grad.gateGrad[frameIdx + frameSize * 3] = rGradOg;
  grad.stateGrad[frameIdx] = rStateGrad;
  if (grad.prevStateGrad) {
    if (isBatch) grad.prevStateGrad += batchIdx * frameSize;
    grad.prevStateGrad[frameIdx] = rPrevStateGrad;
  }

  if (isBatch) {
    if (value.prevStateValue) {
      if (grad.checkIgGrad)
        paddle::platform::CudaAtomicAdd(grad.checkIgGrad + frameIdx,
                                        rCheckIGrad);
      if (grad.checkFgGrad)
        paddle::platform::CudaAtomicAdd(grad.checkFgGrad + frameIdx,
                                        rCheckFGrad);
    }
    if (grad.checkOgGrad)
      paddle::platform::CudaAtomicAdd(grad.checkOgGrad + frameIdx, rCheckOGrad);
  } else {
    if (value.prevStateValue) {
      if (grad.checkIgGrad) grad.checkIgGrad[frameIdx] += rCheckIGrad;
      if (grad.checkFgGrad) grad.checkFgGrad[frameIdx] += rCheckFGrad;
    }
    if (grad.checkOgGrad) grad.checkOgGrad[frameIdx] += rCheckOGrad;
  }
}

template <class T, class Op>
void gpu_lstm_forward(const platform::DeviceContext& context, Op op,
                      LstmMetaValue<T> value, int frameSize, int batchSize,
                      activation_mode_t active_node,
                      activation_mode_t active_gate,
                      activation_mode_t active_state) {
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

  auto stream =
      reinterpret_cast<const platform::CUDADeviceContext&>(context).stream();
  if (batchSize == 1) {
    KeLstmForward<T, Op,
                  /* isBatch= */ false><<<grid, threads, 0, stream>>>(
        op, value, frameSize, batchSize, active_node, active_gate,
        active_state);
  } else {
    KeLstmForward<T, Op,
                  /* isBatch= */ true><<<grid, threads, 0, stream>>>(
        op, value, frameSize, batchSize, active_node, active_gate,
        active_state);
  }
}

template <class T, class Op>
void gpu_lstm_backward(const platform::DeviceContext& context, Op op,
                       LstmMetaValue<T> value, LstmMetaGrad<T> grad,
                       int frameSize, int batchSize,
                       activation_mode_t active_node,
                       activation_mode_t active_gate,
                       activation_mode_t active_state) {
  dim3 threads;
  dim3 grid;
  if (batchSize == 1) {
    int framePerBlock = frameSize <= 1024 ? frameSize : 1024;
    int frameBlocks = (frameSize + 1024 - 1) / 1024;
    threads = dim3(framePerBlock, 1);
    grid = dim3(frameBlocks, 1);
  } else {
    /* framePerBlock = 32 batchPerBlock = 16 */
    threads = dim3(32, 16);
    grid = dim3((frameSize + 32 - 1) / 32, (batchSize + 16 - 1) / 16);
  }

  auto stream =
      reinterpret_cast<const platform::CUDADeviceContext&>(context).stream();
  if (batchSize == 1) {
    KeLstmBackward<T, Op,
                   /* isBatch= */ false><<<grid, threads, 0, stream>>>(
        op, value, grad, frameSize, batchSize, active_node, active_gate,
        active_state);
  } else {
    KeLstmBackward<T, Op,
                   /* isBatch= */ true><<<grid, threads, 0, stream>>>(
        op, value, grad, frameSize, batchSize, active_node, active_gate,
        active_state);
  }
}

}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
