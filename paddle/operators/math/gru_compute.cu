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

#include "paddle/operators/math/detail/gru_gpu_kernel.h"
#include "paddle/operators/math/detail/gru_kernel.h"
#include "paddle/operators/math/gru_compute.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct GRUUnitFunctor<platform::GPUPlace, T> {
  static void compute(const platform::DeviceContext &context,
                      hl_gru_value<T> value, int frameSize, int batchSize,
                      activation_mode_t active_node,
                      activation_mode_t active_gate) {
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext &>(context).stream();
    dim3 threads;
    dim3 grid;
    if (batchSize == 1) {
      int framePerBlock = frameSize <= 1024 ? frameSize : 1024;
      int frameBlocks = (frameSize + 1024 - 1) / 1024;
      threads = dim3(framePerBlock, 1);
      grid = dim3(frameBlocks, 1);
    } else {
      threads = dim3(32, 32);
      grid = dim3((frameSize + 32 - 1) / 32, (batchSize + 32 - 1) / 32);
    }

    if (value.prevOutValue) {
      math::gemm<platform::GPUPlace, T>(
          context, false, false, batchSize, frameSize * 2, frameSize, 1,
          value.prevOutValue, frameSize, value.gateWeight, frameSize * 2, 1,
          value.gateValue, frameSize * 3);
    }

    if (batchSize == 1) {
      detail::KeGruForwardResetOutput<detail::forward::gru_resetOutput<T>,
                                      /* isBatch= */ false,
                                      T><<<grid, threads, 0, stream>>>(
          detail::forward::gru_resetOutput<T>(), value.gateValue,
          value.resetOutputValue, value.prevOutValue, frameSize, batchSize,
          active_gate);
    } else {
      detail::KeGruForwardResetOutput<detail::forward::gru_resetOutput<T>,
                                      /* isBatch= */ true,
                                      T><<<grid, threads, 0, stream>>>(
          detail::forward::gru_resetOutput<T>(), value.gateValue,
          value.resetOutputValue, value.prevOutValue, frameSize, batchSize,
          active_gate);
    }

    if (value.prevOutValue) {
      math::gemm<platform::GPUPlace, T>(
          context, false, false, batchSize, frameSize, frameSize, 1,
          value.resetOutputValue, frameSize, value.stateWeight, frameSize, 1,
          value.gateValue + frameSize * 2, frameSize * 3);
    }

    if (batchSize == 1) {
      detail::KeGruForwardFinalOutput<detail::forward::gru_finalOutput<T>,
                                      /* isBatch= */ false,
                                      T><<<grid, threads, 0, stream>>>(
          detail::forward::gru_finalOutput<T>(), value.gateValue,
          value.prevOutValue, value.outputValue, frameSize, batchSize,
          active_node);
    } else {
      detail::KeGruForwardFinalOutput<detail::forward::gru_finalOutput<T>,
                                      /* isBatch= */ true,
                                      T><<<grid, threads, 0, stream>>>(
          detail::forward::gru_finalOutput<T>(), value.gateValue,
          value.prevOutValue, value.outputValue, frameSize, batchSize,
          active_node);
    }
  }
};

template <typename T>
struct GRUUnitGradFunctor<platform::GPUPlace, T> {
  static void compute(const platform::DeviceContext &context,
                      hl_gru_value<T> value, hl_gru_grad<T> grad, int frameSize,
                      int batchSize, activation_mode_t active_node,
                      activation_mode_t active_gate) {
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext &>(context).stream();
    dim3 threads;
    dim3 grid;
    if (batchSize == 1) {
      int framePerBlock = frameSize <= 1024 ? frameSize : 1024;
      int frameBlocks = (frameSize + 1024 - 1) / 1024;
      threads = dim3(framePerBlock, 1);
      grid = dim3(frameBlocks, 1);
    } else {
      threads = dim3(32, 32);
      grid = dim3((frameSize + 32 - 1) / 32, (batchSize + 32 - 1) / 32);
    }

    if (batchSize == 1) {
      detail::KeGruBackwardStateGrad<
          detail::backward::gru_stateGrad<T>,
          /* isBatch= */ false><<<grid, threads, 0, stream>>>(
          detail::backward::gru_stateGrad<T>(), value.gateValue, grad.gateGrad,
          value.prevOutValue, grad.prevOutGrad, grad.outputGrad, frameSize,
          batchSize, active_node);
    } else {
      detail::KeGruBackwardStateGrad<
          detail::backward::gru_stateGrad<T>,
          /* isBatch= */ true><<<grid, threads, 0, stream>>>(
          detail::backward::gru_stateGrad<T>(), value.gateValue, grad.gateGrad,
          value.prevOutValue, grad.prevOutGrad, grad.outputGrad, frameSize,
          batchSize, active_node);
    }

    if (value.prevOutValue && grad.prevOutGrad) {
      math::gemm<platform::GPUPlace, T>(
          context, false, true, batchSize, frameSize, frameSize, 1,
          grad.gateGrad + frameSize * 2, frameSize * 3, value.stateWeight,
          frameSize, 0, grad.resetOutputGrad, frameSize);

      if (grad.stateWeightGrad) {
        math::gemm<platform::GPUPlace, T>(
            context, true, false, frameSize, frameSize, batchSize, 1,
            value.resetOutputValue, frameSize, grad.gateGrad + frameSize * 2,
            frameSize * 3, 1, grad.stateWeightGrad, frameSize);
      }
    }

    if (batchSize == 1) {
      detail::KeGruBackwardResetGrad<
          detail::backward::gru_resetGrad<T>,
          /* isBatch= */ false><<<grid, threads, 0, stream>>>(
          detail::backward::gru_resetGrad<T>(), value.gateValue, grad.gateGrad,
          value.prevOutValue, grad.prevOutGrad, grad.resetOutputGrad, frameSize,
          batchSize, active_gate);
    } else {
      detail::KeGruBackwardResetGrad<
          detail::backward::gru_resetGrad<T>,
          /* isBatch= */ true><<<grid, threads, 0, stream>>>(
          detail::backward::gru_resetGrad<T>(), value.gateValue, grad.gateGrad,
          value.prevOutValue, grad.prevOutGrad, grad.resetOutputGrad, frameSize,
          batchSize, active_gate);
    }

    if (grad.prevOutGrad && value.prevOutValue) {
      math::gemm<platform::GPUPlace, T>(
          context, false, true, batchSize, frameSize, frameSize * 2, 1,
          grad.gateGrad, frameSize * 3, value.gateWeight, frameSize * 2, 1,
          grad.prevOutGrad, frameSize);

      if (grad.gateWeightGrad) {
        math::gemm<platform::GPUPlace, T>(
            context, true, false, frameSize, frameSize * 2, batchSize, 1,
            value.prevOutValue, frameSize, grad.gateGrad, frameSize * 3, 1,
            grad.gateWeightGrad, frameSize * 2);
      }
    }
  }
};

template struct GRUUnitFunctor<platform::GPUPlace, float>;
template struct GRUUnitFunctor<platform::GPUPlace, double>;
template struct GRUUnitGradFunctor<platform::GPUPlace, float>;
template struct GRUUnitGradFunctor<platform::GPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
