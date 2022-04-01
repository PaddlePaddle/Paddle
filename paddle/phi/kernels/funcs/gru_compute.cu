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

#include <paddle/fluid/platform/device_context.h>
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/detail/gru_gpu_kernel.h"
#include "paddle/phi/kernels/funcs/detail/gru_kernel.h"
#include "paddle/phi/kernels/funcs/gru_compute.h"

namespace phi {
namespace funcs {

template <typename T>
struct GRUUnitFunctor<paddle::platform::CUDADeviceContext, T> {
  static void compute(const paddle::platform::CUDADeviceContext &context,
                      GRUMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      const phi::funcs::detail::ActivationType active_node,
                      const phi::funcs::detail::ActivationType active_gate,
                      bool origin_mode) {
    auto stream = context.stream();
    dim3 threads;
    dim3 grid;
    if (batch_size == 1) {
      if (context.GetComputeCapability() >= 70) {
        if (frame_size < 16) {
          constexpr int tiled_size = 8;
          int frame_blocks = (frame_size * 2 + tiled_size - 1) / tiled_size;
          threads = dim3(tiled_size, 1);
          grid = dim3(frame_blocks, 1);
          detail::KeFastCollectiveGruGate<
              T,
              tiled_size><<<grid, threads, 0, stream>>>(
              value.gate_value,
              value.prev_out_value,
              value.gate_weight,
              value.reset_output_value,
              frame_size,
              active_gate);

          frame_blocks = (frame_size + tiled_size - 1) / tiled_size;
          grid = dim3(frame_blocks, 1);
          detail::KeFastCollectiveGruOut<
              T,
              tiled_size><<<grid, threads, 0, stream>>>(
              value.state_weight,
              value.prev_out_value,
              value.output_value,
              value.gate_value,
              value.reset_output_value,
              frame_size,
              active_node,
              origin_mode);
        } else {
          constexpr int tiled_size = 16;
          int frame_blocks = (frame_size * 2 + tiled_size - 1) / tiled_size;
          threads = dim3(tiled_size, 1);
          grid = dim3(frame_blocks, 1);
          detail::KeFastCollectiveGruGate<
              T,
              tiled_size><<<grid, threads, 0, stream>>>(
              value.gate_value,
              value.prev_out_value,
              value.gate_weight,
              value.reset_output_value,
              frame_size,
              active_gate);

          frame_blocks = (frame_size + tiled_size - 1) / tiled_size;
          grid = dim3(frame_blocks, 1);
          detail::KeFastCollectiveGruOut<
              T,
              tiled_size><<<grid, threads, 0, stream>>>(
              value.state_weight,
              value.prev_out_value,
              value.output_value,
              value.gate_value,
              value.reset_output_value,
              frame_size,
              active_node,
              origin_mode);
        }
        return;
      } else {
        int frame_per_block = frame_size <= 1024 ? frame_size : 1024;
        int frame_blocks = (frame_size + 1024 - 1) / 1024;
        threads = dim3(frame_per_block, 1);
        grid = dim3(frame_blocks, 1);
      }
    } else {
      threads = dim3(32, 32);
      grid = dim3((frame_size + 32 - 1) / 32, (batch_size + 32 - 1) / 32);
    }
    auto blas =
        phi::funcs::GetBlas<paddle::platform::CUDADeviceContext, T>(context);
    if (value.prev_out_value) {
      blas.GEMM(false,
                false,
                batch_size,
                frame_size * 2,
                frame_size,
                1,
                value.prev_out_value,
                frame_size,
                value.gate_weight,
                frame_size * 2,
                1,
                value.gate_value,
                frame_size * 3);
    }

    if (batch_size == 1) {
      detail::KeGruForwardResetOutput<
          phi::funcs::detail::forward::gru_resetOutput<T>,
          /* is_batch= */ false,
          T><<<grid, threads, 0, stream>>>(
          phi::funcs::detail::forward::gru_resetOutput<T>(),
          value.gate_value,
          value.reset_output_value,
          value.prev_out_value,
          frame_size,
          batch_size,
          active_gate);
    } else {
      detail::KeGruForwardResetOutput<
          phi::funcs::detail::forward::gru_resetOutput<T>,
          /* is_batch= */ true,
          T><<<grid, threads, 0, stream>>>(
          phi::funcs::detail::forward::gru_resetOutput<T>(),
          value.gate_value,
          value.reset_output_value,
          value.prev_out_value,
          frame_size,
          batch_size,
          active_gate);
    }

    if (value.prev_out_value) {
      blas.GEMM(false,
                false,
                batch_size,
                frame_size,
                frame_size,
                1,
                value.reset_output_value,
                frame_size,
                value.state_weight,
                frame_size,
                1,
                value.gate_value + frame_size * 2,
                frame_size * 3);
    }

    if (batch_size == 1) {
      detail::KeGruForwardFinalOutput<
          phi::funcs::detail::forward::gru_finalOutput<T>,
          /* is_batch= */ false,
          T><<<grid, threads, 0, stream>>>(
          phi::funcs::detail::forward::gru_finalOutput<T>(),
          value.gate_value,
          value.prev_out_value,
          value.output_value,
          frame_size,
          batch_size,
          active_node,
          origin_mode);
    } else {
      detail::KeGruForwardFinalOutput<
          phi::funcs::detail::forward::gru_finalOutput<T>,
          /* is_batch= */ true,
          T><<<grid, threads, 0, stream>>>(
          phi::funcs::detail::forward::gru_finalOutput<T>(),
          value.gate_value,
          value.prev_out_value,
          value.output_value,
          frame_size,
          batch_size,
          active_node,
          origin_mode);
    }
  }
};

template <typename T>
struct GRUUnitGradFunctor<paddle::platform::CUDADeviceContext, T> {
  static void compute(const paddle::platform::CUDADeviceContext &context,
                      GRUMetaValue<T> value,
                      GRUMetaGrad<T> grad,
                      int frame_size,
                      int batch_size,
                      const phi::funcs::detail::ActivationType active_node,
                      const phi::funcs::detail::ActivationType active_gate,
                      bool origin_mode) {
    auto stream = context.stream();
    dim3 threads;
    dim3 grid;
    if (batch_size == 1) {
      int frame_per_block = frame_size <= 1024 ? frame_size : 1024;
      int frame_blocks = (frame_size + 1024 - 1) / 1024;
      threads = dim3(frame_per_block, 1);
      grid = dim3(frame_blocks, 1);
    } else {
      threads = dim3(32, 32);
      grid = dim3((frame_size + 32 - 1) / 32, (batch_size + 32 - 1) / 32);
    }

    if (batch_size == 1) {
      detail::KeGruBackwardStateGrad<
          phi::funcs::detail::backward::gru_stateGrad<T>,
          /* is_batch= */ false><<<grid, threads, 0, stream>>>(
          phi::funcs::detail::backward::gru_stateGrad<T>(),
          value.gate_value,
          grad.gate_grad,
          value.prev_out_value,
          grad.prev_out_grad,
          grad.output_grad,
          frame_size,
          batch_size,
          active_node,
          origin_mode);
    } else {
      detail::KeGruBackwardStateGrad<
          phi::funcs::detail::backward::gru_stateGrad<T>,
          /* is_batch= */ true><<<grid, threads, 0, stream>>>(
          phi::funcs::detail::backward::gru_stateGrad<T>(),
          value.gate_value,
          grad.gate_grad,
          value.prev_out_value,
          grad.prev_out_grad,
          grad.output_grad,
          frame_size,
          batch_size,
          active_node,
          origin_mode);
    }

    auto blas =
        phi::funcs::GetBlas<paddle::platform::CUDADeviceContext, T>(context);

    if (value.prev_out_value && grad.prev_out_grad) {
      blas.GEMM(false,
                true,
                batch_size,
                frame_size,
                frame_size,
                1,
                grad.gate_grad + frame_size * 2,
                frame_size * 3,
                value.state_weight,
                frame_size,
                0,
                grad.reset_output_grad,
                frame_size);

      if (grad.state_weight_grad) {
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size,
                  batch_size,
                  1,
                  value.reset_output_value,
                  frame_size,
                  grad.gate_grad + frame_size * 2,
                  frame_size * 3,
                  1,
                  grad.state_weight_grad,
                  frame_size);
      }
    }

    if (batch_size == 1) {
      detail::KeGruBackwardResetGrad<
          phi::funcs::detail::backward::gru_resetGrad<T>,
          /* is_batch= */ false><<<grid, threads, 0, stream>>>(
          phi::funcs::detail::backward::gru_resetGrad<T>(),
          value.gate_value,
          grad.gate_grad,
          value.prev_out_value,
          grad.prev_out_grad,
          grad.reset_output_grad,
          frame_size,
          batch_size,
          active_gate);
    } else {
      detail::KeGruBackwardResetGrad<
          phi::funcs::detail::backward::gru_resetGrad<T>,
          /* is_batch= */ true><<<grid, threads, 0, stream>>>(
          phi::funcs::detail::backward::gru_resetGrad<T>(),
          value.gate_value,
          grad.gate_grad,
          value.prev_out_value,
          grad.prev_out_grad,
          grad.reset_output_grad,
          frame_size,
          batch_size,
          active_gate);
    }

    if (grad.prev_out_grad && value.prev_out_value) {
      blas.GEMM(false,
                true,
                batch_size,
                frame_size,
                frame_size * 2,
                1,
                grad.gate_grad,
                frame_size * 3,
                value.gate_weight,
                frame_size * 2,
                1,
                grad.prev_out_grad,
                frame_size);

      if (grad.gate_weight_grad) {
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size * 2,
                  batch_size,
                  1,
                  value.prev_out_value,
                  frame_size,
                  grad.gate_grad,
                  frame_size * 3,
                  1,
                  grad.gate_weight_grad,
                  frame_size * 2);
      }
    }
  }
};

template struct GRUUnitFunctor<paddle::platform::CUDADeviceContext, float>;
template struct GRUUnitFunctor<paddle::platform::CUDADeviceContext, double>;
template struct GRUUnitGradFunctor<paddle::platform::CUDADeviceContext, float>;
template struct GRUUnitGradFunctor<paddle::platform::CUDADeviceContext, double>;

}  // namespace funcs
}  // namespace phi
