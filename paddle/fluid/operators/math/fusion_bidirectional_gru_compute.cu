// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <paddle/fluid/platform/device_context.h>
#include <fstream>
#include <sstream>
#include <type_traits>
#include "paddle/fluid/operators/math/detail/fusion_bidirectional_gru_kernel.h"
#include "paddle/fluid/operators/math/fusion_bidirectional_gru_compute.h"

namespace paddle {
namespace operators {
namespace math {

#define TILED_SIZE 32
#define TILED_X 8
#define TILED_Y 8

template <typename T>
struct FusionBidirectionalGRUFunctor<platform::CUDADeviceContext, T> {
  static void compute(const platform::CUDADeviceContext &context,
                      FusionGRUMetaValue<T> v, int m, int n, int k, int q,
                      const detail::ActivationType active_gate,
                      const detail::ActivationType active_node, int reverse) {
    auto stream = context.stream();
    dim3 threads = dim3(TILED_Y, TILED_Y, 1);
    dim3 grids =
        dim3((m + TILED_Y - 1) / TILED_X, (n + TILED_Y - 1) / TILED_Y, 1);

    // mul + elementwise_add
    if (reverse == 0) {
      detail::FusionGRUCUDAKernel_premul<
          T, TILED_Y><<<grids, threads, 0, stream>>>(
          m, n, k, v.x, v.wx0, v.wx1, v.bias_x0, v.bias_x1, v.mul_o0, v.mul_o1,
          v.bias_h0, v.bias_h1);
    } else if (reverse == 1) {
      detail::FusionGRUCUDAKernel_premul<
          T, TILED_Y><<<grids, threads, 0, stream>>>(
          m, n, k, v.x, v.wx0, v.wx1, v.bias_x0, v.bias_x1, v.mul_o1, v.mul_o0,
          v.bias_h0, v.bias_h1);
    }

    int seq_len = n;
    int frame_size = m / 3;
    // GRU step
    for (int i = 0; i < seq_len; ++i) {
      threads = dim3(TILED_SIZE, 1, 1);
      grids = dim3((frame_size * 2 + TILED_SIZE - 1) / TILED_SIZE, 1, 1);

      detail::FusionGRUCUDAKernelGru_gate<
          T, TILED_SIZE><<<grids, threads, 0, stream>>>(
          frame_size, &v.mul_o0[i * m], &v.mul_o1[i * m], v.wh0, v.wh1, v.hp0,
          v.hp1, v.gate0, v.gate1, active_gate, i);

      grids = dim3((frame_size + TILED_SIZE - 1) / TILED_SIZE, 1, 1);

      if (reverse == 0) {
        detail::FusionGRUCUDAKernelGru_out<
            T, TILED_SIZE><<<grids, threads, 0, stream>>>(
            frame_size, &v.mul_o0[i * m], &v.mul_o1[i * m],
            &v.gru_o0[(n - 1 - i) * frame_size], &v.gru_o1[i * frame_size],
            v.wh0 + frame_size * 2 * frame_size,
            v.wh1 + frame_size * 2 * frame_size, v.hp0, v.hp1, v.gate0, v.gate1,
            active_node);
      } else if (reverse == 1) {
        detail::FusionGRUCUDAKernelGru_out<
            T, TILED_SIZE><<<grids, threads, 0, stream>>>(
            frame_size, &v.mul_o0[i * m], &v.mul_o1[i * m],
            &v.gru_o0[i * frame_size], &v.gru_o1[(n - 1 - i) * frame_size],
            v.wh0 + frame_size * 2 * frame_size,
            v.wh1 + frame_size * 2 * frame_size, v.hp0, v.hp1, v.gate0, v.gate1,
            active_node);
      }

      if (reverse == 0) {
        v.hp0 = &v.gru_o0[(n - 1 - i) * frame_size];
        v.hp1 = &v.gru_o1[i * frame_size];
      } else if (reverse == 1) {
        v.hp1 = &v.gru_o1[(n - 1 - i) * frame_size];
        v.hp0 = &v.gru_o0[i * frame_size];
      }
    }

    threads = dim3(TILED_X, TILED_X);
    grids = dim3((q + TILED_X - 1) / TILED_X, (n + TILED_X - 1) / TILED_X);
    // mul + sum
    detail::FusionGRUCUDAKernel_sufmul<T,
                                       TILED_X><<<grids, threads, 0, stream>>>(
        q, n, frame_size, v.gru_o0, v.gru_o1, v.wx2, v.wx3, v.out);
  }
};

template struct FusionBidirectionalGRUFunctor<platform::CUDADeviceContext,
                                              float>;
template struct FusionBidirectionalGRUFunctor<platform::CUDADeviceContext,
                                              double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
