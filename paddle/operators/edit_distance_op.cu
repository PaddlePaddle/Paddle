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

#include <algorithm>
#include "paddle/framework/op_registry.h"
#include "paddle/platform/cuda_helper.h"
#include "paddle/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void FillFirstRow(T* dist, const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N + 1) {
    dist[idx] = idx;
  }
}

template <typename T>
__global__ void FillFirstColumn(T* dist, const int M, const int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < M + 1) {
    dist[idx * (N + 1)] = idx;
  }
}

template <typename T>
__global__ void Levenshtein(T* dist, const int* x1, const int* x2, const int M,
                            const int N, const int start) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = N;
  int index = start + idx * offset;
  int row = index / (N + 1);
  int col = index % (N + 1);
  if (row > 0 && col > 0 && row < M + 1 && col < N + 1) {
    int cost = x1[row - 1] == x2[col - 1] ? 0 : 1;
    int dels = dist[(row - 1) * (N + 1) + col] + 1;
    int ins = dist[row * (N + 1) + col - 1] + 1;
    int subs = dist[(row - 1) * (N + 1) + (col - 1)] + cost;
    dist[index] = min(dels, min(ins, subs));
  }
}

template <typename T>
__global__ void SetOutput(T* out, const T* dist, const int M, const int N,
                          bool normalized) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx == 0) {
    out[0] = normalized ? dist[M * (N + 1) + N] / N : dist[M * (N + 1) + N];
  }
}

template <typename Place, typename T>
class EditDistanceGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<framework::Tensor>("Out");

    auto* x1_t = ctx.Input<framework::Tensor>("Hyp");
    auto* x2_t = ctx.Input<framework::Tensor>("Ref");

    out_t->mutable_data<T>(ctx.GetPlace());
    auto out = out_t->data<T>();

    auto normalized = ctx.Attr<bool>("normalized");
    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                      ctx.device_context())
                      .stream();

    auto m = x1_t->numel();
    auto n = x2_t->numel();
    T distance = 0.0;
    if (m == 0 || n == 0) {
      distance = std::max(m, n);
      if (normalized) {
        distance = distance / n;
      }
      memory::Copy(boost::get<Place>(ctx.GetPlace()), out, platform::CPUPlace(),
                   &distance, sizeof(T), stream);
    } else {
      framework::Tensor dist_t;
      dist_t.Resize({m + 1, n + 1});
      dist_t.mutable_data<T>(ctx.GetPlace());
      auto dist = dist_t.data<T>();
      auto x1 = x1_t->data<int>();
      auto x2 = x2_t->data<int>();

      FillFirstColumn<T><<<1 + m / PADDLE_CUDA_NUM_THREADS,
                           PADDLE_CUDA_NUM_THREADS, 0, stream>>>(dist, m, n);

      FillFirstRow<T><<<1 + n / PADDLE_CUDA_NUM_THREADS,
                        PADDLE_CUDA_NUM_THREADS, 0, stream>>>(dist, n);
      // Compute the elements of distance matrix in the anti-diagonal diretion
      for (int64_t slice = 2; slice < m + n + 1; ++slice) {
        int z_m = slice < m + 1 ? 0 : slice - m;
        int z_n = slice < n + 1 ? 0 : slice - n;
        int size = slice - (z_m + z_n) + 1;  // number of elments in the same
                                             // anti-diagonal line to update
        // the start index at which computes from
        int start = slice < n + 1 ? slice : (z_n + 1) * (n + 1) - 1;
        Levenshtein<T><<<1 + (size - 1) / PADDLE_CUDA_NUM_THREADS,
                         PADDLE_CUDA_NUM_THREADS, 0, stream>>>(dist, x1, x2, m,
                                                               n, start);
      }
      SetOutput<T><<<1, 1, 0, stream>>>(out, dist, m, n, normalized);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    edit_distance,
    ops::EditDistanceGPUKernel<paddle::platform::CUDAPlace, float>);
