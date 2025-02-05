// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/edit_distance_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

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
__global__ void Levenshtein(T* dist,
                            const int64_t* x1,
                            const int64_t* x2,
                            const int M,
                            const int N,
                            const int start) {
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
__global__ void SetOutput(
    T* out, const T* dist, const int M, const int N, bool normalized) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx == 0) {
    out[0] = normalized ? dist[M * (N + 1) + N] / N : dist[M * (N + 1) + N];
  }
}

template <typename T, typename Context>
void EditDistanceKernel(const Context& ctx,
                        const DenseTensor& hyps,
                        const DenseTensor& refs,
                        const paddle::optional<DenseTensor>& hypslength,
                        const paddle::optional<DenseTensor>& refslength,
                        bool normalized,
                        DenseTensor* sequencenum,
                        DenseTensor* out) {
  ctx.template Alloc<int64_t>(sequencenum);
  auto batch_size = hyps.dims()[0];

  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();

  phi::Vector<size_t> hyp_lod(batch_size + 1);
  phi::Vector<size_t> ref_lod(batch_size + 1);

  bool use_length = hypslength.get_ptr() != nullptr;

  if (use_length) {
    DenseTensor hyp_length_cpu;
    DenseTensor ref_length_cpu;
    phi::Copy(
        ctx, *(hypslength.get_ptr()), phi::CPUPlace(), false, &hyp_length_cpu);
    phi::Copy(
        ctx, *(refslength.get_ptr()), phi::CPUPlace(), false, &ref_length_cpu);

    for (auto i = 0; i < batch_size; i++) {
      hyp_lod[i + 1] = hyp_lod[i] + hyp_length_cpu.data<int64_t>()[i];
      ref_lod[i + 1] = ref_lod[i] + ref_length_cpu.data<int64_t>()[i];
    }

  } else {
    hyp_lod = hyps.lod()[0];
    ref_lod = refs.lod()[0];
  }

  if (normalized) {
    for (size_t i = 1; i < ref_lod.size(); ++i) {
      PADDLE_ENFORCE_GT(
          ref_lod[i],
          ref_lod[i - 1],
          errors::InvalidArgument("Reference string %d is empty.", i));
    }
  }

  const size_t num_strs = hyp_lod.size() - 1;
  phi::funcs::SetConstant<GPUContext, int64_t> set_constant;
  set_constant(ctx, sequencenum, static_cast<int64_t>(num_strs));

  out->Resize({static_cast<int64_t>(num_strs), 1});
  ctx.template Alloc<T>(out);
  auto out_data = out->data<T>();

  T distance = 0.0;
  for (size_t num = 0; num < num_strs; num++) {
    auto m = static_cast<int64_t>(hyp_lod[num + 1] - hyp_lod[num]);
    auto n = static_cast<int64_t>(ref_lod[num + 1] - ref_lod[num]);
    if (m == 0 || n == 0) {
      distance = std::max(m, n);
      if (normalized) {
        distance = distance / n;
      }
      memory_utils::Copy(ctx.GetPlace(),
                         out_data + num,
                         CPUPlace(),
                         &distance,
                         sizeof(T),
                         stream);
    } else {
      DenseTensor dist_t;
      dist_t.Resize({m + 1, n + 1});
      ctx.template Alloc<T>(&dist_t);
      auto dist = dist_t.data<T>();
      auto hyp_offset = use_length ? num * hyps.dims()[1] : hyp_lod[num];
      auto ref_offset = use_length ? num * refs.dims()[1] : ref_lod[num];
      auto x1 = hyps.data<int64_t>() + hyp_offset;
      auto x2 = refs.data<int64_t>() + ref_offset;

      FillFirstColumn<T><<<1 + m / PADDLE_CUDA_NUM_THREADS,
                           PADDLE_CUDA_NUM_THREADS,
                           0,
                           stream>>>(dist, m, n);

      FillFirstRow<T><<<1 + n / PADDLE_CUDA_NUM_THREADS,
                        PADDLE_CUDA_NUM_THREADS,
                        0,
                        stream>>>(dist, n);

      // Compute the elements of distance matrix in the anti-diagonal diretion
      for (int64_t slice = 2; slice < m + n + 1; ++slice) {
        int z_m = slice < m + 1 ? 0 : slice - m;
        int z_n = slice < n + 1 ? 0 : slice - n;
        int size = slice - (z_m + z_n) + 1;  // number of elements in the same
                                             // anti-diagonal line to update
        // the start index at which computes from
        int start = slice < n + 1 ? slice : (z_n + 1) * (n + 1) - 1;
        Levenshtein<T><<<1 + (size - 1) / PADDLE_CUDA_NUM_THREADS,
                         PADDLE_CUDA_NUM_THREADS,
                         0,
                         stream>>>(dist, x1, x2, m, n, start);
      }
      SetOutput<T><<<1, 1, 0, stream>>>(out_data + num, dist, m, n, normalized);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    edit_distance, GPU, ALL_LAYOUT, phi::EditDistanceKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
