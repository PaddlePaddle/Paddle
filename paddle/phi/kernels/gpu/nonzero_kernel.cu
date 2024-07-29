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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "paddle/common/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/select_impl.cu.h"
#include "paddle/phi/kernels/nonzero_kernel.h"

namespace phi {
template <typename MaskT, typename IndexT, typename OutT>
struct IndexFunctor {
  IndexT strides[phi::DDim::kMaxRank];
  int rank;

  explicit IndexFunctor(const phi::DDim &in_dims) {
    rank = in_dims.size();
    // Get strides according to in_dims
    strides[0] = 1;
    for (IndexT i = 1; i < rank; i++) {
      strides[i] = strides[i - 1] * in_dims[rank - i];
    }
  }

  HOSTDEVICE inline void operator()(OutT *out,
                                    const MaskT *mask,
                                    const IndexT *index,
                                    const int num) {
    int store_fix = 0;
    for (int idx = 0; idx < num; idx++) {
      if (mask==nullptr || mask[idx]) {
        IndexT data_index = index[idx];
        // get index
        for (int rank_id = rank - 1; rank_id >= 0; --rank_id) {
          out[store_fix] = static_cast<OutT>(data_index / strides[rank_id]);
          data_index = data_index % strides[rank_id];
          store_fix++;
        }
      }
    }
  }
};

template <typename Functor, int VecSize>
__global__ void ConvertIdxKernel(int64_t *out,
                                 const int64_t *in,
                                 Functor index_functor,
                                 const int64_t numel,
                                 const int64_t rank) {
  using VecType = kps::details::VectorType<int64_t, VecSize>;
  int64_t ins[VecSize];
  int64_t outs[VecSize * phi::DDim::kMaxRank];
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel / VecSize) {
    reinterpret_cast<VecType *>(ins)[0] =
        reinterpret_cast<const VecType *>(in)[idx];

    index_functor(outs, nullptr, ins, VecSize);

#pragma unroll
    for (int i = 0; i < VecSize * rank; ++i) {
      out[idx * VecSize * rank + i] = outs[i];
    }
  }

  if (idx == numel / VecSize) {
#pragma unroll
    for (int64_t i = idx * VecSize; i < numel; ++i) {
      index_functor(&out[i * rank], nullptr, &in[i], 1);
    }
  }
}

template <typename T>
struct nonZero {
  HOSTDEVICE __forceinline__ bool operator()(const T &x) const {
    return x != (T)0;
  }
};

template <typename T, typename Context>
void NonZeroKernel(const Context &dev_ctx,
                   const DenseTensor &condition,
                   DenseTensor *out) {
  // DenseTensor in_data;
  auto dims = condition.dims();
  using Functor = IndexFunctor<T, int64_t, int64_t>;
  Functor index_functor = Functor(dims);
  // phi::funcs::SelectKernel<T, T, int64_t, 0, Functor>(
  //     dev_ctx, condition, in_data, out, index_functor);

  auto stream = dev_ctx.stream();
  const int64_t numel = condition.numel();
  const T *mask_data = condition.data<T>();
  const int64_t rank = dims.size();

  DenseTensor linIdx;
  int64_t *lin_data;
  if (rank == 1) {
    out->Resize(common::make_ddim({numel}));
    lin_data = dev_ctx.template Alloc<int64_t>(out);
  } else {
    linIdx.Resize(common::make_ddim({numel}));
    lin_data = dev_ctx.template Alloc<int64_t>(&linIdx);
  }

  int64_t *end_ptr =
      thrust::copy_if(thrust::device.on(stream),
                      thrust::make_counting_iterator<int64_t>(0),
                      thrust::make_counting_iterator<int64_t>(numel),
                      mask_data,
                      lin_data,
                      nonZero<T>());
  dev_ctx.Wait();
  int64_t keep_num = (int64_t)(end_ptr - lin_data);
  out->Resize(common::make_ddim({keep_num, rank}));
  if (keep_num <= 0 || rank == 1) return;

  int64_t *out_data = dev_ctx.template Alloc<int64_t>(out);
  const int kVecSize = sizeof(float4) / sizeof(int64_t);
  const int block = 256;
  const int num_per_block = block * kVecSize;
  const int grid = (keep_num + num_per_block - 1) / num_per_block;

  ConvertIdxKernel<Functor, kVecSize><<<grid, block, 0, stream>>>(
      out_data, lin_data, index_functor, keep_num, rank);
}
}  // namespace phi

PD_REGISTER_KERNEL(nonzero,
                   GPU,
                   ALL_LAYOUT,
                   phi::NonZeroKernel,
                   int64_t,
                   int,
                   int16_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   float,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
