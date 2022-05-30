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

#include "paddle/phi/kernels/argsort_kernel.h"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#ifdef __HIPCC__
namespace rocprim {
namespace detail {
template <>
struct radix_key_codec_base<phi::dtype::float16>
    : radix_key_codec_integral<phi::dtype::float16, uint16_t> {};
}  // namespace detail
}  // namespace rocprim
#else
// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<phi::dtype::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::dtype::float16> {};
}  // namespace cub
#endif

namespace phi {

template <typename T, typename IndType>
static __global__ void FillFlattenGrad(const T* dO,
                                       const IndType* indices,
                                       int64_t size,
                                       T* dX) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    dX[indices[i]] = dO[i];
  }
}

template <typename T, typename IndType>
static __global__ void FillGrad(const T* dO,
                                const IndType* indices,
                                T* dX,
                                IndType num_rows,
                                IndType num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (IndType j = row_id; j < num_rows; j += gridDim.x) {
    for (IndType i = col_id; i < num_cols; i += blockDim.x) {
      dX[j * num_cols + indices[j * num_cols + i]] = dO[j * num_cols + i];
    }
  }
}

template <typename T, typename IndType>
void ArgFullAssign(const phi::GPUContext& ctx,
                   const DenseTensor* dO,
                   const DenseTensor* indices,
                   DenseTensor* dX,
                   const IndType num_rows,
                   const IndType num_cols) {
  auto cu_stream = ctx.stream();

  auto ComputeBlockSize = [](IndType col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else if (col > 64 && col <= 128)
      return 128;
    else
      return 64;
  };

  int block_size = ComputeBlockSize(num_cols);

  int maxGridDimX = ctx.GetCUDAMaxGridDimSize()[0];
  // actually, int num_rows < max_grid_size
  int grid_size = num_rows < maxGridDimX ? num_rows : maxGridDimX;
  FillGrad<<<grid_size, block_size, 0, cu_stream>>>(dO->data<T>(),
                                                    indices->data<IndType>(),
                                                    dX->data<T>(),
                                                    num_rows,
                                                    num_cols);
}

template <typename T>
void ArgFlattenAssign(const phi::GPUContext& ctx,
                      const DenseTensor* dO,
                      const DenseTensor* indices,
                      int64_t size,
                      DenseTensor* dX) {
  auto cu_stream = ctx.stream();

  const int64_t block_size =
      std::min(size, static_cast<int64_t>(ctx.GetMaxThreadsPerBlock()));
  int64_t max_threads = ctx.GetMaxPhysicalThreadCount();
  const int64_t max_blocks =
      std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
  const int64_t grid_size =
      std::min(max_blocks, (size + block_size - 1) / block_size);

  FillFlattenGrad<<<grid_size, block_size, 0, cu_stream>>>(
      dO->data<T>(), indices->data<int64_t>(), size, dX->data<T>());
}

template <typename T, typename Context>
void ArgsortGradKernel(const Context& dev_ctx,
                       const DenseTensor& indices,
                       const DenseTensor& input,
                       const DenseTensor& out_grad,
                       int axis,
                       bool descending,
                       DenseTensor* in_grad) {
  dev_ctx.template Alloc<T>(in_grad);
  if (out_grad.numel() == 0) return;
  auto in_dims = in_grad->dims();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  int64_t size = in_grad->numel();

  // Parallel acceleration when the input size is equal to the length of the
  // ‘axis’ dimension.
  // Compared to 'special case for full sort' below, the gradient calculation
  // is 10 times faster.
  if (size == in_dims[axis]) {
    ArgFlattenAssign<T>(dev_ctx, &out_grad, &indices, size, in_grad);
    return;
  }

  // Special case for full sort, speedup ~190x.
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int64_t input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];
    ArgFullAssign<T, int64_t>(
        dev_ctx, &out_grad, &indices, in_grad, input_height, input_width);
  } else {
    // if not full sort, do transpose first
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.push_back(i);
    }
    trans.push_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans.push_back(i);
    }
    trans.push_back(axis);
    phi::DDim trans_dims(in_dims);
    for (int i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
    }

    DenseTensor trans_dO;
    trans_dO.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&trans_dO);
    DenseTensor trans_ind;
    trans_ind.Resize(trans_dims);
    dev_ctx.template Alloc<int64_t>(&trans_ind);
    TransposeKernel<T, Context>(dev_ctx, out_grad, trans, &trans_dO);
    TransposeKernel<int64_t, Context>(dev_ctx, indices, trans, &trans_ind);

    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    DenseTensor tmp_out;
    tmp_out.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&tmp_out);

    ArgFullAssign<T, int64_t>(
        dev_ctx, &trans_dO, &trans_ind, &tmp_out, input_height, input_width);

    // transpose back
    TransposeKernel<T, Context>(dev_ctx, tmp_out, trans, in_grad);
    return;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(argsort_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArgsortGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
