/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/phi/kernels/funcs/sparse/flatten_indices.cu.h"
#include "paddle/phi/kernels/funcs/sparse/scatter.cu.h"
#include "paddle/phi/kernels/funcs/sparse/utils.cu.h"
#include "paddle/phi/kernels/sparse/coalesced_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
void CoalescedGPUKernel(const GPUContext& dev_ctx,
                        const SparseCooTensor& x,
                        SparseCooTensor* out) {
  const DenseTensor& x_indices = x.non_zero_indices();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor out_indices = phi::EmptyLike<IntT>(dev_ctx, x_indices);
  DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, x_values);

  const int64_t nnz = x.nnz();
  const int64_t sparse_dim = x.non_zero_indices().dims()[0];
  std::vector<IntT> sparse_offsets(sparse_dim);

  phi::funcs::sparse::CalcOffsetsPerDim<IntT>(
      x.dims(), sparse_dim, sparse_offsets.data());

  DenseTensorMeta sparse_offset_meta(
      paddle::experimental::CppTypeToDataType<IntT>::Type(),
      {sparse_dim},
      DataLayout::NCHW);
  DenseTensor d_sparse_offsets =
      phi::Empty<GPUContext>(dev_ctx, std::move(sparse_offset_meta));
  DenseTensor indexs = phi::Empty(
      dev_ctx, DenseTensorMeta(x_indices.dtype(), {nnz}, x_indices.layout()));
  IntT* indexs_ptr = indexs.data<IntT>();

  phi::backends::gpu::GpuMemcpyAsync(d_sparse_offsets.data<IntT>(),
                                     sparse_offsets.data(),
                                     sizeof(IntT) * sparse_dim,
#ifdef PADDLE_WITH_HIP
                                     hipMemcpyHostToDevice,
#else
                                     cudaMemcpyHostToDevice,
#endif
                                     dev_ctx.stream());

  // 1. flatten indices
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, nnz, 1);
  phi::funcs::sparse::FlattenIndicesKernel<<<config.block_per_grid,
                                             config.thread_per_block,
                                             0,
                                             dev_ctx.stream()>>>(
      x.non_zero_indices().data<IntT>(),
      d_sparse_offsets.data<IntT>(),
      indexs.numel(),
      sparse_dim,
      indexs_ptr);

  // 2. get the address of each non-zero values
  const T* x_values_ptr = x_values.data<T>();
  const int64_t stride =
      x.dims().size() == sparse_dim ? 1 : x.non_zero_elements().dims()[1];
  DenseTensor values_indexs = phi::Empty(
      dev_ctx, DenseTensorMeta(DataType::INT32, {nnz}, DataLayout::NCHW));
  int* values_indexs_ptr = values_indexs.data<int>();
  DenseTensor public_indexs = phi::EmptyLike<int>(dev_ctx, values_indexs);

  // values_indexs = [0,1,2,,,nnz-1]
  phi::IndexKernel<int, kps::IdentityFunctor<int>>(
      dev_ctx, &values_indexs, kps::IdentityFunctor<int>());
  phi::IndexKernel<int, kps::IdentityFunctor<int>>(
      dev_ctx, &public_indexs, kps::IdentityFunctor<int>());

// 3. sort (indices, values index)
#ifdef PADDLE_WITH_HIP
  thrust::sort_by_key(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::sort_by_key(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                      indexs_ptr,
                      indexs_ptr + nnz,
                      values_indexs_ptr);

  // 4. unique index
  thrust::pair<IntT*, int*> new_end =
#ifdef PADDLE_WITH_HIP
      thrust::unique_by_key(thrust::hip::par.on(dev_ctx.stream()),
#else
      thrust::unique_by_key(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                            indexs_ptr,
                            indexs_ptr + nnz,
                            public_indexs.data<int>());

  phi::funcs::sparse::DistanceKernel<<<1, 1, 0, dev_ctx.stream()>>>(
      indexs_ptr, new_end.first, out_indices.data<IntT>());

  IntT out_nnz = 0;
  phi::backends::gpu::GpuMemcpyAsync(&out_nnz,
                                     out_indices.data<IntT>(),
                                     sizeof(IntT),
#ifdef PADDLE_WITH_HIP
                                     hipMemcpyDeviceToHost,
#else
                                     cudaMemcpyDeviceToHost,
#endif
                                     dev_ctx.stream());
  dev_ctx.Wait();

  out_indices.Resize({x_indices.dims()[0], out_nnz});
  if (out_values.dims().size() == 1) {
    out_values.Resize(phi::make_ddim({out_nnz}));
  } else {
    out_values.Resize(phi::make_ddim({out_nnz, x_values.dims()[1]}));
  }

  // 5. scatter the values
  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, nnz * stride, 1);
  phi::funcs::sparse::ScatterKernel<T><<<config.block_per_grid,
                                         config.thread_per_block,
                                         0,
                                         dev_ctx.stream()>>>(
      x_values_ptr,
      public_indexs.data<int>(),
      values_indexs_ptr,
      out_nnz,
      nnz,
      stride,
      out_values.data<T>());

  // 6. convert index to coordinate
  Dim<DDim::kMaxRank> const_dims;
  for (int i = 0; i < x.dims().size(); i++) {
    const_dims[i] = x.dims()[i];
  }

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_nnz, 1);
  phi::funcs::sparse::IndexToCoordinateKernel<<<config.block_per_grid,
                                                config.thread_per_block,
                                                0,
                                                dev_ctx.stream()>>>(
      indexs_ptr, const_dims, out_nnz, sparse_dim, out_indices.data<IntT>());

  out->SetMember(out_indices, out_values, x.dims(), true);
}

template <typename T, typename Context>
void CoalescedKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     SparseCooTensor* out) {
  PD_VISIT_INTEGRAL_TYPES(
      x.non_zero_indices().dtype(), "CoalescedGPUKernel", ([&] {
        CoalescedGPUKernel<T, data_t>(dev_ctx, x, out);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sort,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CoalescedKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
