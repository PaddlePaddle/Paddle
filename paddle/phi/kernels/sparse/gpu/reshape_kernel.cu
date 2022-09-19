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

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"
// #include "paddle/phi/kernels/sparse/gpu/sparse_utils_kernel.cu"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"


namespace phi {
namespace sparse {

__global__ void ReshapeCooCudaKernel(const int64_t *x_indices_data,
                                       const int& num_x_sparse_part_dims,
                                      //  const int *perm,
                                       const int& num_out_sparse_part_dims,
                                       const int64_t& x_nnz,
                                       const int64_t* x_sparse_part_strides,
                                       const int64_t* out_sparse_part_strides,
                                       int64_t *out_indices_data) {
                                        
  // for (std::size_t i = 0; i < n_dim; ++i) {
  //   CUDA_KERNEL_LOOP_TYPE(j, x_nnz, int64_t) {
  //     out_indices_data[j + i * x_nnz] = x_indices_data[j + perm[i] * x_nnz];
  //   }
  // }
  
    CUDA_KERNEL_LOOP_TYPE(j, x_nnz, int64_t) {
        int64_t location = 0;
        for (int i = 0; i < num_x_sparse_part_dims; ++i) {
            location += x_indices_data[i * x_nnz + j] * x_sparse_part_strides[i];
        }
        for (int i = 0; i < num_out_sparse_part_dims; ++i) {
            out_indices_data[i * x_nnz + j] = location / out_sparse_part_strides[i];
            location %= out_sparse_part_strides[i];
        }
    }
}



template <typename T, typename Context>
void ReshapeCooKernel(const Context &dev_ctx,
                        const SparseCooTensor &x,
                        const std::vector<int64_t>& new_shape,
                        SparseCooTensor *out) {
  // create "out" sparse tensor
  int64_t x_nnz = x.nnz();
  // DDim out_dims = x.dims().transpose(perm);
  DDim out_dims = phi::make_ddim(new_shape);

    ///////  get sparse part dimensions of x and out
  std::vector<int64_t> x_sparse_part_dims;
  std::vector<int64_t> out_sparse_part_dims;
  for (int i = 0; i < x.sparse_dim(); ++i) {
    x_sparse_part_dims.push_back(x.dims()[i]);
  }
  for (int i = 0; i < out_dims.size() - x.dense_dim(); ++i) {
    out_sparse_part_dims.push_back(out_dims[i]);
  }
  // DenseTensor out_indices = EmptyLike<int64_t, Context>(dev_ctx, x.indices());
  DenseTensor out_indices = Empty<int64_t, Context>(dev_ctx, 
            {static_cast<int64_t>(out_sparse_part_dims.size()), x_nnz}
  );
  
  DenseTensor out_values(x.values());

  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  // compute values of indices
  const DenseTensor& x_indices = x.indices();
  const auto *x_indices_data = x_indices.data<int64_t>();
  auto *out_indices_data = out_indices.data<int64_t>();





//////////////
//   int *d_perm;
// #ifdef PADDLE_WITH_HIP
//   hipMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
//   hipMemcpy(
//       d_perm, perm.data(), sizeof(int) * perm.size(), hipMemcpyHostToDevice);
// #else
//   cudaMalloc(reinterpret_cast<void **>(&d_perm), sizeof(int) * perm.size());
//   cudaMemcpy(
//       d_perm, perm.data(), sizeof(int) * perm.size(), cudaMemcpyHostToDevice);
// #endif
//////////


    const phi::DDim& x_sparse_part_strides = phi::stride(phi::make_ddim(x_sparse_part_dims));
    const phi::DDim& out_sparse_part_strides = phi::stride(phi::make_ddim(out_sparse_part_dims));

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, x_nnz, 1);
  ReshapeCooCudaKernel<<<config.block_per_grid.x,
                           config.thread_per_block.x,
                           0,
                           dev_ctx.stream()>>>(
      // x_indices_data, d_perm, perm.size(), x_nnz, out_indices_data
      x_indices_data, 
      x_sparse_part_dims.size(), out_sparse_part_dims.size(), x_nnz,
      x_sparse_part_strides.Get(), out_sparse_part_strides.Get(),
      out_indices_data
      );
}


// just copy from paddle\phi\kernels\sparse\cpu\reshape_kernel.cc
template <typename T, typename Context>
void ReshapeCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const std::vector<int64_t>& new_shape,
                        SparseCsrTensor* out) {
 /*将csr格式转化为coo格式后处理*/
const SparseCooTensor x_coo = CsrToCoo<T, Context>(dev_ctx, x);
SparseCooTensor out_coo;
ReshapeCooKernel<T, Context>(dev_ctx, x_coo, new_shape, &out_coo);
CooToCsrKernel<T, Context>(dev_ctx, out_coo, out);     
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(reshape_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCooKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(reshape_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCsrKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}