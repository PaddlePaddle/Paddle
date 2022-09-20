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

// #include "paddle/phi/core/ddim.cc"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"
// #include "paddle/phi/kernels/sparse/cpu/sparse_utils_kernel.cc"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void ReshapeCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        // const std::vector<int>& sparse_part_permutation,
                        const std::vector<int64_t>& new_shape,
                        SparseCooTensor* out) {
   /*
   目前只能针对 sparse part dims 部分进行reshape
   */                       
  // create "out" sparse tensor
  int64_t x_nnz = x.nnz();
  // DDim out_dims = x.dims().transpose(perm);
  DDim out_dims = phi::make_ddim(new_shape);
  ////get sparse part dimensions of x and out
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
  const auto* x_indices_data = x_indices.data<int64_t>();
  auto* out_indices_data = out_indices.data<int64_t>();
//   //i 表示 indices 的 行标
//   for (unsigned int i = 0; i < perm.size(); ++i) {
//     // j 表示 indices 的 列标
//     for (int64_t j = 0; j < x_nnz; ++j) {
//         // 修改 out indices 的索引为 (i, j)的元素值
//     /* Caution : 这是原来的计算逻辑，我认为是 错误的，
//         这里计算逻辑是： 原tensor的shape是  (10, 20, 30, 40, 50)
//         一个非零元素的索引为 (1, 2, 3, 4, 5)
//         进行transpose 后, tensor的shape 是 (30, 10, 50, 20, 40)
//         这里的计算逻辑就认为该非零元素的新索引就是 (3, 1, 5, 2, 4)
//       没错，这就是transpose的计算逻辑，transpose后元素在内存中的位置改变了
//     你更改的逻辑其实是 reshape的计算逻辑，reshape后所有元素在内存中的位置均不变
//     */
//      out_indices_data[j + i * x_nnz] = x_indices_data[j + perm[i] * x_nnz];
//     }
//   }

    // 我的更改后的计算逻辑如下：
    const phi::DDim& x_sparse_part_strides = phi::stride(phi::make_ddim(x_sparse_part_dims));
    const phi::DDim& out_sparse_part_strides = phi::stride(phi::make_ddim(out_sparse_part_dims));
    int64_t location = 0;

    for (int64_t j = 0; j < x_nnz; ++j) {
        location = 0;
        for (int i = 0; i < x.sparse_dim(); ++i) {
            location += x_indices_data[i * x_nnz + j] * x_sparse_part_strides[i];
        }
        for (size_t i = 0; i < out_sparse_part_dims.size(); ++i) {
            out_indices_data[i * x_nnz + j] = location / out_sparse_part_strides[i];
            location %= out_sparse_part_strides[i];
        }
    }

}


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
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(reshape_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ReshapeCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}