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

#include "paddle/phi/kernels/sparse/coalesced_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/sparse/flatten_indices.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT>
void CoalescedCPUKernel(const CPUContext& dev_ctx,
                        const SparseCooTensor& x,
                        SparseCooTensor* out) {
  const DenseTensor& x_indices = x.non_zero_indices();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor out_indices = phi::EmptyLike<IntT>(dev_ctx, x_indices);
  DenseTensor out_values = phi::EmptyLike<T>(dev_ctx, x_values);

  const int64_t sparse_dim = x.non_zero_indices().dims()[0];
  std::vector<IntT> sparse_offsets(sparse_dim), x_indexs(x.nnz());
  phi::funcs::sparse::CalcOffsetsPerDim<IntT>(
      x.dims(), sparse_dim, sparse_offsets.data());

  phi::funcs::sparse::FlattenIndices(x.non_zero_indices().data<IntT>(),
                                     sparse_offsets.data(),
                                     x.nnz(),
                                     sparse_dim,
                                     0,
                                     1,
                                     x_indexs.data());

  const T* x_values_ptr = x_values.data<T>();
  const int64_t stride =
      x.dims().size() == sparse_dim ? 1 : x.non_zero_elements().dims()[1];

  std::map<IntT, std::vector<int64_t>> indices_to_index;
  for (uint64_t i = 0; i < x_indexs.size(); i++) {
    IntT index = x_indexs[i];
    if (indices_to_index.find(index) == indices_to_index.end()) {
      std::vector<int64_t> indexs;
      indexs.push_back(i);
      indices_to_index[index] = indexs;
    } else {
      indices_to_index[index].push_back(i);
    }
  }

  const int64_t out_nnz = indices_to_index.size();

  out_indices.Resize({x_indices.dims()[0], out_nnz});
  if (out_values.dims().size() == 1) {
    out_values.Resize(phi::make_ddim({out_nnz}));
  } else {
    out_values.Resize(phi::make_ddim({out_nnz, x_values.dims()[1]}));
  }

  IntT* out_indices_ptr = out_indices.data<IntT>();
  T* out_values_ptr = out_values.data<T>();
  auto iter = indices_to_index.begin();

  Dim<DDim::kMaxRank> const_dims;
  for (int i = 0; i < x.dims().size(); i++) {
    const_dims[i] = x.dims()[i];
  }

  for (int i = 0; iter != indices_to_index.end(); iter++, i++) {
    phi::funcs::sparse::IndexToCoordinate(
        iter->first, const_dims, out_nnz, sparse_dim, i, out_indices_ptr);
    memcpy(out_values_ptr + i * stride,
           x_values_ptr + iter->second[0] * stride,
           stride * sizeof(T));
    for (uint64_t j = 1; j < iter->second.size(); j++) {
      for (int k = 0; k < stride; k++) {
        out_values_ptr[i * stride + k] +=
            x_values_ptr[iter->second[j] * stride + k];
      }
    }
  }

  out->SetMember(out_indices, out_values, x.dims(), true);
}

template <typename T, typename Context>
void CoalescedKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     SparseCooTensor* out) {
  PD_VISIT_INTEGRAL_TYPES(
      x.non_zero_indices().dtype(), "CoalescedCPUKernel", ([&] {
        CoalescedCPUKernel<T, data_t>(dev_ctx, x, out);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sort,
                   CPU,
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
