/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

namespace phi {
namespace funcs {
namespace sparse {

/* Given the indices of a sparse tensor, return a vector of offsets
for the entries in the equivalent dense tensor. */
template <typename IntT, typename Context>
inline DenseTensor GetOffsets(const Context& dev_ctx,
                              const DenseTensor& indices,
                              const std::vector<IntT>& sizes,
                              const IntT dim) {
#ifdef __HIPCC__
  const auto& policy = thrust::hip::par.on(dev_ctx.stream());
#else
  const auto& policy = thrust::cuda::par.on(dev_ctx.stream());
#endif

  auto ndim = indices.dims()[0];
  auto nnz = indices.dims()[1];
  std::vector<IntT> host_strides(ndim, 1);
  if (ndim > 1) {
    for (IntT i = ndim - 2; i >= 0; i--) {
      host_strides[i] = host_strides[i + 1] * (i + 1 == dim ? 1 : sizes[i + 1]);
    }
  }

  const IntArray strides_shape(phi::vectorize<IntT>(indices.dims()));
  DenseTensor strides = phi::Empty<IntT>(dev_ctx, strides_shape);
  auto strides_ptr = strides.data<IntT>();
  memory_utils::Copy(dev_ctx.GetPlace(),
                     strides_ptr,
                     phi::CPUPlace(),
                     host_strides.data(),
                     sizeof(IntT) * host_strides.size(),
                     dev_ctx.stream());

  DenseTensor offsets = phi::Empty<IntT>(dev_ctx, {nnz});
  auto indices_ptr = indices.data<IntT>();

  thrust::transform(
      policy,
      thrust::make_counting_iterator(IntT(0)),
      thrust::make_counting_iterator(IntT(nnz)),
      thrust::device_ptr<IntT>(offsets.data<IntT>()),
      [strides_ptr, indices_ptr, nnz, dim, ndim] __device__(IntT x) {
        IntT pool_index = 0;
        for (IntT j = 0; j < ndim; j++) {
          if (j != dim) {
            auto indice_cur_ptr = indices_ptr + j * nnz + x;
            auto stride = strides_ptr[j];
            pool_index += stride * (*indice_cur_ptr);
          }
        }
        return pool_index;
      });
  return offsets;
}

/* Return pools of indices that align with the given dimension and the
corresponding max values for each pool. */
template <typename T,
          typename IntT,
          typename Context,
          bool requireMxRows = true>
std::tuple<DenseTensor, DenseTensor, DenseTensor, DenseTensor> ComputePoolMax(
    const Context& dev_ctx,
    const DenseTensor& indices,
    const DenseTensor& values,
    const std::vector<IntT>& sizes,
    IntT nvalues,
    const IntT dim) {
#ifdef __HIPCC__
  const auto& policy = thrust::hip::par.on(dev_ctx.stream());
#else
  const auto& policy = thrust::cuda::par.on(dev_ctx.stream());
#endif
  using thrust_ptr = thrust::device_ptr<IntT>;
  auto nnz = indices.dims()[1];
  DenseTensor offsets = phi::funcs::sparse::GetOffsets<IntT, Context>(
      dev_ctx, indices, sizes, dim);
  auto offsets_ptr = offsets.data<IntT>();

  phi::DenseTensor sorted_indices = phi::Empty<IntT>(dev_ctx, {nnz});
  thrust_ptr sorted_indices_thrust_ptr(sorted_indices.data<IntT>());
  thrust::sequence(
      policy, sorted_indices_thrust_ptr, sorted_indices_thrust_ptr + nnz, 0);

  /* sort indices corresponding to offsets */
  thrust::sort(policy,
               sorted_indices_thrust_ptr,
               sorted_indices_thrust_ptr + nnz,
               [offsets_ptr] __device__(IntT x, IntT y) {
                 return offsets_ptr[x] < offsets_ptr[y];
               });

  DenseTensor pool_sizes = phi::Empty<IntT>(dev_ctx, {nnz});

  /* reduce the elements which are groupped by pool index,
  returns all the pool indexes with unique offset value for each. */
  auto new_end =
      thrust::reduce_by_key(policy,
                            sorted_indices_thrust_ptr,
                            sorted_indices_thrust_ptr + nnz,
                            thrust::make_constant_iterator(IntT(1)),
                            thrust::make_discard_iterator(),
                            thrust_ptr(pool_sizes.data<IntT>()),
                            [offsets_ptr] __device__(IntT x, IntT y) {
                              return offsets_ptr[x] == offsets_ptr[y];
                            });
  auto new_sz =
      thrust::distance(thrust_ptr(pool_sizes.data<IntT>()), new_end.second);
  pool_sizes.Resize(phi::make_ddim({new_sz}));

  DenseTensor pool_offsets;
  pool_offsets.Resize(phi::make_ddim({new_sz}));
  dev_ctx.template Alloc<T>(&pool_offsets);
  phi::Copy(dev_ctx, pool_sizes, dev_ctx.GetPlace(), false, &pool_offsets);

  /* accumulate value for each pool index */
  thrust_ptr pool_offsets_thrust_ptr(pool_offsets.data<IntT>());
  thrust::exclusive_scan(policy,
                         pool_offsets_thrust_ptr,
                         pool_offsets_thrust_ptr + new_sz,
                         pool_offsets_thrust_ptr);

  DenseTensor mx_buffer;
  if (requireMxRows) {
    mx_buffer = phi::Full<T>(
        dev_ctx, {new_sz * nvalues}, -std::numeric_limits<T>::infinity());
    auto mx_buffer_ptr = mx_buffer.data<T>();

    auto pool_sizes_ptr = pool_sizes.data<IntT>();
    auto sorted_indices_ptr = sorted_indices.data<IntT>();
    auto pool_offsets_ptr = pool_offsets.data<IntT>();
    auto values_ptr = values.data<T>();

    /* calculate max value in each pool. */
    thrust::for_each(policy,
                     thrust::make_counting_iterator(IntT(0)),
                     thrust::make_counting_iterator(IntT(new_sz)),
                     [sorted_indices_ptr,
                      pool_sizes_ptr,
                      pool_offsets_ptr,
                      mx_buffer_ptr,
                      values_ptr,
                      nvalues] __device__(IntT index) {
                       IntT curr_pool_size = pool_sizes_ptr[index];
                       auto mx_row = mx_buffer_ptr + index * nvalues;
                       IntT offset = pool_offsets_ptr[index];
                       for (IntT p = 0; p < curr_pool_size; p++) {
                         IntT i = *(sorted_indices_ptr + offset + p);
                         for (IntT j = 0; j < nvalues; j++) {
                           auto value_tmp = *(values_ptr);
                           mx_row[j] = std::max(mx_row[j], value_tmp);
                         }
                       }
                     });
  }
  return std::make_tuple(sorted_indices, pool_offsets, pool_sizes, mx_buffer);
}

inline int GetNumThreads(int nElem) {
#if defined(PADLDE_WITH_ROCM)
  int threadSizes[5] = {16, 32, 64, 128, 256};
#else
  int threadSizes[5] = {32, 64, 128, 256, 512};
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return threadSizes[4];
}

}  // namespace sparse
}  // namespace funcs
}  // namespace phi
