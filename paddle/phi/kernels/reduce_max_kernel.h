#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out);

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                DenseTensor* out);

template <typename T, typename Context>
DenseTensor Max(const Context& dev_ctx,
                 const DenseTensor& x,
                 const std::vector<int64_t>& axis,
                 bool keep_dim) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  ReduceInferMetaBase(x, axis, keep_dim, false, x.dtype(), &meta_out);
  MeanKernel<T, Context>(dev_ctx, x, axis, keep_dim, &dense_out);
  return dense_out;
}
} // namespace phi
