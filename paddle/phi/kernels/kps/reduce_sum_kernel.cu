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

#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include <limits>
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#ifndef PADDLE_WITH_XPU_KP
#include "paddle/phi/kernels/funcs/eigen/common.h"
#endif

namespace phi {

#ifndef PADDLE_WITH_XPU_KP
template <typename T,
          int EigenDimSize = 5,
          int ReducedDimSize = 1,
          bool ReduceAll = false>
void ReduceSumEigen(const KPDevice& dev_ctx,
                    const DenseTensor& x,
                    bool reduce_all,
                    const std::vector<int64_t>& dims,
                    DataType out_dtype,
                    DenseTensor* out,
                    std::vector<int>* reduce_dims) {
  // Resize Input Tensor
  auto new_x = x;
  int added_dims = EigenDimSize - x.dims().size();
  std::array<int64_t, EigenDimSize> new_x_dim;
  new_x_dim.fill(1);
  for (int i = 0; i < x.dims().size(); i++) {
    new_x_dim[i + added_dims] = x.dims().at(i);
  }
  new_x.Resize(phi::DDim(new_x_dim.data(), new_x_dim.size()));
  auto eigen_x_tensor = EigenTensor<T, EigenDimSize>::From(new_x);

  // Create Out Tensor
  dev_ctx.Alloc<T>(out);
  auto origin_out_dims = out->dims();
  constexpr int kReduceOutRank = ReduceAll ? 1 : EigenDimSize - ReducedDimSize;
  // Resize Out Tensor
  std::array<int64_t, kReduceOutRank> new_out_dim;
  new_out_dim.fill(1);
  for (int i = 0; i < out->dims().size(); i++) {
    new_out_dim[i + added_dims] = out->dims().at(i);
  }
  out->Resize(phi::DDim(new_out_dim.data(), new_out_dim.size()));

  auto eigen_out_tensor = EigenTensor<T, kReduceOutRank>::From(*out);
  for (int i = 0; i < ReducedDimSize; i++) {
    (*reduce_dims)[i] += added_dims;
  }
  auto eigen_reduce_dim =
      EigenDim<ReducedDimSize>::From(phi::make_ddim(*reduce_dims));
  // Caculate
  eigen_out_tensor.device(*dev_ctx.eigen_device()) =
      eigen_x_tensor.sum(eigen_reduce_dim);
  out->Resize(origin_out_dims);
}
#endif

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DataType out_dtype,
                  DenseTensor* out) {
  if (out_dtype == DataType::UNDEFINED && out->dtype() != x.dtype()) {
    out_dtype = out->dtype();
  }
  if (x.numel() > std::numeric_limits<int32_t>::max()) {
#ifndef PADDLE_WITH_XPU_KP
    if (out_dtype != phi::DataType::UNDEFINED && out_dtype != x.dtype()) {
      PADDLE_THROW(phi::errors::Fatal(
          "If Input.numel() > INT32_MAX, reduce_sum kernel uses EigenTensor "
          "sum for reduce_sum function. As a result, input dtype should be "
          "the same as out dtype"));
    }

    std::vector<int> reduce_dims = phi::funcs::details::GetReduceDim(
        dims.GetData(), x.dims().size(), reduce_all);

#define CALL_EIGEN_REDUCE_SUM_KERNEL(reduce_rank)              \
  case reduce_rank: {                                          \
    if (reduce_all) {                                          \
      ReduceSumEigen<T, 5, reduce_rank, true>(dev_ctx,         \
                                              x,               \
                                              reduce_all,      \
                                              dims.GetData(),  \
                                              out_dtype,       \
                                              out,             \
                                              &reduce_dims);   \
    } else {                                                   \
      ReduceSumEigen<T, 5, reduce_rank, false>(dev_ctx,        \
                                               x,              \
                                               reduce_all,     \
                                               dims.GetData(), \
                                               out_dtype,      \
                                               out,            \
                                               &reduce_dims);  \
    }                                                          \
    break;                                                     \
  }

    switch (reduce_dims.size()) {
      CALL_EIGEN_REDUCE_SUM_KERNEL(1);
      CALL_EIGEN_REDUCE_SUM_KERNEL(2);
      CALL_EIGEN_REDUCE_SUM_KERNEL(3);
      CALL_EIGEN_REDUCE_SUM_KERNEL(4);
      CALL_EIGEN_REDUCE_SUM_KERNEL(5);
      default:
        PADDLE_THROW(phi::errors::Fatal(
            "If Input.numel() > INT32_MAX, reduce_sum kernel uses EigenTensor "
            "sum for reduce_sum function. As a result, its dim should be <= "
            "5."));
        break;
    }
#undef CALL_EIGEN_REDUCE_SUM_KERNEL
#else
    PADDLE_THROW(phi::errors::Fatal(
        "If Input.numel() > INT32_MAX, reduce_sum kernel uses EigenTensor "
        "sum for reduce_sum function. Such case is only supported on GPU "
        "now."));
#endif
  } else {
    phi::Reduce<T, kps::AddFunctor, kps::IdentityFunctor>(
        dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
  }
}
}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(sum_raw, KPS, ALL_LAYOUT, phi::SumRawKernel, float) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
#else
using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(sum_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::SumRawKernel,
                   bool,
                   float,
                   double,
                   float16,
                   bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
#endif
