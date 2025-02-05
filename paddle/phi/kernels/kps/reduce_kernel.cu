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

#include <limits>
#include <set>
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/legacy/reduce_max_kernel.h"
#include "paddle/phi/kernels/prod_kernel.h"
#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "paddle/phi/kernels/reduce_amin_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/reduce_min_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#ifndef PADDLE_WITH_XPU_KP
#include "paddle/phi/kernels/funcs/eigen/common.h"
#endif

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

namespace phi {

template <typename T, typename Context>
void ProdKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& dims,
                bool keep_dim,
                bool reduce_all,
                DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::MulFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void AllRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = phi::DataType::BOOL;
  phi::Reduce<T, kps::LogicalAndFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void AMaxRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::MaxFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void AMinRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::MinFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void AnyRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = phi::DataType::BOOL;
  phi::Reduce<T, kps::LogicalOrFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& dims,
               bool keep_dim,
               DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  phi::MaxRawKernel<T, Context>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MeanRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::AddFunctor, kps::IdentityFunctor, true>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::MinFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

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
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
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
      EigenDim<ReducedDimSize>::From(common::make_ddim(*reduce_dims));
  // Calculate
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
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  if (out_dtype == DataType::UNDEFINED && out->dtype() != x.dtype()) {
    out_dtype = out->dtype();
  }
  if (x.numel() == 0) {
    auto x_dims = x.dims();
    std::vector<int> out_dims;
    if (reduce_all) {
      if (keep_dim) {
        out_dims.resize(x_dims.size(), 1);
      } else {
        out_dims = std::vector<int>();
      }
    } else {
      std::set<int> reduce_dims;
      auto dims_vec = dims.GetData();
      for (auto dim : dims_vec) {
        PADDLE_ENFORCE_GE(dim,
                          -x_dims.size(),
                          common::errors::InvalidArgument(
                              "The dimension index is out of range, "
                              "expected index >= %d, but received %d.",
                              -x_dims.size(),
                              dim));
        PADDLE_ENFORCE_LT(dim,
                          x_dims.size(),
                          common::errors::InvalidArgument(
                              "The dimension index is out of range, "
                              "expected index < %d, but received %d.",
                              x_dims.size(),
                              dim));
        if (dim < 0) {
          dim += x_dims.size();
        }
        reduce_dims.insert(dim);
      }
      if (keep_dim) {
        out_dims.resize(x_dims.size());
        for (int i = 0; i < x_dims.size(); ++i) {
          if (reduce_dims.count(i)) {
            out_dims[i] = 1;
          } else {
            out_dims[i] = x_dims[i];
          }
        }
      } else {
        for (int i = 0; i < x_dims.size(); ++i) {
          if (!reduce_dims.count(i)) {
            out_dims.push_back(x_dims[i]);
          }
        }
      }
    }
    out->Resize(phi::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(out);
    FullKernel<T, Context>(
        dev_ctx, out_dims, 0, phi::CppTypeToDataType<T>::Type(), out);
    return;
  }
  if (x.numel() > std::numeric_limits<int32_t>::max()) {
#ifndef PADDLE_WITH_XPU_KP
    if (out_dtype != phi::DataType::UNDEFINED && out_dtype != x.dtype()) {
      PADDLE_THROW(common::errors::Fatal(
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
        PADDLE_THROW(common::errors::Fatal(
            "If Input.numel() > INT32_MAX, reduce_sum kernel uses EigenTensor "
            "sum for reduce_sum function. As a result, its dim should be <= "
            "5."));
        break;
    }
#undef CALL_EIGEN_REDUCE_SUM_KERNEL
#else
    PADDLE_THROW(common::errors::Fatal(
        "If Input.numel() > INT32_MAX, reduce_sum kernel uses EigenTensor "
        "sum for reduce_sum function. Such case is only supported on GPU "
        "now."));
#endif
  } else {
    if (x.dtype() == phi::DataType::BFLOAT16 &&
        out_dtype == phi::DataType::FLOAT32) {
      std::vector<int> reduce_dims = phi::funcs::details::GetReduceDim(
          dims.GetData(), x.dims().size(), reduce_all);

      phi::funcs::ReduceKernel<
          phi::dtype::bfloat16,
          float,
          kps::AddFunctor,
          kps::IdentityFunctor<phi::dtype::bfloat16, float>>(
          dev_ctx,
          x,
          out,
          kps::IdentityFunctor<phi::dtype::bfloat16, float>(),
          reduce_dims);
    } else {
      phi::Reduce<T, kps::AddFunctor, kps::IdentityFunctor>(
          dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
    }
  }
}
}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(all_raw, KPS, ALL_LAYOUT, phi::AllRawKernel, bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(amax_raw, KPS, ALL_LAYOUT, phi::AMaxRawKernel, float) {}

PD_REGISTER_KERNEL(prod, KPS, ALL_LAYOUT, phi::ProdKernel, float) {}

PD_REGISTER_KERNEL(amin_raw, KPS, ALL_LAYOUT, phi::AMinRawKernel, float) {}

PD_REGISTER_KERNEL(any_raw, KPS, ALL_LAYOUT, phi::AnyRawKernel, bool) {}

PD_REGISTER_KERNEL(max, KPS, ALL_LAYOUT, phi::MaxKernel, float) {}

PD_REGISTER_KERNEL(mean_raw, KPS, ALL_LAYOUT, phi::MeanRawKernel, float) {}

PD_REGISTER_KERNEL(min_raw, KPS, ALL_LAYOUT, phi::MinRawKernel, float) {}

PD_REGISTER_KERNEL(sum_raw, KPS, ALL_LAYOUT, phi::SumRawKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
#else
using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(all_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::AllRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(amax_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::AMaxRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(amin_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::AMinRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(any_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::AnyRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(max,
                   KPS,
                   ALL_LAYOUT,
                   phi::MaxKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2) {}

PD_REGISTER_KERNEL(mean_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::MeanRawKernel,
                   float,
                   double,
                   bool,
                   phi::dtype::bfloat16,
                   float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(min_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::MinRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(sum_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::SumRawKernel,
                   bool,
                   float,
                   double,
                   float16,
                   bfloat16,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(prod,
                   KPS,
                   ALL_LAYOUT,
                   phi::ProdKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
