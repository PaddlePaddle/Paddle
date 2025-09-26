// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/phi/kernels/prod_kernel.h"
#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "paddle/phi/kernels/reduce_amax_kernel.h"
#include "paddle/phi/kernels/reduce_amin_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/reduce_min_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/phi/kernels/stride/reduce_stride_base.cu.h"

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename T, typename Context>
void AMaxStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel || x.offset() != 0) {
    if (!x.meta().is_contiguous() || x.offset() != 0) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }

  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::AMaxKernel<T, Context>(dev_ctx, x_, dims, keep_dim, out);
    return;
  }

  T ident = std::numeric_limits<T>::lowest();
  ReduceStrideImpl<T, Context, kps::MaxFunctor>(
      dev_ctx, x_, dims, keep_dim, ident, out);
  return;
}

template <typename T, typename Context>
void AMinStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel || x.offset() != 0) {
    if (!x.meta().is_contiguous() || x.offset() != 0) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }
  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::AMinKernel<T, Context>(dev_ctx, x_, dims, keep_dim, out);
    return;
  }

  T ident = std::numeric_limits<T>::max();
  ReduceStrideImpl<T, Context, kps::MinFunctor>(
      dev_ctx, x_, dims, keep_dim, ident, out);
  return;
}

template <typename T, typename Context>
void MaxStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }

  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::MaxKernel<T, Context>(dev_ctx, x_, dims, keep_dim, out);
    return;
  }

  T ident = std::numeric_limits<T>::lowest();
  ReduceStrideImpl<T, Context, kps::MaxFunctor>(
      dev_ctx, x_, dims.GetData(), keep_dim, ident, out);
  return;
}

template <typename T, typename Context>
void MinStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous() || x.offset() != 0) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }
  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::MinKernel<T, Context>(dev_ctx, x_, dims, keep_dim, out);
    return;
  }

  T ident = std::numeric_limits<T>::max();
  ReduceStrideImpl<T, Context, kps::MinFunctor>(
      dev_ctx, x_, dims.GetData(), keep_dim, ident, out);
  return;
}

template <typename T, typename Context>
void ProdStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const IntArray& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }
  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::ProdKernel<T, Context>(dev_ctx, x_, dims, keep_dim, reduce_all, out);
    return;
  }

  if (x_.numel() == 0) {
    // fill with 1.
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(out->dims())), 1, out);
    return;
  }

  T ident = T(1);
  ReduceStrideImpl<T, Context, kps::MulFunctor>(
      dev_ctx, x_, dims.GetData(), keep_dim, ident, out);
  return;
}

template <typename T, typename Context>
void AllStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int64_t>& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }
  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::AllKernel<T, Context>(dev_ctx, x_, dims, keep_dim, out);
    return;
  }

  if (x_.numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    if (out->numel() > 0) {
      std::vector<int64_t> vec_dims = common::vectorize(out->dims());
      phi::Full<bool, Context>(dev_ctx, phi::IntArray(vec_dims), 0, out);
    }
    return;
  }

  auto out_dtype = phi::DataType::BOOL;
  if (out_dtype != phi::DataType::UNDEFINED && out_dtype != x_.dtype()) {
    auto tmp_tensor = phi::Cast<T>(dev_ctx, x, out_dtype);
    PD_VISIT_BOOL_AND_FLOATING_AND_COMPLEX_AND_4_TYPES(
        phi::DataType::INT32,
        phi::DataType::INT64,
        phi::DataType::FLOAT16,
        phi::DataType::BFLOAT16,
        out_dtype,
        "ReduceStrideImpl",
        ([&] {
          data_t ident = data_t(1);
          ReduceStrideImpl<data_t, Context, kps::LogicalAndFunctor>(
              dev_ctx, tmp_tensor, dims, keep_dim, ident, out);
        }));
  } else {
    T ident = T(1);
    ReduceStrideImpl<T, Context, kps::LogicalAndFunctor>(
        dev_ctx, x_, dims, keep_dim, ident, out);
  }
  return;
}

template <typename T, typename Context>
void AnyStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int64_t>& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }
  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::AnyKernel<T, Context>(dev_ctx, x_, dims, keep_dim, out);
    return;
  }

  auto out_dtype = phi::DataType::BOOL;
  if (out_dtype != phi::DataType::UNDEFINED && out_dtype != x_.dtype()) {
    auto tmp_tensor = phi::Cast<T>(dev_ctx, x, out_dtype);
    PD_VISIT_BOOL_AND_FLOATING_AND_COMPLEX_AND_4_TYPES(
        phi::DataType::INT32,
        phi::DataType::INT64,
        phi::DataType::FLOAT16,
        phi::DataType::BFLOAT16,
        out_dtype,
        "ReduceStrideImpl",
        ([&] {
          data_t ident = static_cast<data_t>(0);
          ReduceStrideImpl<data_t, Context, kps::LogicalOrFunctor>(
              dev_ctx, tmp_tensor, dims, keep_dim, ident, out);
        }));
  } else {
    T ident = 0;
    ReduceStrideImpl<T, Context, kps::LogicalOrFunctor>(
        dev_ctx, x_, dims, keep_dim, ident, out);
  }
  return;
}

template <typename T, typename Context>
void SumStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& dims,
                     DataType out_dtype,
                     bool keep_dim,
                     DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }
  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::SumKernel<T, Context>(dev_ctx, x_, dims, out_dtype, keep_dim, out);
    return;
  }

  if (out_dtype == DataType::UNDEFINED && out->dtype() != x_.dtype()) {
    out_dtype = out->dtype();
  }
  if (x_.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    if (out_dtype == DataType::INT64) {
      FullKernel<int64_t, Context>(
          dev_ctx,
          phi::IntArray(common::vectorize(out->dims())),
          0,
          out_dtype,  // not used
          out);
    } else {
      FullKernel<T, Context>(dev_ctx,
                             phi::IntArray(common::vectorize(out->dims())),
                             0,
                             out_dtype,  // not used
                             out);
    }
    return;
  }

  if (x.dtype() == phi::DataType::BFLOAT16 &&
      out_dtype == phi::DataType::FLOAT32) {
    phi::dtype::bfloat16 ident = static_cast<phi::dtype::bfloat16>(0);
    ReduceStrideImpl<phi::dtype::bfloat16, Context, kps::AddFunctor>(
        dev_ctx, x_, dims.GetData(), keep_dim, ident, out);
    *out = phi::Cast<phi::dtype::bfloat16>(dev_ctx, x_, out_dtype);
  } else if (out_dtype != phi::DataType::UNDEFINED && out_dtype != x_.dtype()) {
    auto tmp_tensor = phi::Cast<T>(dev_ctx, x_, out_dtype);
    PD_VISIT_BOOL_AND_FLOATING_AND_COMPLEX_AND_4_TYPES(
        phi::DataType::INT32,
        phi::DataType::INT64,
        phi::DataType::FLOAT16,
        phi::DataType::BFLOAT16,
        out_dtype,
        "ReduceStrideImpl",
        ([&] {
          data_t ident = static_cast<data_t>(0);
          ReduceStrideImpl<data_t, Context, kps::AddFunctor>(
              dev_ctx, tmp_tensor, dims.GetData(), keep_dim, ident, out);
        }));
  } else {
    T ident = static_cast<T>(0);
    ReduceStrideImpl<T, Context, kps::AddFunctor>(
        dev_ctx, x_, dims.GetData(), keep_dim, ident, out);
  }
  return;
}

template <typename T, typename Context>
void MeanStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const IntArray& dims,
                      bool keep_dim,
                      DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }
  if (x_.meta().is_contiguous() || (out->dims().size() > 0)) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::MeanKernel<T, Context>(dev_ctx, x_, dims, keep_dim, out);
    return;
  }

  if (x_.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(out->dims())), NAN, out);
    return;
  }

  if (std::is_same<T, int>::value || std::is_same<T, int64_t>::value ||
      std::is_same<T, bool>::value) {
    using Type =
        typename std::conditional<std::is_same<T, int>::value ||
                                      std::is_same<T, int64_t>::value ||
                                      std::is_same<T, bool>::value,
                                  float,
                                  T>::type;
    DenseTensor x_float =
        phi::Cast<T, Context>(dev_ctx, x_, phi::DataType::FLOAT32);
    DenseTensor* out_float = new DenseTensor();
    out_float->Resize(out->dims());
    MeanRawKernel<Type>(
        dev_ctx, x_float, dims, keep_dim, reduce_all, out_float);

    Type ident = static_cast<Type>(0);
    ReduceStrideImpl<Type, Context, kps::AddFunctor, true>(
        dev_ctx, x_float, dims.GetData(), keep_dim, ident, out_float);

    phi::CastKernel<Type, Context>(dev_ctx, *out_float, x_.dtype(), out);
  } else {
    T ident = static_cast<T>(0);
    ReduceStrideImpl<T, Context, kps::AddFunctor, true>(
        dev_ctx, x_, dims.GetData(), keep_dim, ident, out);
  }
  return;
}

}  // namespace phi

using float16 = phi::float16;
using bfloat16 = phi::bfloat16;
using complex64 = ::phi::complex64;
using complex128 = ::phi::complex128;

PD_REGISTER_KERNEL(
    amax, GPU, STRIDED, phi::AMaxStrideKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    amin, GPU, STRIDED, phi::AMinStrideKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    max, GPU, STRIDED, phi::MaxStrideKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    min, GPU, STRIDED, phi::MinStrideKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(prod,
                   GPU,
                   STRIDED,
                   phi::ProdStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_REGISTER_KERNEL(any,
                   GPU,
                   STRIDED,
                   phi::AnyStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(all,
                   GPU,
                   STRIDED,
                   phi::AllStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(sum,
                   GPU,
                   STRIDED,
                   phi::SumStrideKernel,
                   bool,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(mean,
                   GPU,
                   STRIDED,
                   phi::MeanStrideKernel,
                   float,
                   double,
                   bool,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::complex64,
                   phi::complex128) {}

#endif
