// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
#include "paddle/phi/kernels/funcs/fused_gemm_epilogue.h"
#endif
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {
namespace fusion {

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060

template <typename T, typename MT, typename Context>
void FusedLinearParamGradAddImpl(const Context &ctx,
                                 const DenseTensor &x,
                                 const DenseTensor &dout,
                                 const paddle::optional<DenseTensor> &dbias,
                                 int64_t M,
                                 int64_t K,
                                 int64_t N,
                                 bool use_addto,
                                 DenseTensor *dweight_out,
                                 DenseTensor *dbias_out) {
  constexpr bool kIsMultiPrecision = !std::is_same<T, MT>::value;

  const bool fuse_bias_grad = kIsMultiPrecision && dweight_out;
  if (dweight_out) {
    phi::funcs::ComputeFusedGemmEpilogueBackward<T, T, MT>(
        ctx,
        &dout,
        &x,
        nullptr,
        nullptr,
        M,
        N,
        K,
        false,
        false,
        "none",
        nullptr,
        dweight_out,
        fuse_bias_grad ? dbias_out : nullptr,
        false,
        use_addto);
  }

  if (dbias_out == nullptr) return;

  if (!fuse_bias_grad) {
    auto dout_copy = dout;
    dout_copy.Resize({M, N});
    if (kIsMultiPrecision) {
      *dbias_out = phi::Sum<T, Context>(
          ctx, dout_copy, {0}, phi::CppTypeToDataType<MT>::Type(), false);
    } else {
      *dbias_out = phi::Sum<T, Context>(
          ctx, dout_copy, {0}, phi::CppTypeToDataType<T>::Type(), false);
    }
  }

  if (dbias) {
    if (kIsMultiPrecision) {
      phi::AddKernel<MT, Context>(ctx, *dbias_out, dbias.get(), dbias_out);
    } else {
      phi::AddKernel<T, Context>(ctx, *dbias_out, dbias.get(), dbias_out);
    }
  }
}

template <int LogLevel = 10>
static void PrintMeta(const DenseTensor &t, const char *name) {
  PADDLE_ENFORCE_EQ(
      t.initialized(),
      true,
      phi::errors::InvalidArgument("Tensor(%s) is not initialized.", name));
  std::stringstream ss;
  ss << "Tensor(" << name << "): ";
  ss << "dtype(" << t.dtype() << "), ";
  ss << "shape(" << t.dims() << "), ";
  ss << "place(" << t.place() << "), ";
  ss << "ptr(" << t.data() << ")";
  VLOG(LogLevel) << ss.str();
}

template <int LogLevel = 10>
static void PrintMeta(const DenseTensor *t, const char *name) {
  if (t == nullptr) {
    VLOG(LogLevel) << "Tensor(" << name << "): None";
  } else {
    PrintMeta<LogLevel>(*t, name);
  }
}

template <int LogLevel = 10>
static void PrintMeta(const paddle::optional<DenseTensor> &t,
                      const char *name) {
  const auto *t_ptr = t ? &(t.get()) : nullptr;
  PrintMeta<LogLevel>(t_ptr, name);
}

template <typename T, typename Context>
void FusedLinearParamGradAdd(const Context &ctx,
                             const DenseTensor &x,
                             const DenseTensor &dout,
                             const paddle::optional<DenseTensor> &dweight,
                             const paddle::optional<DenseTensor> &dbias,
                             bool multi_precision,
                             DenseTensor *dweight_out,
                             DenseTensor *dbias_out) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  bool use_addto = false;
  if (dweight_out) {
    if (dweight) {
      use_addto = true;
      *dweight_out = dweight.get();
      if (multi_precision) {
        PADDLE_ENFORCE_EQ(
            dweight_out->dtype(),
            phi::CppTypeToDataType<MT>::Type(),
            phi::errors::InvalidArgument("Invaid data type error."));
      } else {
        PADDLE_ENFORCE_EQ(
            dweight_out->dtype(),
            phi::CppTypeToDataType<T>::Type(),
            phi::errors::InvalidArgument("Invaid data type error."));
      }
    } else {
      if (multi_precision) {
        ctx.template Alloc<MT>(dweight_out);
      } else {
        ctx.template Alloc<T>(dweight_out);
      }
    }
  }

  if (std::is_same<T, MT>::value) {
    multi_precision = false;
  }

  if (dbias_out) {
    ctx.template Alloc<T>(dbias_out);
  }

  int64_t K = x.dims()[x.dims().size() - 1];
  int64_t M = x.numel() / K;
  int64_t N = dout.dims()[dout.dims().size() - 1];

  constexpr int kLogLevel = 10;
  if (VLOG_IS_ON(kLogLevel)) {
    PrintMeta<kLogLevel>(x, "x");
    PrintMeta<kLogLevel>(dout, "dout");
    PrintMeta<kLogLevel>(dweight, "dweight");
    PrintMeta<kLogLevel>(dbias, "dbias");
    PrintMeta<kLogLevel>(dweight_out, "dweight_out");
    PrintMeta<kLogLevel>(dbias_out, "dbias_out");
    VLOG(kLogLevel) << "multi_precision = " << multi_precision;
    VLOG(kLogLevel) << "use_addto = " << use_addto;
    VLOG(kLogLevel) << "M = " << M;
    VLOG(kLogLevel) << "N = " << N;
    VLOG(kLogLevel) << "K = " << K;
  }

  if (multi_precision) {
    FusedLinearParamGradAddImpl<T, MT, Context>(
        ctx, x, dout, dbias, M, K, N, use_addto, dweight_out, dbias_out);
  } else {
    FusedLinearParamGradAddImpl<T, T, Context>(
        ctx, x, dout, dbias, M, K, N, use_addto, dweight_out, dbias_out);
  }
}

#else
template <typename T, typename Context>
void FusedLinearParamGradAdd(const Context &ctx,
                             const DenseTensor &x,
                             const DenseTensor &dout,
                             const paddle::optional<DenseTensor> &dweight,
                             const paddle::optional<DenseTensor> &dbias,
                             bool multi_precision,
                             DenseTensor *dweight_out,
                             DenseTensor *dbias_out) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "FusedLinearParamGradAdd is only supported when CUDA_VERSION >= 11.6."));
}
#endif

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_linear_param_grad_add,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedLinearParamGradAdd,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
