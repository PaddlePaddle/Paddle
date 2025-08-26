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

#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/selu_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename Functor,
          typename OutT,
          int Arity,
          int NumOuts,
          int VecSize,
          int vt>
__global__ void UnaryElementwiseKernel(
    Array<const _ptr_ char *__restrict__, Arity> ins,
    Array<_ptr_ OutT *, NumOuts> outs,
    uint32_t numel,
    int read_lens,
    Functor func,
    funcs::OffsetCalculator<Arity + NumOuts> offset_calc) {
  int64_t tid = THREAD_ID_X;
  int64_t nv = BLOCK_NUM_X * vt;
  int64_t idx = nv * BLOCK_ID_X + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < numel) {
      auto offsets = offset_calc.get(idx);
      using Traits = phi::funcs::FunctionTraits<Functor>;
      using ArgsT = typename Traits::ArgsTuple;
      __simd__ ArgsT args[VecSize];
      __simd__ ConditionalT<OutT, NumOuts> result[VecSize];
      std::get<0>(args[idx]) =
          *(reinterpret_cast<const _ptr_ std::tuple_element_t<0, ArgsT> *>(
              reinterpret_cast<const _ptr_ char *>(ins[0]) + offsets[1]));
      funcs::SameDimsElementwisePrimitiveCaller<ConditionalT<OutT, NumOuts>,
                                                VecSize,
                                                Functor,
                                                ArgsT,
                                                Arity>()(
          func, args, result, read_lens);
      char *out_ptr = reinterpret_cast<char *>(outs[0]) + offsets[0];
      *reinterpret_cast<OutT *>(out_ptr) =
          *reinterpret_cast<const OutT *>(&(result[0]));
      idx += BLOCK_NUM_X;
    }
  }
}

template <typename OutT, typename Context, typename Functor, int NumOuts = 1>
void UnaryStrideElementwiseKernel(const Context &dev_ctx,
                                  const std::vector<const DenseTensor *> &ins,
                                  std::vector<DenseTensor *> *outs,
                                  Functor func) {
  using Traits = phi::funcs::FunctionTraits<Functor>;
  const int Arity = Traits::arity;
  bool have_0_size = false;
  for (int i = 0; i < outs->size(); ++i) {
    if (outs->at(i)->numel() == 0) {
      have_0_size = true;
    }
    if (i > 0) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          common::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, "
              "but %dth output tensor`s shape is not.",
              i));
    }
    dev_ctx.template Alloc<OutT>((*outs)[i]);
  }
  if (have_0_size) {
    return;
  }
  int max_rank = 0;
  int min_rank = phi::DDim::kMaxRank;
  for (auto *in : ins) {
    max_rank = std::max(max_rank, in->dims().size());
    min_rank = std::min(min_rank, in->dims().size());
  }
  if (ins.size() == 1) {
    max_rank = std::max(max_rank, (*outs)[0]->dims().size());
  }
  int axis = max_rank - min_rank;
  auto classifier =
      funcs::BroadcastTypeClassifier<OutT, Functor, Arity, NumOuts>(
          ins, outs, axis);
  DenseTensorIteratorConfig config;
  config.add_output(*((*outs)[0]));
  config.add_const_input(*(ins[0]));
  DenseTensorIterator iter = config.build();
  const int &numel = iter.numel();
  funcs::OffsetCalculator offset_calc = funcs::make_offset_calculator<2>(iter);
  constexpr int unroll_factor = sizeof(OutT) >= 4 ? 2 : 4;
  auto stream = dev_ctx.stream();
  auto threads = 128;
  auto blocks = (numel + 128 * unroll_factor - 1) / (128 * unroll_factor);
  int vec_size = 1;
  UnaryElementwiseKernel<Functor, OutT, Arity, NumOuts, 1, unroll_factor>
      <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                       classifier.outs_data,
                                       numel,
                                       vec_size,
                                       func,
                                       offset_calc);
}

template <typename T, typename Context, typename Functor>
void LaunchUnaryElementwiseStrideKernel(const Context &dev_ctx,
                                        const DenseTensor &x,
                                        Functor func,
                                        DenseTensor *out) {
  std::vector<const DenseTensor *> inputs = {&x};
  std::vector<DenseTensor *> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  UnaryStrideElementwiseKernel<T, Context>(dev_ctx, inputs, &outputs, func);
}

template <typename Context>
phi::DenseTensor Tensor2Contiguous(const Context &dev_ctx,
                                   const phi::DenseTensor &tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  PD_VISIT_ALL_TYPES(tensor.dtype(), "Tensor2Contiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

#define DEFINE_CUDA_ACTIVATION_STRIDE_OP(name, functor_class)                 \
  template <typename T, typename Context>                                     \
  void name##StrideKernel(                                                    \
      const Context &dev_ctx, const DenseTensor &x, DenseTensor *out) {       \
    if (!FLAGS_use_stride_kernel) {                                           \
      PADDLE_THROW(common::errors::Fatal(                                     \
          "FLAGS_use_stride_kernel is closed. Strided kernel "                \
          "be called, something wrong has happened!"));                       \
    }                                                                         \
    DenseTensor x_;                                                           \
    if (!FLAGS_use_stride_compute_kernel || x.offset() != 0) {                \
      if (!x.meta().is_contiguous() || x.offset() != 0) {                     \
        x_ = Tensor2Contiguous<Context>(dev_ctx, x);                          \
      } else {                                                                \
        x_ = x;                                                               \
      }                                                                       \
    } else {                                                                  \
      x_ = x;                                                                 \
    }                                                                         \
    if (x_.meta().is_contiguous()) {                                          \
      auto meta = out->meta();                                                \
      meta.strides = meta.calc_strides(out->dims());                          \
      out->set_meta(meta);                                                    \
      phi::name##Kernel<T, Context>(dev_ctx, x_, out);                        \
      return;                                                                 \
    }                                                                         \
    if (!FLAGS_use_stride_compute_kernel) {                                   \
      PADDLE_THROW(                                                           \
          common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. " \
                                "Kernel using DenseTensorIterator "           \
                                "be called, something wrong has happened!")); \
    }                                                                         \
    LaunchUnaryElementwiseStrideKernel<T, Context>(                           \
        dev_ctx, x_, funcs::functor_class<T>(), out);                         \
  }

DEFINE_CUDA_ACTIVATION_STRIDE_OP(Cos, CudaCosFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Sin, CudaSinFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Tan, CudaTanFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Acos, CudaAcosFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Asin, CudaAsinFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Atan, CudaAtanFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Sinh, CudaSinhFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Cosh, CudaCoshFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Asinh, CudaAsinhFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Acosh, CudaAcoshFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Atanh, CudaAtanhFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Relu, CudaReluFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Tanh, CudaTanhFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Silu, CudaSiluFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Reciprocal, CudaReciprocalFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Square, CudaSquareFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Sqrt, CudaSqrtFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Rsqrt, CudaRsqrtFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Softsign, CudaSoftsignFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Sigmoid, CudaSigmoidFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(LogSigmoid, CudaLogSigmoidFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Floor, CudaFloorFunctor)
DEFINE_CUDA_ACTIVATION_STRIDE_OP(Ceil, CudaCeilFunctor)

#undef DEFINE_CUDA_ACTIVATION_STRIDE_OP

#define DEFINE_CUDA_ACTIVATION_WITH_INT_IN_FLOAT_OUT_STRIDE_OP(name,          \
                                                               functor_class) \
  template <typename T, typename Context>                                     \
  void name##StrideKernel(                                                    \
      const Context &dev_ctx, const DenseTensor &x, DenseTensor *out) {       \
    if (!FLAGS_use_stride_kernel) {                                           \
      PADDLE_THROW(common::errors::Fatal(                                     \
          "FLAGS_use_stride_kernel is closed. Strided kernel "                \
          "be called, something wrong has happened!"));                       \
    }                                                                         \
    DenseTensor x_;                                                           \
    if (!FLAGS_use_stride_compute_kernel || x.offset() != 0) {                \
      if (!x.meta().is_contiguous() || x.offset() != 0) {                     \
        x_ = Tensor2Contiguous<Context>(dev_ctx, x);                          \
      } else {                                                                \
        x_ = x;                                                               \
      }                                                                       \
    } else {                                                                  \
      x_ = x;                                                                 \
    }                                                                         \
    if (x_.meta().is_contiguous()) {                                          \
      auto meta = out->meta();                                                \
      meta.strides = meta.calc_strides(out->dims());                          \
      out->set_meta(meta);                                                    \
      phi::name##Kernel<T, Context>(dev_ctx, x_, out);                        \
      return;                                                                 \
    }                                                                         \
    if (!FLAGS_use_stride_compute_kernel) {                                   \
      PADDLE_THROW(                                                           \
          common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. " \
                                "Kernel using DenseTensorIterator "           \
                                "be called, something wrong has happened!")); \
    }                                                                         \
    using U =                                                                 \
        typename std::conditional_t<std::is_integral<T>::value, float, T>;    \
    LaunchUnaryElementwiseStrideKernel<U, Context>(                           \
        dev_ctx, x_, funcs::functor_class<T>(), out);                         \
  }

DEFINE_CUDA_ACTIVATION_WITH_INT_IN_FLOAT_OUT_STRIDE_OP(Log, CudaLogFunctor)
DEFINE_CUDA_ACTIVATION_WITH_INT_IN_FLOAT_OUT_STRIDE_OP(Log2, CudaLog2Functor)
DEFINE_CUDA_ACTIVATION_WITH_INT_IN_FLOAT_OUT_STRIDE_OP(Log10, CudaLog10Functor)
DEFINE_CUDA_ACTIVATION_WITH_INT_IN_FLOAT_OUT_STRIDE_OP(Log1p, CudaLog1pFunctor)
DEFINE_CUDA_ACTIVATION_WITH_INT_IN_FLOAT_OUT_STRIDE_OP(Exp, CudaExpFunctor)
DEFINE_CUDA_ACTIVATION_WITH_INT_IN_FLOAT_OUT_STRIDE_OP(Expm1, CudaExpm1Functor)

#undef DEFINE_CUDA_ACTIVATION_WITH_INT_IN_FLOAT_OUT_STRIDE_OP

#define DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS(                          \
    name, functor_class, attr)                                                 \
  template <typename T, typename Context>                                      \
  void name##StrideKernel(const Context &dev_ctx,                              \
                          const DenseTensor &x,                                \
                          float attr,                                          \
                          DenseTensor *out) {                                  \
    if (!FLAGS_use_stride_kernel) {                                            \
      PADDLE_THROW(common::errors::Fatal(                                      \
          "FLAGS_use_stride_kernel is closed. Strided kernel "                 \
          "be called, something wrong has happened!"));                        \
    }                                                                          \
    DenseTensor x_;                                                            \
    if (!FLAGS_use_stride_compute_kernel || x.offset() != 0) {                 \
      if (!x.meta().is_contiguous() || x.offset() != 0) {                      \
        x_ = Tensor2Contiguous<Context>(dev_ctx, x);                           \
      } else {                                                                 \
        x_ = x;                                                                \
      }                                                                        \
    } else {                                                                   \
      x_ = x;                                                                  \
    }                                                                          \
    if (x_.meta().is_contiguous()) {                                           \
      auto meta = out->meta();                                                 \
      meta.strides = meta.calc_strides(out->dims());                           \
      out->set_meta(meta);                                                     \
      phi::name##Kernel<T, Context>(dev_ctx, x_, attr, out);                   \
      return;                                                                  \
    }                                                                          \
    if (!FLAGS_use_stride_compute_kernel) {                                    \
      PADDLE_THROW(                                                            \
          common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "  \
                                "Kernel using DenseTensorIterator "            \
                                "be called, something wrong has happened!"));  \
    }                                                                          \
    funcs::functor_class<T> functor;                                           \
    auto attrs = functor.GetAttrs();                                           \
    *(attrs[0].second) = attr;                                                 \
    LaunchUnaryElementwiseStrideKernel<T, Context>(dev_ctx, x_, functor, out); \
  }

DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS(LeakyRelu,
                                             CudaLeakyReluFunctor,
                                             alpha)
DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS(HardShrink,
                                             CudaHardShrinkFunctor,
                                             threshold)
DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS(SoftShrink,
                                             CudaSoftShrinkFunctor,
                                             lambda)
DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS(Elu, CudaELUFunctor, alpha)
DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS(Celu, CudaCELUFunctor, alpha)
DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS(Mish, CudaMishFunctor, threshold)
#undef DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS

#define DEFINE_CUDA_ACTIVATION_STRIDE_WITH_TWO_ATTRS(                          \
    name, functor_class, attr1, attr2)                                         \
  template <typename T, typename Context>                                      \
  void name##StrideKernel(const Context &dev_ctx,                              \
                          const DenseTensor &x,                                \
                          float attr1,                                         \
                          float attr2,                                         \
                          DenseTensor *out) {                                  \
    if (!FLAGS_use_stride_kernel) {                                            \
      PADDLE_THROW(common::errors::Fatal(                                      \
          "FLAGS_use_stride_kernel is closed. Strided kernel "                 \
          "be called, something wrong has happened!"));                        \
    }                                                                          \
    DenseTensor x_;                                                            \
    if (!FLAGS_use_stride_compute_kernel || x.offset() != 0) {                 \
      if (!x.meta().is_contiguous() || x.offset() != 0) {                      \
        x_ = Tensor2Contiguous<Context>(dev_ctx, x);                           \
      } else {                                                                 \
        x_ = x;                                                                \
      }                                                                        \
    } else {                                                                   \
      x_ = x;                                                                  \
    }                                                                          \
    if (x_.meta().is_contiguous()) {                                           \
      auto meta = out->meta();                                                 \
      meta.strides = meta.calc_strides(out->dims());                           \
      out->set_meta(meta);                                                     \
      phi::name##Kernel<T, Context>(dev_ctx, x_, attr1, attr2, out);           \
      return;                                                                  \
    }                                                                          \
    if (!FLAGS_use_stride_compute_kernel) {                                    \
      PADDLE_THROW(                                                            \
          common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "  \
                                "Kernel using DenseTensorIterator "            \
                                "be called, something wrong has happened!"));  \
    }                                                                          \
    funcs::functor_class<T> functor;                                           \
    auto attrs = functor.GetAttrs();                                           \
    *(attrs[0].second) = attr1;                                                \
    *(attrs[1].second) = attr2;                                                \
    LaunchUnaryElementwiseStrideKernel<T, Context>(dev_ctx, x_, functor, out); \
  }

DEFINE_CUDA_ACTIVATION_STRIDE_WITH_TWO_ATTRS(HardTanh,
                                             CudaHardTanhFunctor,
                                             t_min,
                                             t_max)
DEFINE_CUDA_ACTIVATION_STRIDE_WITH_TWO_ATTRS(Softplus,
                                             CudaSoftplusFunctor,
                                             beta,
                                             threshold)
DEFINE_CUDA_ACTIVATION_STRIDE_WITH_TWO_ATTRS(HardSigmoid,
                                             CudaHardSigmoidFunctor,
                                             slope,
                                             offset)
DEFINE_CUDA_ACTIVATION_STRIDE_WITH_TWO_ATTRS(Selu,
                                             CudaSeluFunctor,
                                             scale,
                                             alpha)

#undef DEFINE_CUDA_ACTIVATION_STRIDE_WITH_ONE_ATTRS

template <typename T, typename Context>
void RoundStrideKernel(const Context &dev_ctx,
                       const DenseTensor &x,
                       const int decimals,
                       DenseTensor *out) {
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
  if (x_.meta().is_contiguous()) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::RoundKernel<T, Context>(dev_ctx, x_, decimals, out);
    return;
  }
  if (!FLAGS_use_stride_compute_kernel) {
    PADDLE_THROW(
        common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
                              "Kernel using DenseTensorIterator "
                              "be called, something wrong has happened!"));
  }
  funcs::CudaRoundFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = decimals;
  LaunchUnaryElementwiseStrideKernel<T, Context>(dev_ctx, x_, functor, out);
}

template <typename T, typename Context>
void HardSwishStrideKernel(const Context &dev_ctx,
                           const DenseTensor &x,
                           DenseTensor *out) {
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
  if (x_.meta().is_contiguous()) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::HardSwishKernel<T, Context>(dev_ctx, x_, out);
    return;
  }
  if (!FLAGS_use_stride_compute_kernel) {
    PADDLE_THROW(
        common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
                              "Kernel using DenseTensorIterator "
                              "be called, something wrong has happened!"));
  }
  funcs::CudaHardSwishFunctor<T> functor;
  float threshold = 6;
  float scale = 6;
  float offset = 3;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = threshold;
  *(attrs[1].second) = scale;
  *(attrs[2].second) = offset;
  LaunchUnaryElementwiseStrideKernel<T, Context>(dev_ctx, x_, functor, out);
}

template <typename T, typename Enable = void>
struct CudaAbsFunctor;

template <typename T>
struct CudaAbsFunctor<T, phi::funcs::Complex<T, phi::dtype::Real<T>>> {
  __device__ __forceinline__ phi::dtype::Real<T> operator()(const T x) const {
    return abs(x);
  }
};

template <typename T>
struct CudaAbsFunctor<
    T,
    std::enable_if_t<std::is_same<T, phi::dtype::Real<T>>::value &&
                     std::is_same<T, phi::dtype::bfloat16>::value>> {
  __device__ __forceinline__ T operator()(const T x) const { return abs(x); }
};

template <typename T>
struct CudaAbsFunctor<
    T,
    std::enable_if_t<std::is_same<T, phi::dtype::Real<T>>::value &&
                     !std::is_same<T, phi::dtype::bfloat16>::value>> {
  __device__ __forceinline__ T operator()(const T x) const {
    return std::abs(x);
  }
};

template <typename T, typename Context>
void AbsStrideKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     DenseTensor *out) {
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
  if (x_.meta().is_contiguous()) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::AbsKernel<T, Context>(dev_ctx, x_, out);
    return;
  }
  if (!FLAGS_use_stride_compute_kernel) {
    PADDLE_THROW(
        common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
                              "Kernel using DenseTensorIterator "
                              "be called, something wrong has happened!"));
  }
  auto functor = CudaAbsFunctor<T>();
  LaunchUnaryElementwiseStrideKernel<phi::dtype::Real<T>, Context>(
      dev_ctx, x_, functor, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(abs,
                   GPU,
                   STRIDED,
                   phi::AbsStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

#define REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(cos, func) \
  PD_REGISTER_KERNEL(cos,                                         \
                     GPU,                                         \
                     STRIDED,                                     \
                     phi::func,                                   \
                     float,                                       \
                     double,                                      \
                     phi::dtype::float16,                         \
                     phi::dtype::bfloat16,                        \
                     phi::dtype::complex<float>,                  \
                     phi::dtype::complex<double>) {}

#define REGISTER_ACTIVATION_STRIDE_KERNEL(leaky_relu, func) \
  PD_REGISTER_KERNEL(leaky_relu,                            \
                     GPU,                                   \
                     STRIDED,                               \
                     phi::func,                             \
                     float,                                 \
                     double,                                \
                     phi::dtype::float16,                   \
                     phi::dtype::bfloat16) {}

REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(cos, CosStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(sin, SinStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(tan, TanStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(acos, AcosStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(asin, AsinStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(atan, AtanStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(sinh, SinhStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(cosh, CoshStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(asinh, AsinhStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(acosh, AcoshStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(atanh, AtanhStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(silu, SiluStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(reciprocal,
                                               ReciprocalStrideKernel)
PD_REGISTER_KERNEL(square,
                   GPU,
                   STRIDED,
                   phi::SquareStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(sqrt, SqrtStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(rsqrt, RsqrtStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(softsign, SoftsignStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(sigmoid, SigmoidStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(logsigmoid,
                                               LogSigmoidStrideKernel)

PD_REGISTER_KERNEL(floor,
                   GPU,
                   STRIDED,
                   phi::FloorStrideKernel,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(ceil,
                   GPU,
                   STRIDED,
                   phi::CeilStrideKernel,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(log,
                   GPU,
                   STRIDED,
                   phi::LogStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(log2,
                   GPU,
                   STRIDED,
                   phi::Log2StrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(log10,
                   GPU,
                   STRIDED,
                   phi::Log10StrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(log1p,
                   GPU,
                   STRIDED,
                   phi::Log1pStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(exp,
                   GPU,
                   STRIDED,
                   phi::ExpStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(expm1,
                   GPU,
                   STRIDED,
                   phi::Expm1StrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

REGISTER_ACTIVATION_STRIDE_KERNEL(leaky_relu, LeakyReluStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(hard_shrink, HardShrinkStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(softshrink, SoftShrinkStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(celu, CeluStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(elu, EluStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(mish, MishStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(hardtanh, HardTanhStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(softplus, SoftplusStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(hardsigmoid, HardSigmoidStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL(selu, SeluStrideKernel)
REGISTER_ACTIVATION_STRIDE_KERNEL_WITH_COMPLEX(hardswish, HardSwishStrideKernel)

PD_REGISTER_KERNEL(round,
                   GPU,
                   STRIDED,
                   phi::RoundStrideKernel,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#endif
