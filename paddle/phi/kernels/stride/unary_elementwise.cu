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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/logical_functor.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

#include "paddle/phi/kernels/funcs/bitwise_functors.h"

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
__global__ void BinaryElementwiseKernel(
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
      std::get<1>(args[idx]) =
          *(reinterpret_cast<const _ptr_ std::tuple_element_t<1, ArgsT> *>(
              reinterpret_cast<const _ptr_ char *>(ins[1]) + offsets[2]));
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

// Not Support Vectorized Kernel For Now
#define VEC_SIZE 1

template <typename OutT, typename Context, typename Functor, int NumOuts = 1>
void BinaryStrideElementwiseKernel(const Context &dev_ctx,
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
  config.add_const_input(*(ins[1]));
  DenseTensorIterator iter = config.build();
  const int &numel = iter.numel();
  funcs::OffsetCalculator offset_calc = funcs::make_offset_calculator<3>(iter);
  constexpr int unroll_factor = sizeof(OutT) >= 4 ? 2 : 4;
  auto stream = dev_ctx.stream();
  auto threads = 128;
  auto blocks = (numel + 128 * unroll_factor - 1) / (128 * unroll_factor);
  int vec_size = VEC_SIZE;
  BinaryElementwiseKernel<Functor,
                          OutT,
                          Arity,
                          NumOuts,
                          VEC_SIZE,
                          unroll_factor>
      <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                       classifier.outs_data,
                                       numel,
                                       vec_size,
                                       func,
                                       offset_calc);
}

template <typename OutT, typename Context, typename Functor, int NumOuts = 1>
void BinaryStrideBroadcastKernel(const Context &dev_ctx,
                                 const std::vector<const DenseTensor *> &ins,
                                 std::vector<DenseTensor *> *outs,
                                 Functor func,
                                 int axis = -1) {
  using Traits = phi::funcs::FunctionTraits<Functor>;
  const int Arity = Traits::arity;
  for (auto i = 0; i < outs->size(); ++i) {
    if (i > 0) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          common::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, but "
              "%d-th output tensor`s shape is not.",
              i));
    }
    dev_ctx.template Alloc<OutT>((*outs)[i]);
  }
  if ((*outs)[0]->numel() == 0) {
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
  axis = axis == -1 ? max_rank - min_rank : axis;
  auto classifier =
      funcs::BroadcastTypeClassifier<OutT, Functor, Arity, NumOuts>(
          ins, outs, axis);
  DenseTensorIteratorConfig config;
  config.add_output(*((*outs)[0]));
  config.add_const_input(*(ins[0]));
  config.add_const_input(*(ins[1]));
  DenseTensorIterator iter = config.build();
  const int &numel = iter.numel();
  funcs::OffsetCalculator offset_calc = funcs::make_offset_calculator<3>(iter);
  constexpr int unroll_factor = sizeof(OutT) >= 4 ? 2 : 4;
  auto stream = dev_ctx.stream();
  auto threads = 128;
  auto blocks = (numel + 128 * unroll_factor - 1) / (128 * unroll_factor);
  int vec_size = VEC_SIZE;
  BinaryElementwiseKernel<Functor,
                          OutT,
                          Arity,
                          NumOuts,
                          VEC_SIZE,
                          unroll_factor>
      <<<blocks, threads, 0, stream>>>(classifier.ins_data,
                                       classifier.outs_data,
                                       numel,
                                       vec_size,
                                       func,
                                       offset_calc);
}

template <typename T, typename Context, typename Functor>
void LaunchBoolBinaryElementwiseStrideKernel(const Context &dev_ctx,
                                             const DenseTensor &x,
                                             const DenseTensor &y,
                                             Functor func,
                                             DenseTensor *out) {
  std::vector<const DenseTensor *> inputs = {&x, &y};
  std::vector<DenseTensor *> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  BinaryStrideElementwiseKernel<T, Context>(dev_ctx, inputs, &outputs, func);
}

template <typename T, typename Context, typename Functor>
void LaunchBinaryElementwiseStrideKernel(const Context &dev_ctx,
                                         const DenseTensor &x,
                                         const DenseTensor &y,
                                         Functor func,
                                         int axis,
                                         DenseTensor *out) {
  std::vector<const DenseTensor *> inputs = {&x, &y};
  std::vector<DenseTensor *> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  BinaryStrideBroadcastKernel<T, Context>(
      dev_ctx, inputs, &outputs, func, axis);
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
  int vec_size = VEC_SIZE;
  UnaryElementwiseKernel<Functor, OutT, Arity, NumOuts, VEC_SIZE, unroll_factor>
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

#define DEFINE_CUDA_BINARY_ELEMENTWISE_WITH_BOOL_STRIDE_OP(name)              \
  template <typename T, typename Context>                                     \
  void Bitwise##name##StrideKernel(const Context &dev_ctx,                    \
                                   const DenseTensor &x,                      \
                                   const DenseTensor &y,                      \
                                   bool is_arithmetic,                        \
                                   DenseTensor *out) {                        \
    if (!FLAGS_use_stride_kernel) {                                           \
      PADDLE_THROW(common::errors::Fatal(                                     \
          "FLAGS_use_stride_kernel is closed. Strided kernel "                \
          "be called, something wrong has happened!"));                       \
    }                                                                         \
    DenseTensor x_;                                                           \
    DenseTensor y_;                                                           \
    if (!FLAGS_use_stride_compute_kernel || x.offset() != 0 ||                \
        y.offset() != 0) {                                                    \
      if (!x.meta().is_contiguous() || x.offset() != 0) {                     \
        x_ = Tensor2Contiguous<Context>(dev_ctx, x);                          \
      } else {                                                                \
        x_ = x;                                                               \
      }                                                                       \
      if (!y.meta().is_contiguous() || y.offset() != 0) {                     \
        y_ = Tensor2Contiguous<Context>(dev_ctx, y);                          \
      } else {                                                                \
        y_ = y;                                                               \
      }                                                                       \
    } else {                                                                  \
      x_ = x;                                                                 \
      y_ = y;                                                                 \
    }                                                                         \
    if (x_.meta().is_contiguous() && y_.meta().is_contiguous()) {             \
      auto meta = out->meta();                                                \
      meta.strides = meta.calc_strides(out->dims());                          \
      out->set_meta(meta);                                                    \
      dev_ctx.template Alloc<T>(out);                                         \
      std::vector<const DenseTensor *> ins = {&x_, &y_};                      \
      std::vector<DenseTensor *> outs = {out};                                \
      if (is_arithmetic) {                                                    \
        funcs::Bitwise##name##ArithmeticFunctor<T> func;                      \
        funcs::BroadcastKernel<T>(dev_ctx, ins, &outs, func);                 \
      } else {                                                                \
        funcs::Bitwise##name##LogicFunctor<T> func;                           \
        funcs::BroadcastKernel<T>(dev_ctx, ins, &outs, func);                 \
      }                                                                       \
      return;                                                                 \
    }                                                                         \
    if (!FLAGS_use_stride_compute_kernel) {                                   \
      PADDLE_THROW(                                                           \
          common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. " \
                                "Kernel using DenseTensorIterator "           \
                                "be called, something wrong has happened!")); \
    }                                                                         \
    if (is_arithmetic) {                                                      \
      LaunchBinaryElementwiseStrideKernel<T, Context>(                        \
          dev_ctx,                                                            \
          x_,                                                                 \
          y_,                                                                 \
          funcs::Bitwise##name##ArithmeticFunctor<T>(),                       \
          -1,                                                                 \
          out);                                                               \
    } else {                                                                  \
      LaunchBinaryElementwiseStrideKernel<T, Context>(                        \
          dev_ctx, x_, y_, funcs::Bitwise##name##LogicFunctor<T>(), -1, out); \
    }                                                                         \
  }

DEFINE_CUDA_BINARY_ELEMENTWISE_WITH_BOOL_STRIDE_OP(LeftShift)
DEFINE_CUDA_BINARY_ELEMENTWISE_WITH_BOOL_STRIDE_OP(RightShift)
#undef DEFINE_CUDA_BINARY_ELEMENTWISE_WITH_BOOL_STRIDE_OP

template <typename T, typename Context>
void BitwiseNotStrideKernel(const Context &dev_ctx,
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
    dev_ctx.template Alloc<T>(out);
    std::vector<const DenseTensor *> ins = {&x_};
    std::vector<DenseTensor *> outs = {out};
    funcs::BitwiseNotFunctor<T> unary_func;
    funcs::ElementwiseKernel<T, funcs::BitwiseNotFunctor<T>>(
        dev_ctx, ins, &outs, unary_func);
    return;
  }
  if (!FLAGS_use_stride_compute_kernel) {
    PADDLE_THROW(
        common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
                              "Kernel using DenseTensorIterator "
                              "be called, something wrong has happened!"));
  }
  LaunchUnaryElementwiseStrideKernel<T, Context>(
      dev_ctx, x_, funcs::BitwiseNotFunctor<T>(), out);
}

template <typename T, typename Context>
void LogicalNotStrideKernel(const Context &dev_ctx,
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
    if (!out->IsSharedWith(x_)) {
      dev_ctx.template Alloc<bool>(out);
      funcs::LogicalNotFunctor<T> unary_func;
      std::vector<const DenseTensor *> ins = {&x_};
      std::vector<DenseTensor *> outs = {out};
      funcs::BroadcastKernel<bool>(dev_ctx, ins, &outs, unary_func);
    } else {
      auto x_origin = x_;
      out->set_type(phi::DataType::BOOL);
      dev_ctx.template Alloc<bool>(out);
      funcs::LogicalNotFunctor<T> unary_func;
      std::vector<const DenseTensor *> ins = {&x_origin};
      std::vector<DenseTensor *> outs = {out};
      funcs::BroadcastKernel<bool>(dev_ctx, ins, &outs, unary_func);
    }

    return;
  }
  dev_ctx.template Alloc<bool>(out);
  if (!out->IsSharedWith(x_)) {
    LaunchUnaryElementwiseStrideKernel<T, Context>(
        dev_ctx, x_, funcs::LogicalNotFunctor<T>(), out);
  } else {
    auto x_origin = x_;
    out->set_type(phi::DataType::BOOL);
    LaunchUnaryElementwiseStrideKernel<T, Context>(
        dev_ctx, x_origin, funcs::LogicalNotFunctor<T>(), out);
  }
}

}  // namespace phi

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(bitwise_left_shift,
                   GPU,
                   STRIDED,
                   phi::BitwiseLeftShiftStrideKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(bitwise_right_shift,
                   GPU,
                   STRIDED,
                   phi::BitwiseRightShiftStrideKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(bitwise_not,
                   GPU,
                   STRIDED,
                   phi::BitwiseNotStrideKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

#define REGISTER_LOGICAL_CUDA_KERNEL(logical_and, func_type) \
  PD_REGISTER_KERNEL(logical_and,                            \
                     GPU,                                    \
                     STRIDED,                                \
                     phi::Logical##func_type##StrideKernel,  \
                     float,                                  \
                     phi::dtype::float16,                    \
                     phi::dtype::bfloat16,                   \
                     double,                                 \
                     bool,                                   \
                     int64_t,                                \
                     int,                                    \
                     int8_t,                                 \
                     phi::dtype::complex<float>,             \
                     phi::dtype::complex<double>,            \
                     int16_t) {                              \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);    \
  }

REGISTER_LOGICAL_CUDA_KERNEL(logical_not, Not)

#endif
