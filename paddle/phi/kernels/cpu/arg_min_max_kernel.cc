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

#include "paddle/phi/kernels/arg_min_max_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

enum ArgMinMaxType { kArgMin, kArgMax };

template <typename Context,
          typename T,
          typename Tout,
          int64_t Rank,
          ArgMinMaxType argMinMaxValue>
struct ArgMinMaxFunctor {};

#define DECLARE_ARG_MIN_MAX_FUNCTOR(eigen_op_type, enum_argminmax_value)  \
  template <typename Context, typename T, typename Tout, int64_t Rank>    \
  struct ArgMinMaxFunctor<Context, T, Tout, Rank, enum_argminmax_value> { \
    void operator()(const Context& dev_ctx,                               \
                    const DenseTensor& in,                                \
                    DenseTensor* out,                                     \
                    phi::DDim x_dims,                                     \
                    int64_t axis,                                         \
                    bool keepdims) {                                      \
      auto in_eigen = EigenTensor<T, Rank>::From(in, x_dims);             \
      if (keepdims) {                                                     \
        auto out_eigen = EigenTensor<Tout, Rank>::From(*out);             \
        out_eigen.device(*(dev_ctx.eigen_device())) =                     \
            in_eigen.eigen_op_type(axis).template cast<Tout>();           \
      } else {                                                            \
        auto out_eigen = EigenTensor<Tout, Rank - 1>::From(*out);         \
        out_eigen.device(*(dev_ctx.eigen_device())) =                     \
            in_eigen.eigen_op_type(axis).template cast<Tout>();           \
      }                                                                   \
    }                                                                     \
  }

DECLARE_ARG_MIN_MAX_FUNCTOR(argmin, ArgMinMaxType::kArgMin);
DECLARE_ARG_MIN_MAX_FUNCTOR(argmax, ArgMinMaxType::kArgMax);

template <typename Context, typename T, ArgMinMaxType EnumArgMinMaxValue>
struct VisitDataArgMinMaxFunctor {
  const Context& dev_ctx;
  const DenseTensor& x;
  int64_t axis;
  bool keepdims;
  bool flatten;
  DenseTensor* out;

  explicit VisitDataArgMinMaxFunctor(const Context& dev_ctx,
                                     const DenseTensor& x,
                                     int64_t axis,
                                     bool keepdims,
                                     bool flatten,
                                     DenseTensor* out)
      : dev_ctx(dev_ctx),
        x(x),
        axis(axis),
        keepdims(keepdims),
        flatten(flatten),
        out(out) {}
  template <typename Tout>
  void apply() const {
    dev_ctx.template Alloc<Tout>(out);
    bool new_keepdims = keepdims;
    if (flatten) new_keepdims = true;

    // if flatten, will construct the new dims for the cacluate
    phi::DDim x_dims;
    int new_axis = axis;
    if (flatten) {
      x_dims = phi::make_ddim({x.numel()});
      // if flatten, the axis just as 0
      new_axis = 0;
    } else {
      x_dims = x.dims();
      if (axis < 0) new_axis = axis + x_dims.size();
    }

#define CALL_ARG_MINMAX_FUNCTOR(rank)                                         \
  ArgMinMaxFunctor<Context, T, Tout, rank, EnumArgMinMaxValue> functor##rank; \
  functor##rank(dev_ctx, x, out, x_dims, new_axis, new_keepdims)

    switch (x_dims.size()) {
      case 1:
        CALL_ARG_MINMAX_FUNCTOR(1);
        break;
      case 2:
        CALL_ARG_MINMAX_FUNCTOR(2);
        break;
      case 3:
        CALL_ARG_MINMAX_FUNCTOR(3);
        break;
      case 4:
        CALL_ARG_MINMAX_FUNCTOR(4);
        break;
      case 5:
        CALL_ARG_MINMAX_FUNCTOR(5);
        break;
      case 6:
        CALL_ARG_MINMAX_FUNCTOR(6);
        break;
      default:
        PADDLE_ENFORCE_LE(
            x_dims.size(),
            6,
            phi::errors::InvalidArgument(
                "%s operator doesn't supports tensors whose ranks are greater "
                "than 6.",
                (EnumArgMinMaxValue == kArgMin ? "argmin" : "argmax")));
        break;
#undef CALL_ARG_MINMAX_FUNCTOR
    }
  }
};

template <typename Context, typename T, ArgMinMaxType EnumArgMinMaxValue>
void ArgMinMaxKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     int64_t axis,
                     bool keepdims,
                     bool flatten,
                     int dtype,
                     DenseTensor* out) {
  if (dtype < 0) {
    paddle::framework::VisitDataTypeTiny(
        static_cast<paddle::framework::proto::VarType::Type>(
            paddle::framework::proto::VarType::INT64),
        VisitDataArgMinMaxFunctor<Context, T, EnumArgMinMaxValue>(
            dev_ctx, x, axis, keepdims, flatten, out));
    return;
  }
  paddle::framework::VisitDataTypeTiny(
      static_cast<paddle::framework::proto::VarType::Type>(dtype),
      VisitDataArgMinMaxFunctor<Context, T, EnumArgMinMaxValue>(
          dev_ctx, x, axis, keepdims, flatten, out));
}

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int64_t axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  DenseTensor* out) {
  ArgMinMaxKernel<Context, T, ArgMinMaxType::kArgMin>(
      dev_ctx, x, axis, keepdims, flatten, dtype, out);
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int64_t axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  DenseTensor* out) {
  ArgMinMaxKernel<Context, T, ArgMinMaxType::kArgMax>(
      dev_ctx, x, axis, keepdims, flatten, dtype, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(arg_min,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArgMinKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   int16_t,
                   uint8_t) {}

PD_REGISTER_KERNEL(arg_max,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArgMaxKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   int16_t,
                   uint8_t) {}
