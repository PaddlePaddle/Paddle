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

#pragma once

#include <string>

#include "paddle/phi/kernels/quantize_linear_kernel.h"

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"

namespace phi {

template <typename Context, typename T>
struct DequantizeFunctor {
  void operator()(const Context& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  T max_range,
                  phi::DenseTensor* out);
};

template <typename Context, typename T>
struct ChannelDequantizeFunctorV2 {
  void operator()(const Context& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor** scales,
                  const int scale_num,
                  T max_range,
                  const int quant_axis,
                  phi::DenseTensor* out);
};

template <typename T, typename Context, typename D>
void DeQuantizeLinearImpl(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& scale,
                          int quant_axis,
                          int qmax,
                          bool only_observer,
                          DenseTensor* out) {
  auto* in = &x;

  auto in_tmp = phi::Cast<T>(dev_ctx, *in, phi::CppTypeToDataType<D>::Type());

  dev_ctx.template Alloc<D>(out, out->numel() * sizeof(D));

  if (only_observer) {
    phi::Copy(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
    return;
  }

  if (quant_axis < 0) {
    float max_range = qmax;
    DequantizeFunctor<Context, D>()(
        dev_ctx, &in_tmp, &scale, static_cast<D>(max_range), out);
  } else {
    PADDLE_ENFORCE_EQ(
        scale.numel(),
        in_tmp.dims()[quant_axis],
        common::errors::PreconditionNotMet(
            "The number of first scale values must be the same with "
            "quant_axis dimension value of Input(X) when the `scale` has "
            "only one element, but %ld != %ld here.",
            scale.numel(),
            in_tmp.dims()[quant_axis]));
    int max_range = qmax;

    ChannelDequantizeFunctorV2<Context, D>()(
        dev_ctx, &in_tmp, &scale, static_cast<D>(max_range), quant_axis, out);
  }
}

// Note: We should re-design this kernel's args when we abandon fluid op
// definition
template <typename T, typename Context>
void DeQuantizeLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& in_scale,
                            const DenseTensor& zero_point,
                            const paddle::optional<DenseTensor>& in_accum,
                            const paddle::optional<DenseTensor>& in_state,
                            int quant_axis,
                            int bit_length,
                            int qmin,
                            int qmax,
                            int round_type,
                            bool is_test,
                            bool only_observer,
                            DenseTensor* out,
                            DenseTensor* out_state,
                            DenseTensor* out_accum,
                            DenseTensor* out_scale) {
  PADDLE_ENFORCE_NE(in_scale.get_ptr(),
                    nullptr,
                    common::errors::PreconditionNotMet(
                        "in_scale can't be nullptr in DeQuantizeLinearKernel"));
  auto scale = in_scale.get();
  switch (scale.dtype()) {
    case phi::DataType::FLOAT64:
      DeQuantizeLinearImpl<T, Context, double>(
          dev_ctx, x, scale, quant_axis, qmax, only_observer, out);
      break;
    case phi::DataType::FLOAT32:
      DeQuantizeLinearImpl<T, Context, float>(
          dev_ctx, x, scale, quant_axis, qmax, only_observer, out);
      break;
    case phi::DataType::FLOAT16:
      DeQuantizeLinearImpl<T, Context, float16>(
          dev_ctx, x, scale, quant_axis, qmax, only_observer, out);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "In DeQuantizeLinearKernel, "
          "data type %d for scale/output is not supported ",
          scale.dtype()));
      break;
  }
}

template <typename T, typename Context>
void QuantizeLinearTrainKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const paddle::optional<DenseTensor>& scale,
                               const DenseTensor& zero_point,
                               const paddle::optional<DenseTensor>& in_accum,
                               const paddle::optional<DenseTensor>& in_state,
                               int quant_axis,
                               int bit_length,
                               int qmin,
                               int qmax,
                               int round_type,
                               bool only_observer,
                               DenseTensor* out,
                               DenseTensor* out_state,
                               DenseTensor* out_accum,
                               DenseTensor* out_scale) {
  PADDLE_ENFORCE_NE(scale.get_ptr(),
                    nullptr,
                    common::errors::PreconditionNotMet(
                        "in_scale can't be nullptr in DeQuantizeLinearKernel"));
  auto* in = &x;
  dev_ctx.template Alloc<float>(out);

  if (quant_axis < 0) {
    // training
    phi::DenseTensor tmp_scale;
    tmp_scale.Resize(common::make_dim(1));
    T* cur_scale_data = dev_ctx.template Alloc<T>(&tmp_scale);

    phi::funcs::FindAbsMaxFunctor<Context, T>()(
        dev_ctx, in->data<T>(), in->numel(), cur_scale_data);

    dev_ctx.template Alloc<T>(out_state);
    dev_ctx.template Alloc<T>(out_accum);
    dev_ctx.template Alloc<T>(out_scale);

    phi::funcs::FindMovingAverageAbsMaxFunctor<Context, T>()(dev_ctx,
                                                             in_accum.get(),
                                                             in_state.get(),
                                                             cur_scale_data,
                                                             0.9,
                                                             out_state,
                                                             out_accum,
                                                             out_scale);
    if (only_observer) {
      phi::Copy<Context>(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
    } else {
      phi::funcs::ClipAndFakeQuantFunctor<Context, T>()(
          dev_ctx, *in, *out_scale, qmax, round_type, out);
    }
  } else {
    T* out_scale_data = dev_ctx.template Alloc<T>(out_scale);
    phi::funcs::FindChannelAbsMaxFunctor<Context, T>()(
        dev_ctx, *in, quant_axis, out_scale_data);
    if (only_observer) {
      phi::Copy<Context>(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
    } else {
      phi::funcs::ChannelClipAndFakeQuantFunctor<Context, T>()(
          dev_ctx, *in, *out_scale, qmax, round_type, quant_axis, out);
    }
  }
}

template <typename T, typename Context>
void QuantizeLinearInferKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const paddle::optional<DenseTensor>& scale,
                               const DenseTensor& zero_point,
                               int quant_axis,
                               int bit_length,
                               int qmin,
                               int qmax,
                               int round_type,
                               bool only_observer,
                               DenseTensor* out) {
  PADDLE_ENFORCE_NE(scale.get_ptr(),
                    nullptr,
                    common::errors::PreconditionNotMet(
                        "in_scale can't be nullptr in DeQuantizeLinearKernel"));
  auto* in = &x;
  auto* in_scale = scale.get_ptr();
  dev_ctx.template Alloc<float>(out);

  if (quant_axis < 0) {
    if (only_observer) {
      phi::Copy<Context>(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
    } else {
      phi::funcs::ClipAndFakeQuantFunctor<Context, T>()(
          dev_ctx, *in, *in_scale, qmax, round_type, out);
    }
  } else {
    if (only_observer) {
      phi::Copy<Context>(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
    } else {
      phi::funcs::ChannelClipAndFakeQuantFunctor<Context, T>()(
          dev_ctx, *in, *in_scale, qmax, round_type, quant_axis, out);
    }
  }
}

// Note: We should re-design this kernel's args when we abandon fluid op
// definition
template <typename T, typename Context>
void QuantizeLinearKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const paddle::optional<DenseTensor>& scale,
                          const DenseTensor& zero_point,
                          const paddle::optional<DenseTensor>& in_accum,
                          const paddle::optional<DenseTensor>& in_state,
                          int quant_axis,
                          int bit_length,
                          int qmin,
                          int qmax,
                          int round_type,
                          bool is_test,
                          bool only_observer,
                          DenseTensor* out,
                          DenseTensor* out_state,
                          DenseTensor* out_accum,
                          DenseTensor* out_scale) {
  if (!is_test) {
    QuantizeLinearTrainKernel<T, Context>(dev_ctx,
                                          x,
                                          scale,
                                          zero_point,
                                          in_accum,
                                          in_state,
                                          quant_axis,
                                          bit_length,
                                          qmin,
                                          qmax,
                                          round_type,
                                          only_observer,
                                          out,
                                          out_state,
                                          out_accum,
                                          out_scale);
  } else {
    QuantizeLinearInferKernel<T, Context>(dev_ctx,
                                          x,
                                          scale,
                                          zero_point,
                                          quant_axis,
                                          bit_length,
                                          qmin,
                                          qmax,
                                          round_type,
                                          only_observer,
                                          out);
  }
}

template <typename T, typename Context>
void QuantizeLinearDeprecatedTrainKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& in_scale,
    const DenseTensor& zero_point,
    const paddle::optional<DenseTensor>& in_accum,
    const paddle::optional<DenseTensor>& in_state,
    int quant_axis,
    int bit_length,
    int qmin,
    int qmax,
    int round_type,
    bool only_observer,
    DenseTensor* out,
    DenseTensor* out_state,
    DenseTensor* out_accum,
    DenseTensor* out_scale) {
  paddle::optional<phi::DenseTensor> scale =
      paddle::make_optional<phi::DenseTensor>(in_scale);
  QuantizeLinearTrainKernel<T, Context>(dev_ctx,
                                        x,
                                        scale,
                                        zero_point,
                                        in_accum,
                                        in_state,
                                        quant_axis,
                                        bit_length,
                                        qmin,
                                        qmax,
                                        round_type,
                                        only_observer,
                                        out,
                                        out_state,
                                        out_accum,
                                        out_scale);
}

template <typename T, typename Context>
void QuantizeLinearDeprecatedInferKernel(const Context& dev_ctx,
                                         const DenseTensor& x,
                                         const DenseTensor& in_scale,
                                         const DenseTensor& zero_point,
                                         int quant_axis,
                                         int bit_length,
                                         int qmin,
                                         int qmax,
                                         int round_type,
                                         bool only_observer,
                                         DenseTensor* out) {
  paddle::optional<phi::DenseTensor> scale =
      paddle::make_optional<phi::DenseTensor>(in_scale);
  QuantizeLinearInferKernel<T, Context>(dev_ctx,
                                        x,
                                        scale,
                                        zero_point,
                                        quant_axis,
                                        bit_length,
                                        qmin,
                                        qmax,
                                        round_type,
                                        only_observer,
                                        out);
}

template <typename T, typename Context>
void DeQuantizeLinearDeprecatedKernel(const Context& dev_ctx,
                                      const DenseTensor& x,
                                      const DenseTensor& in_scale,
                                      const DenseTensor& zero_point,
                                      int quant_axis,
                                      int bit_length,
                                      int qmin,
                                      int qmax,
                                      int round_type,
                                      bool only_observer,
                                      DenseTensor* out) {
  paddle::optional<phi::DenseTensor> scale =
      paddle::make_optional<phi::DenseTensor>(in_scale);
  DeQuantizeLinearKernel<T, Context>(dev_ctx,
                                     x,
                                     scale,
                                     zero_point,
                                     nullptr,
                                     nullptr,
                                     quant_axis,
                                     bit_length,
                                     qmin,
                                     qmax,
                                     round_type,
                                     true,
                                     only_observer,
                                     out,
                                     nullptr,
                                     nullptr,
                                     nullptr);
}

}  // namespace phi
