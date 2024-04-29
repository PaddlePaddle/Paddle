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
                          int bit_length,
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
    float max_range = (std::pow(2, bit_length - 1) - 1);
    DequantizeFunctor<Context, D>()(
        dev_ctx, &in_tmp, &scale, static_cast<D>(max_range), out);
  } else {
    PADDLE_ENFORCE_EQ(
        scale.numel(),
        in_tmp.dims()[quant_axis],
        phi::errors::PreconditionNotMet(
            "The number of first scale values must be the same with "
            "quant_axis dimension value of Input(X) when the `scale` has "
            "only one element, but %ld != %ld here.",
            scale.numel(),
            in_tmp.dims()[quant_axis]));
    int max_range = (std::pow(2, bit_length - 1) - 1);

    ChannelDequantizeFunctorV2<Context, D>()(
        dev_ctx, &in_tmp, &scale, static_cast<D>(max_range), quant_axis, out);
  }
}

template <typename T, typename Context>
void DeQuantizeLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& zero_point,
                            const paddle::optional<DenseTensor>& in_accum,
                            const paddle::optional<DenseTensor>& in_state,
                            int quant_axis,
                            int bit_length,
                            int round_type,
                            bool is_test,
                            bool only_observer,
                            DenseTensor* out,
                            DenseTensor* out_state,
                            DenseTensor* out_accum,
                            DenseTensor* out_scale) {
  switch (scale.dtype()) {
    case phi::DataType::FLOAT64:
      DeQuantizeLinearImpl<T, Context, double>(
          dev_ctx, x, scale, quant_axis, bit_length, only_observer, out);
      break;
    case phi::DataType::FLOAT32:
      DeQuantizeLinearImpl<T, Context, float>(
          dev_ctx, x, scale, quant_axis, bit_length, only_observer, out);
      break;
    case phi::DataType::FLOAT16:
      DeQuantizeLinearImpl<T, Context, float16>(
          dev_ctx, x, scale, quant_axis, bit_length, only_observer, out);
      break;
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "In DeQuantizeLinearKernel, "
          "data type %d for scale/output is not supported ",
          scale.dtype()));
      break;
  }
}

template <typename T, typename Context>
void QuantizeLinearKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& scale,
                          const DenseTensor& zero_point,
                          const paddle::optional<DenseTensor>& in_accum,
                          const paddle::optional<DenseTensor>& in_state,
                          int quant_axis,
                          int bit_length,
                          int round_type,
                          bool is_test,
                          bool only_observer,
                          float moving_rate,
                          DenseTensor* out,
                          DenseTensor* out_state,
                          DenseTensor* out_accum,
                          DenseTensor* out_scale) {
  auto* in = &x;
  auto* in_scale = &scale;
  auto* out = dev_ctx.template Alloc<float>(out);
  // int bit_length = context.Attr<int>("bit_length");
  // int round_type = context.Attr<int>("round_type");
  int bin_cnt = std::pow(2, bit_length - 1) - 1;
  // int quant_axis = context.Attr<int>("quant_axis");
  // bool is_test = context.Attr<bool>("is_test");
  // bool only_observer = context.Attr<bool>("only_observer");
  // auto& dev_ctx = context.template device_context<DeviceContext>();

  if (quant_axis < 0) {
    if (!is_test) {
      // training
      auto* in_accum = &in_accum;
      auto* in_state = &in_state;
      phi::DenseTensor tmp_scale;
      tmp_scale.Resize(common::make_dim(1));
      T* cur_scale_data = dev_ctx.template Alloc<T>(&tmp_scale);

      phi::funcs::FindAbsMaxFunctor<DeviceContext, T>()(
          dev_ctx, in.data<T>(), in.numel(), cur_scale_data);

      auto* out_state = dev_ctx.template Alloc<T>(out_state);
      auto* out_accum = dev_ctx.template Alloc<T>(out_accum);
      auto* out_scale = dev_ctx.template Alloc<T>(out_scale);

      FindMovingAverageAbsMaxFunctor<DeviceContext, T>()(dev_ctx,
                                                         *in_accum,
                                                         *in_state,
                                                         cur_scale_data,
                                                         moving_rate,
                                                         out_state,
                                                         out_accum,
                                                         out_scale);
      if (only_observer) {
        framework::TensorCopy(*in, dev_ctx.GetPlace(), dev_ctx, out);
      } else {
        phi::funcs::ClipAndFakeQuantFunctor<DeviceContext, T>()(
            dev_ctx, *in, *out_scale, bin_cnt, round_type, out);
      }
    } else {
      if (only_observer) {
        framework::TensorCopy(*in, dev_ctx.GetPlace(), dev_ctx, out);
      } else {
        phi::funcs::ClipAndFakeQuantFunctor<DeviceContext, T>()(
            dev_ctx, *in, *in_scale, bin_cnt, round_type, out);
      }
    }
  } else {
    if (!is_test) {
      T* out_scale_data = dev_ctx.template Alloc<T>(out_scale);
      FindChannelAbsMaxFunctor<DeviceContext, T>()(
          dev_ctx, *in, quant_axis, out_scale_data);
      if (only_observer) {
        framework::TensorCopy(*in, dev_ctx.GetPlace(), dev_ctx, out);
      } else {
        ChannelClipAndFakeQuantFunctor<DeviceContext, T>()(
            dev_ctx, *in, *out_scale, bin_cnt, round_type, quant_axis, out);
      }
    } else {
      if (only_observer) {
        framework::TensorCopy(*in, dev_ctx.GetPlace(), dev_ctx, out);
      } else {
        ChannelClipAndFakeQuantFunctor<DeviceContext, T>()(
            dev_ctx, *in, *in_scale, bin_cnt, round_type, quant_axis, out);
      }
    }
  }
}

}  // namespace phi
