/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"

namespace phi::funcs {

template <typename Context, typename T>
void FindAbsMaxFunctor<Context, T>::operator()(const Context &ctx,
                                               const T *in,
                                               const int num,
                                               T *out) {
  *out = std::abs(*(std::max_element(in + 0, in + num, Compare<T>())));
}

template <typename Context, typename T>
void ClipAndFakeQuantFunctor<Context, T>::operator()(const Context &ctx,
                                                     const DenseTensor &in,
                                                     const DenseTensor &scale,
                                                     const int qmax,
                                                     const int round_type,
                                                     DenseTensor *out) {
  T s = scale.data<T>()[0];
  T inv_s = inverse(s);
  phi::Transform<Context> trans;
  if (round_type == 0) {
    trans(ctx,
          in.data<T>(),
          in.data<T>() + in.numel(),
          ctx.template Alloc<T>(out),
          QuantTensorFunctor<T>(static_cast<T>(qmax), inv_s));
  } else {
    trans(ctx,
          in.data<T>(),
          in.data<T>() + in.numel(),
          ctx.template Alloc<T>(out),
          phi::ClipFunctor<T>(-s, s));
    auto out_e = EigenVector<T>::Flatten(*out);
    out_e.device(*ctx.eigen_device()) = (qmax * inv_s * out_e).round();
  }
}

template <typename Context, typename T>
void FindMovingAverageAbsMaxFunctor<Context, T>::operator()(
    const Context &ctx,
    const DenseTensor &in_accum,
    const DenseTensor &in_state,
    const T *cur_scale,
    const float rate,
    DenseTensor *out_state,
    DenseTensor *out_accum,
    DenseTensor *out_scale) {
  T accum = in_accum.data<T>()[0];
  T state = in_state.data<T>()[0];
  T scale = cur_scale[0];

  state = rate * state + 1;
  accum = rate * accum + scale;
  scale = accum / state;

  T *out_state_data = ctx.template Alloc<T>(out_state);
  T *out_accum_data = ctx.template Alloc<T>(out_accum);
  T *out_scale_data = ctx.template Alloc<T>(out_scale);

  out_state_data[0] = state;
  out_accum_data[0] = accum;
  out_scale_data[0] = scale;
}

template <typename Context, typename T>
void FindChannelAbsMaxFunctor<Context, T>::operator()(
    const Context &ctx,
    const DenseTensor &in_tensor,
    const int quant_axis,
    T *out_abs_max) {
  // At present, channelwise quantization supports conv2d, depthwise_conv2d
  // conv2d_transpose and mul
  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                      "the received is %d",
                                      quant_axis));
  auto *in_data = in_tensor.data<T>();
  auto in_dims = in_tensor.dims();
  const int64_t channel = in_dims[quant_axis];
  if (quant_axis == 0) {
    const int64_t channel_size = in_tensor.numel() / channel;
    for (int64_t i = 0; i < channel; i++) {
      auto *start = in_data + i * channel_size;
      auto *end = in_data + (i + 1) * channel_size;
      out_abs_max[i] = std::abs(*(std::max_element(start, end, Compare<T>())));
    }
  } else if (quant_axis == 1) {
    for (int64_t i = 0; i < channel; i++) {
      out_abs_max[i] = 0;
    }
    const int64_t step_i = in_tensor.numel() / in_dims[0];
    const int64_t step_j = in_tensor.numel() / (in_dims[0] * in_dims[1]);
    for (int64_t i = 0; i < in_dims[0]; i++) {
      for (int64_t j = 0; j < in_dims[1]; j++) {
        auto *start = in_data + i * step_i + j * step_j;
        auto *end = in_data + i * step_i + (j + 1) * step_j;
        T abs_max = std::abs(*(std::max_element(start, end, Compare<T>())));
        out_abs_max[j] = std::max(out_abs_max[j], abs_max);
      }
    }
  }
}

template <typename Context, typename T>
void ChannelClipAndFakeQuantFunctor<Context, T>::operator()(
    const Context &ctx,
    const DenseTensor &in,
    const DenseTensor &scale,
    const int qmax,
    const int round_type,
    const int quant_axis,
    DenseTensor *out) {
  // At present, channelwise quantization supports conv2d, depthwise_conv2d
  // conv2d_transpose and mul
  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                      "the received is %d",
                                      quant_axis));
  auto *scale_data = scale.data<T>();
  auto *in_data = in.data<T>();
  auto *out_data = ctx.template Alloc<T>(out);
  auto in_dims = in.dims();
  const int64_t channel = in_dims[quant_axis];
  phi::Transform<Context> trans;
  if (quant_axis == 0) {
    const int64_t channel_size = in.numel() / channel;
    for (int64_t i = 0; i < channel; i++) {
      T s = scale_data[i];
      auto *start = in_data + i * channel_size;
      auto *end = in_data + (i + 1) * channel_size;
      T inv_s = inverse(s);
      if (round_type == 0) {
        trans(ctx,
              start,
              end,
              out_data + i * channel_size,
              QuantTensorFunctor<T>(static_cast<T>(qmax), inv_s));
      } else {
        trans(ctx,
              start,
              end,
              out_data + i * channel_size,
              ClipFunctor<T>(-s, s));
      }
    }
    if (round_type == 1) {
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_data[i];
        T inv_s = inverse(s);
        DenseTensor one_channel_out = out->Slice(i, i + 1);
        auto out_e = EigenVector<T>::Flatten(one_channel_out);
        out_e.device(*ctx.eigen_device()) = (qmax * inv_s * out_e).round();
      }
    }
  } else if (quant_axis == 1) {
    const int64_t step_i = in.numel() / in_dims[0];
    const int64_t step_j = in.numel() / (in_dims[0] * in_dims[1]);
    for (int i = 0; i < in_dims[0]; i++) {
      for (int j = 0; j < in_dims[1]; j++) {
        T s = scale_data[j];
        T inv_s = inverse(s);
        auto *start = in_data + i * step_i + j * step_j;
        auto *end = in_data + i * step_i + (j + 1) * step_j;
        auto *cur_out_data = out_data + i * step_i + j * step_j;
        if (round_type == 0) {
          trans(ctx,
                start,
                end,
                cur_out_data,
                QuantTensorFunctor<T>(static_cast<T>(qmax), inv_s));
        } else {
          trans(ctx, start, end, cur_out_data, ClipFunctor<T>(-s, s));
          for (int k = 0; k < step_j; k++) {
            cur_out_data[k] = std::round(qmax * inv_s * cur_out_data[k]);
          }
        }
      }
    }
  }
}

template <typename Context, typename T>
void ChannelClipFakeQuantDequantFunctor<Context, T>::operator()(
    const Context &ctx,
    const DenseTensor &in,
    const DenseTensor &scale,
    const int bin_cnt,
    const int round_type,
    const int quant_axis,
    DenseTensor *out) {
  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                      "the received is %d",
                                      quant_axis));

  auto *scale_data = scale.data<T>();
  auto *in_data = in.data<T>();
  auto *out_data = ctx.template Alloc<T>(out);
  auto in_dims = in.dims();
  const int64_t channel = in_dims[quant_axis];
  phi::Transform<Context> trans;
  if (quant_axis == 0) {
    const int64_t channel_size = in.numel() / channel;
    for (int i = 0; i < channel; i++) {
      T s = scale_data[i];
      auto *start = in_data + i * channel_size;
      auto *end = in_data + (i + 1) * channel_size;
      if (round_type == 0) {
        T inv_s = inverse(s);
        trans(ctx,
              start,
              end,
              out_data + i * channel_size,
              QuantTensorFunctor<T>(static_cast<T>(bin_cnt), inv_s));
      } else {
        trans(ctx,
              start,
              end,
              out_data + i * channel_size,
              ClipFunctor<T>(-s, s));
      }
    }
    for (int i = 0; i < channel; i++) {
      T s = scale_data[i];
      DenseTensor one_channel_out = out->Slice(i, i + 1);
      auto out_e = EigenVector<T>::Flatten(one_channel_out);
      if (round_type == 0) {
        out_e.device(*ctx.eigen_device()) = out_e * s / static_cast<T>(bin_cnt);
      } else {
        T inv_s = inverse(s);
        out_e.device(*ctx.eigen_device()) =
            (bin_cnt * inv_s * out_e).round() * s / static_cast<T>(bin_cnt);
      }
    }
  } else if (quant_axis == 1) {
    const int64_t step_i = in.numel() / in_dims[0];
    const int64_t step_j = in.numel() / (in_dims[0] * in_dims[1]);
    for (int i = 0; i < in_dims[0]; i++) {
      for (int j = 0; j < in_dims[1]; j++) {
        T s = scale_data[j];
        T inv_s = inverse(s);
        auto *start = in_data + i * step_i + j * step_j;
        auto *end = in_data + i * step_i + (j + 1) * step_j;
        auto *cur_out_data = out_data + i * step_i + j * step_j;
        if (round_type == 0) {
          trans(ctx,
                start,
                end,
                cur_out_data,
                QuantTensorFunctor<T>(static_cast<T>(bin_cnt), inv_s));
        } else {
          trans(ctx, start, end, cur_out_data, ClipFunctor<T>(-s, s));
        }
        for (int k = 0; k < step_j; k++) {
          if (round_type == 0) {
            cur_out_data[k] = cur_out_data[k] * s / static_cast<T>(bin_cnt);
          } else {
            cur_out_data[k] = std::round(bin_cnt * inv_s * cur_out_data[k]) *
                              s / static_cast<T>(bin_cnt);
          }
        }
      }
    }
  }
}

template <typename Context, typename T>
void FindRangeAbsMaxFunctor<Context, T>::operator()(
    const Context &ctx,
    const DenseTensor &cur_scale,
    const DenseTensor &last_scale,
    const DenseTensor &iter,
    const int window_size,
    DenseTensor *scales_arr,
    DenseTensor *out_scale) {
  T *scale_arr_data = ctx.template Alloc<T>(scales_arr);
  int64_t it = iter.data<int64_t>()[0];
  int idx = static_cast<int>(it % window_size);
  T removed = scale_arr_data[idx];
  T cur = cur_scale.data<T>()[0];
  scale_arr_data[idx] = cur;

  T max = last_scale.data<T>()[0];
  if (max < cur) {
    max = cur;
  } else if (fabs(removed - max) < 1e-6) {
    int size = static_cast<int>((it > window_size) ? window_size : it);
    phi::funcs::FindAbsMaxFunctor<Context, T>()(
        ctx, scale_arr_data, size, &max);
  }
  T *out_scale_data = ctx.template Alloc<T>(out_scale);
  out_scale_data[0] = max;
}

template <typename Context, typename T>
void ClipAndFakeQuantDequantFunctor<Context, T>::operator()(
    const Context &ctx,
    const DenseTensor &in,
    const DenseTensor &scale,
    const int bin_cnt,
    int round_type,
    DenseTensor *out) {
  T s = scale.data<T>()[0];
  T inv_s = phi::funcs::inverse(s);

  phi::Transform<CPUContext> trans;
  if (round_type == 0) {
    trans(ctx,
          in.data<T>(),
          in.data<T>() + in.numel(),
          ctx.template Alloc<T>(out),
          phi::funcs::QuantTensorFunctor<T>(static_cast<T>(bin_cnt), inv_s));
    auto out_e = phi::EigenVector<T>::Flatten(*out);
    out_e.device(*ctx.eigen_device()) = out_e * s / static_cast<T>(bin_cnt);
  } else {
    trans(ctx,
          in.data<T>(),
          in.data<T>() + in.numel(),
          ctx.template Alloc<T>(out),
          phi::ClipFunctor<T>(-s, s));
    auto out_e = phi::EigenVector<T>::Flatten(*out);
    out_e.device(*ctx.eigen_device()) =
        (bin_cnt * inv_s * out_e).round() * s / static_cast<T>(bin_cnt);
  }
}

template class FindAbsMaxFunctor<CPUContext, float>;
template class ClipAndFakeQuantFunctor<CPUContext, float>;
template class FindMovingAverageAbsMaxFunctor<CPUContext, float>;
template class FindChannelAbsMaxFunctor<CPUContext, float>;
template class ChannelClipAndFakeQuantFunctor<CPUContext, float>;
template class ChannelClipFakeQuantDequantFunctor<CPUContext, float>;
template class FindRangeAbsMaxFunctor<CPUContext, float>;
template class ClipAndFakeQuantDequantFunctor<CPUContext, float>;

}  // namespace phi::funcs
