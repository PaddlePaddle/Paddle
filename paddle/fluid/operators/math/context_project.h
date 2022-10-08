/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

namespace math {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;

/*
 * \brief Context projection concatenates features in adjacent time-steps in
 * a sequence. The i-th row of the output is the concatenation of
 * context_length rows of the input. The context_length rows are the
 * consecutive rows from the i+shift_start row.
 * ContextProjectGradFunctor is the inverse process of ContextProjectFunctor.
 *
 * \param in            Input data.
 * \param Shape         The shape of Input data:
 *                        [mini-batch, input_hidden_size].
 *
 * \param padding_data  Padding data.
 * \param Shape         The shape of Padding data:
 *                        [up_pad + down_pad, input_hidden_size].
 *
 * \param col           Col data.
 * \param Shape         The shape of Col data:
 *                        [mini-batch, context_length * input_hidden_size].
 *
 * For a mini-batch of 2 variable lengths sentences, containing 3, and 1
 * time-steps:
 *
 * Assumed input (X) is a [4, M, N] float LoDTensor, and X->lod()[0] = [0, 3,
 * 4].
 * Besides, for the sake of simplicity, we assume M=1 and N=2.
 *
 * X = [[a1, a2;
 *       b1, b2;
 *       c1, c2]
 *      [d1, d2]]
 *
 * This is to say that input (X) has 4 words and the dimension of each word
 * representation is 2.
 *
 * - Case1:
 *   If context_start is -1 and padding_trainable is false, we use zero to pad
 *   instead of learned weight to pad,
 *   and the context_length is 3, the output (Out) is:
 *
 *   Out =[[0,  0,  a1, a2, b1, b2;
 *          a1, a2, b1, b2, c1, c2;
 *          b1, b2, c1, c2, 0,  0 ]
 *          [0,  0, d1, d2, 0,  0 ]]
 *
 * - Case2:
 *   If context_start is -1 and padding_trainable is true, we use learned weight
 *   to pad,
 *   and the context_length is 3, the output (Out) is:
 *
 *   Out = [[w1, w2, a1, a2, b1, b2;
 *           a1, a2, b1, b2, c1, c2;
 *           b1, b2, c1, c2, w3, w4]
 *          [w1, w2, d1, d2, w3, w4]]
 *
 */

template <typename DeviceContext, typename T>
class ContextProjectFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const LoDTensor& in,
                  const phi::DenseTensor* padding_data,
                  bool padding_trainable,
                  const int context_start,
                  const int context_length,
                  const int context_stride,
                  const int up_pad,
                  const int down_pad,
                  phi::DenseTensor* col) {
    auto lod_level_0 = in.lod()[0];

    math::Im2ColFunctor<math::ColFormat::kOCF, DeviceContext, float> im2col_ocf;

    std::vector<int> dilation({1, 1});
    std::vector<int> padding({up_pad, 0, down_pad, 0});
    std::vector<int> stride({context_stride, 1});

    int input_row_begin, input_row_end;
    int sequence_height, sequence_width;
    sequence_width = in.dims()[1];

    for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
      if (lod_level_0[i] == lod_level_0[i + 1]) continue;

      input_row_begin = (context_start > 0)
                            ? static_cast<int>(lod_level_0[i]) + context_start
                            : static_cast<int>(lod_level_0[i]);
      input_row_end = static_cast<int>(lod_level_0[i + 1]);

      Tensor out_t = col->Slice(static_cast<int>(lod_level_0[i]),
                                static_cast<int>(lod_level_0[i + 1]));

      sequence_height = static_cast<int>(out_t.dims()[0]);

      if (input_row_begin < input_row_end) {
        Tensor in_t = in.Slice(input_row_begin, input_row_end);

        std::vector<int64_t> output_shape(
            {sequence_height,
             1,
             1,
             context_length,
             sequence_width});  // output_height, output_width,
        // input_channels, filter_height, filter_width
        out_t.Resize(phi::make_ddim(output_shape));

        std::vector<int64_t> input_shape(
            {1,
             input_row_end - input_row_begin,
             sequence_width});  // input_channels, input_height, input_width
        in_t.Resize(phi::make_ddim(input_shape));
        im2col_ocf(context, in_t, dilation, stride, padding, &out_t);
        out_t.Resize({sequence_height, context_length * sequence_width});
      }
    }
    if (padding_trainable) {
      PADDLE_ENFORCE_NOT_NULL(
          padding_data,
          platform::errors::InvalidArgument(
              "The input tensor 'padding_data' should not be NULL."));
      for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
        if (lod_level_0[i] == lod_level_0[i + 1]) continue;

        Tensor out_t = col->Slice(static_cast<int>(lod_level_0[i]),
                                  static_cast<int>(lod_level_0[i + 1]));

        sequence_height = static_cast<int>(out_t.dims()[0]);

        // add up trainable data
        out_t.Resize({static_cast<int64_t>(sequence_height) * context_length,
                      sequence_width});

        if (up_pad > 0) {  // add up pad
          int padding_rows = std::min(
              up_pad, static_cast<int>(lod_level_0[i + 1] - lod_level_0[i]));

          for (int k = 0; k < padding_rows; ++k) {
            int padding_size =
                k + context_length < up_pad ? context_length : up_pad - k;
            Tensor out_t_sub = out_t.Slice(k * context_length,
                                           k * context_length + padding_size);
            Tensor w_sub = padding_data->Slice(k, k + padding_size);
            framework::TensorCopy(
                w_sub, context.GetPlace(), context, &out_t_sub);
          }
        }
        if (down_pad > 0) {  // add down pad
          int down_pad_begin_row =
              std::max(0,
                       (sequence_height - context_start - context_length) + 1) +
              1;
          int padding_begin = std::max(0, context_start - sequence_height);
          int padding_size =
              sequence_height - context_start >= context_length
                  ? 1
                  : context_length - (sequence_height - context_start);
          if (context_start >= sequence_height) padding_size = context_length;
          int padding_idx = padding_begin;
          for (int t = 0; t + down_pad_begin_row <= sequence_height;
               ++t, ++padding_size) {
            if (context_start >= sequence_height) padding_size = context_length;
            if (padding_size > context_length) {
              padding_size = context_length;
              padding_idx++;
            }
            if (padding_begin > 0 || sequence_height == context_start)
              padding_idx = padding_begin + t;

            Tensor out_t_sub = out_t.Slice(
                (down_pad_begin_row + t) * context_length - padding_size,
                (down_pad_begin_row + t) * context_length);
            Tensor w_sub = padding_data->Slice(
                up_pad + padding_idx, up_pad + padding_idx + padding_size);
            framework::TensorCopy(
                w_sub, context.GetPlace(), context, &out_t_sub);
          }
        }
        out_t.Resize({sequence_height,
                      static_cast<int64_t>(context_length) * sequence_width});
      }
    }
  }
};

template <typename DeviceContext, typename T>
class ContextProjectGradFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const LoDTensor& in,
                  bool padding_trainable,
                  const int context_start,
                  const int context_length,
                  const int context_stride,
                  const int up_pad,
                  const int down_pad,
                  bool pad_grad,
                  bool input_grad,
                  phi::DenseTensor* padding_data,
                  phi::DenseTensor* col) {
    auto lod_level_0 = in.lod()[0];

    math::Col2ImFunctor<math::ColFormat::kOCF, DeviceContext, float> col2im_ocf;

    std::vector<int> dilation({1, 1});
    std::vector<int> padding({up_pad, 0, down_pad, 0});
    std::vector<int> stride({context_stride, 1});

    int input_row_begin, input_row_end;
    int sequence_height, sequence_width;
    sequence_width = in.dims()[1];
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);

    if (input_grad) {
      for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
        if (lod_level_0[i] == lod_level_0[i + 1]) continue;

        input_row_begin = (context_start > 0)
                              ? static_cast<int>(lod_level_0[i]) + context_start
                              : static_cast<int>(lod_level_0[i]);
        input_row_end = static_cast<int>(lod_level_0[i + 1]);

        Tensor out_t = col->Slice(static_cast<int>(lod_level_0[i]),
                                  static_cast<int>(lod_level_0[i + 1]));

        sequence_height = static_cast<int>(out_t.dims()[0]);

        if (input_row_begin < input_row_end) {
          Tensor in_t = in.Slice(input_row_begin, input_row_end);

          std::vector<int64_t> output_shape(
              {sequence_height,
               1,
               1,
               context_length,
               sequence_width});  // output_height, output_width,
          // input_channels, filter_height, filter_width
          out_t.Resize(phi::make_ddim(output_shape));

          std::vector<int64_t> input_shape(
              {1,
               input_row_end - input_row_begin,
               sequence_width});  // input_channels, input_height, input_width
          in_t.Resize(phi::make_ddim(input_shape));

          col2im_ocf(context, out_t, dilation, stride, padding, &in_t);
          out_t.Resize({sequence_height, context_length * sequence_width});
        }
      }
    }
    if (pad_grad) {
      if (padding_trainable) {
        for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
          if (lod_level_0[i] == lod_level_0[i + 1]) continue;

          Tensor out_t = col->Slice(static_cast<int>(lod_level_0[i]),
                                    static_cast<int>(lod_level_0[i + 1]));

          sequence_height = static_cast<int>(out_t.dims()[0]);
          out_t.Resize({static_cast<int64_t>(sequence_height) * context_length,
                        sequence_width});

          if (up_pad > 0) {
            int padding_rows = std::min(
                up_pad, static_cast<int>(lod_level_0[i + 1] - lod_level_0[i]));

            for (int k = 0; k < padding_rows; ++k) {
              int padding_size =
                  k + context_length < up_pad ? context_length : up_pad - k;
              Tensor out_t_sub = out_t.Slice(k * context_length,
                                             k * context_length + padding_size);
              Tensor w_sub = padding_data->Slice(k, k + padding_size);
              blas.AXPY(w_sub.numel(),
                        static_cast<T>(1),
                        out_t_sub.data<T>(),
                        w_sub.data<T>());
            }
          }
          if (down_pad > 0) {
            int down_pad_begin_row =
                std::max(
                    0, (sequence_height - context_start - context_length) + 1) +
                1;
            int padding_begin = std::max(0, context_start - sequence_height);
            int padding_size =
                sequence_height - context_start >= context_length
                    ? 1
                    : context_length - (sequence_height - context_start);
            if (context_start >= sequence_height) padding_size = context_length;
            int padding_idx = padding_begin;
            for (int t = 0; t + down_pad_begin_row <= sequence_height;
                 ++t, ++padding_size) {
              if (context_start >= sequence_height)
                padding_size = context_length;
              if (padding_size > context_length) {
                padding_size = context_length;
                padding_idx++;
              }
              if (padding_begin > 0 || sequence_height == context_start)
                padding_idx = padding_begin + t;

              Tensor out_t_sub = out_t.Slice(
                  (down_pad_begin_row + t) * context_length - padding_size,
                  (down_pad_begin_row + t) * context_length);
              Tensor w_sub = padding_data->Slice(
                  up_pad + padding_idx, up_pad + padding_idx + padding_size);
              blas.AXPY(w_sub.numel(),
                        static_cast<T>(1),
                        out_t_sub.data<T>(),
                        w_sub.data<T>());
            }
          }
          out_t.Resize({sequence_height,
                        static_cast<int64_t>(context_length) * sequence_width});
        }
      }
    }
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
