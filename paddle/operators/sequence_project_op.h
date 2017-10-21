/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/im2col.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class SequenceProjectKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    // need discuss, is it necessary to set zeros ?
    // Because if padding_trainable is false, padding data should be zeros.
    auto temp = framework::EigenVector<T>::Flatten(*out);
    temp.device(context.GetEigenDevice<Place>()) =
        temp.constant(static_cast<T>(0));

    auto place = context.GetEigenDevice<Place>();

    int context_start = context.Attr<int>("context_start");
    int context_length = context.Attr<int>("context_length");
    bool padding_trainable = context.Attr<bool>("padding_trainable");
    int context_stride = context.Attr<int>("context_stride");

    // InferShape by in_lod
    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_level_0 = in->lod()[0];
    int64_t input_width = in->dims()[1];
    int64_t output_width = out->dims()[1];
    int64_t padding_width = 0;
    PADDLE_ENFORCE(input_width * context_length == output_width,
                   "Input size and pooling size should be consistent.");

    const LoDTensor* padding_data = nullptr;
    if (padding_trainable) {
      padding_data = context.Input<LoDTensor>("PaddingData");
      PADDLE_ENFORCE_EQ(padding_data->dims().size(), 2UL,
                        "Only support one level sequence now.");
      padding_width = padding_data->dims()[1];
      PADDLE_ENFORCE(padding_width == input_width,
                     "Input size and pooling size should be consistent.");
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_height, sequence_width;
    int input_row_begin, input_row_end;

    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kOCF, Place, float>
        im2col_ocf;

    for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
      input_row_begin = (context_start > 0)
                            ? static_cast<int>(lod_level_0[i]) + context_start
                            : static_cast<int>(lod_level_0[i]);
      input_row_end = static_cast<int>(lod_level_0[i + 1]);

      Tensor out_t = out->Slice(static_cast<int>(lod_level_0[i]),
                                static_cast<int>(lod_level_0[i + 1]));

      sequence_height = static_cast<int>(out_t.dims()[0]);
      sequence_width = static_cast<int>(in->dims()[1]);

      std::vector<int64_t> output_shape(
          {sequence_height, 1, 1, context_length,
           sequence_width});  // output_height, output_width,
      // input_channels, filter_height, filter_width
      out_t.Resize(framework::make_ddim(output_shape));

      if (input_row_begin < input_row_end) {
        Tensor in_t = in->Slice(input_row_begin, input_row_end);
        std::vector<int64_t> input_shape(
            {1, input_row_end - input_row_begin,
             sequence_width});  // input_channels, input_height, input_width
        in_t.Resize(framework::make_ddim(input_shape));

        im2col_ocf(context.device_context(), in_t, out_t,
                   /*stride_height*/ context_stride, /*stride_width*/ 0, up_pad,
                   down_pad);
      }

      if (padding_trainable) {
        // add up trainable data
        out_t.Resize(framework::make_ddim(
            {sequence_height * context_length, sequence_width}));

        if (up_pad > 0) {  // add up pad
          int padding_rows = std::min(
              up_pad, static_cast<int>(lod_level_0[i + 1] - lod_level_0[i]));

          for (int k = 0; k < padding_rows; ++k) {
            int padding_size =
                k + context_length < up_pad ? context_length : up_pad - k;
            Tensor out_t_sub = out_t.Slice(k * context_length,
                                           k * context_length + padding_size);
            Tensor w_sub = padding_data->Slice(k, k + padding_size);
            // in this block, using EigenVector<T>::Flatten is ok too.
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            out_t_sub_e.device(place) = w_sub_e;
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
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            out_t_sub_e.device(place) = w_sub_e;
          }
        }
      }
      out_t.Resize(framework::make_ddim(
          {sequence_height, context_length * sequence_width}));
    }
  }
};

template <typename Place, typename T>
class SequenceProjectGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* in = context.Input<LoDTensor>("X");
    in_g->mutable_data<T>(context.GetPlace());
    if (in_g) {
      math::SetConstant<Place, T> functor;
      functor(context.device_context(), in_g, 0);
    }
    auto place = context.GetEigenDevice<Place>();

    int context_start = context.Attr<int>("context_start");
    int context_length = context.Attr<int>("context_length");
    bool padding_trainable = context.Attr<bool>("padding_trainable");
    int context_stride = context.Attr<int>("context_stride");

    // InferShape by in_lod
    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_g_level_0 = in->lod()[0];
    int64_t input_width = in_g->dims()[1];
    int64_t output_width = out_g->dims()[1];
    int64_t padding_width = 0;
    PADDLE_ENFORCE(input_width * context_length == output_width,
                   "Input size and pooling size should be consistent.");

    LoDTensor* padding_data_g = nullptr;
    if (padding_trainable) {
      padding_data_g =
          context.Output<LoDTensor>(framework::GradVarName("PaddingData"));
      padding_data_g->mutable_data<T>(context.GetPlace());
      PADDLE_ENFORCE_EQ(padding_data_g->dims().size(), 2UL,
                        "Only support one level sequence now.");
      padding_width = padding_data_g->dims()[1];
      PADDLE_ENFORCE(padding_width == input_width,
                     "Input size and pooling size should be consistent.");
      math::SetConstant<Place, T> functor;
      functor(context.device_context(), padding_data_g, 0);
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_height, sequence_width;
    int input_row_begin, input_row_end;

    paddle::operators::math::Col2ImFunctor<
        paddle::operators::math::ColFormat::kOCF, Place, float>
        col2im_ocf;

    for (int i = 0; i < static_cast<int>(lod_g_level_0.size()) - 1; ++i) {
      input_row_begin = (context_start > 0)
                            ? static_cast<int>(lod_g_level_0[i]) + context_start
                            : static_cast<int>(lod_g_level_0[i]);
      input_row_end = static_cast<int>(lod_g_level_0[i + 1]);

      Tensor out_g_t = out_g->Slice(static_cast<int>(lod_g_level_0[i]),
                                    static_cast<int>(lod_g_level_0[i + 1]));

      sequence_height = static_cast<int>(out_g_t.dims()[0]);
      sequence_width = static_cast<int>(in_g->dims()[1]);

      if (padding_trainable) {
        // add up trainable data
        out_g_t.Resize(framework::make_ddim(
            {sequence_height * context_length, sequence_width}));

        if (up_pad > 0) {  // add up pad
          int padding_rows = std::min(
              up_pad,
              static_cast<int>(lod_g_level_0[i + 1] - lod_g_level_0[i]));

          for (int k = 0; k < padding_rows; ++k) {
            int padding_size =
                k + context_length < up_pad ? context_length : up_pad - k;
            Tensor out_t_sub = out_g_t.Slice(k * context_length,
                                             k * context_length + padding_size);
            Tensor w_sub = padding_data_g->Slice(k, k + padding_size);
            // in this block, using EigenVector<T>::Flatten is ok too.
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            w_sub_e.device(place) = w_sub_e + out_t_sub_e;
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
            Tensor out_t_sub = out_g_t.Slice(
                (down_pad_begin_row + t) * context_length - padding_size,
                (down_pad_begin_row + t) * context_length);
            Tensor w_sub = padding_data_g->Slice(
                up_pad + padding_idx, up_pad + padding_idx + padding_size);
            auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
            auto w_sub_e = EigenMatrix<T>::From(w_sub);
            w_sub_e.device(place) = w_sub_e + out_t_sub_e;
          }
        }
      }

      if (in_g && input_row_begin < input_row_end) {
        Tensor in_t = in_g->Slice(input_row_begin, input_row_end);

        std::vector<int64_t> output_shape(
            {sequence_height, 1, 1, context_length,
             sequence_width});  // output_height, output_width,
        // input_channels, filter_height, filter_width
        out_g_t.Resize(framework::make_ddim(output_shape));

        std::vector<int64_t> input_shape(
            {1, input_row_end - input_row_begin,
             sequence_width});  // input_channels, input_height, input_width
        in_t.Resize(framework::make_ddim(input_shape));

        col2im_ocf(context.device_context(), in_t, out_g_t,
                   /*stride_height*/ context_stride, /*stride_width*/ 0, up_pad,
                   down_pad);
      }

      out_g_t.Resize(framework::make_ddim(
          {sequence_height, context_length * sequence_width}));
    }
  }
};

}  // namespace operators
}  // namespace paddle
