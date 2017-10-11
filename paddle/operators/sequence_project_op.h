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
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
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
    auto place = context.GetEigenDevice<Place>();

    int context_start = context.Attr<int>("context_start");
    int context_length = context.Attr<int>("context_length");
    bool padding_trainable = context.Attr<bool>("padding_trainable");
    int context_stride = context.Attr<int>("context_stride");

    // InferShape by in_lod
    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_level_0 = in->lod()[0];
    int64_t input_stride = in->dims()[1];
    int64_t output_stride = out->dims()[1];
    int64_t padding_stride = 0;
    PADDLE_ENFORCE(input_stride * context_length == output_stride,
                   "Input size and pooling size should be consistent.");

    const LoDTensor* padding_data = nullptr;
    if (padding_trainable) {
      padding_data = context.Input<LoDTensor>("PaddingData");
      PADDLE_ENFORCE_EQ(padding_data->dims().size(), 2UL,
                        "Only support one level sequence now.");
      padding_stride = padding_data->dims()[1];
      PADDLE_ENFORCE(padding_stride == input_stride,
                     "Input size and pooling size should be consistent.");
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);

    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kOCF, Place, float>
        im2col_ocf;

    for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
      Tensor in_t = in->Slice<T>(static_cast<int>(lod_level_0[i]),
                                 static_cast<int>(lod_level_0[i + 1]));
      Tensor out_t = out->Slice<T>(static_cast<int>(lod_level_0[i]),
                                   static_cast<int>(lod_level_0[i + 1]));

      int sequence_height = in_t.dims()[0];
      int sequence_width = in_t.dims()[1];
      std::vector<int64_t> output_shape(
          {sequence_height, 1, 1, context_length,
           sequence_width});  // output_height, output_width,
                              // input_channels,
                              // filter_height, filter_width
      out_t.Resize(framework::make_ddim(output_shape));
      std::vector<int64_t> input_shape(
          {1, sequence_height,
           sequence_width});  // input_channels, input_height, input_width
      in_t.Resize(framework::make_ddim(input_shape));
      for (int j = 0; j < context_length; ++j) {
        int pad;
        int row_start;

        if (up_pad != 0) {
          pad = up_pad;
          row_start = 0;
        } else if (down_pad != 0) {
          pad = down_pad;
          row_start = down_pad;
        } else {
          pad = 0;
          row_start = 0;
        }

        im2col_ocf(context.device_context(), in_t, out_t,
                   /*stride*/ context_stride, /*pad*/ pad,
                   /*row_start*/ row_start,
                   /*row_end*/ row_start + sequence_height);
        if (padding_trainable) {
          // add up trainable data
          out_t.Resize(framework::make_ddim(
              {sequence_height * context_length, sequence_width}));
          if (up_pad != 0) {
            for (int k = 0; k < up_pad; ++k) {
              Tensor out_t_sub = out_t.Slice<T>(
                  k * context_length, k * context_length + (up_pad - k));
              Tensor w_sub = padding_data->Slice<T>(k, context_length - k);
              auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
              auto w_sub_e = EigenMatrix<T>::From(w_sub);
              out_t_sub_e.device(place) = w_sub_e;
            }
          }
          if (down_pad != 0) {
            int k =
                (sequence_height + up_pad - context_length) / context_stride +
                1;
            for (int t = 0; t + k < sequence_height; ++t) {
              Tensor out_t_sub =
                  out_t.Slice<T>((k + t) * context_length * sequence_width -
                                     t * sequence_width,
                                 (k + t) * context_length * sequence_width);
              Tensor w_sub = padding_data->Slice<T>(up_pad + 1, up_pad + 1 + t);
              auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
              auto w_sub_e = EigenMatrix<T>::From(w_sub);
              out_t_sub_e.device(place) = w_sub_e;
            }
          }
          out_t.Resize(framework::make_ddim(
              {sequence_height, context_length * sequence_width}));
        }
      }
    }
  }
};

template <typename Place, typename T>
class SequenceProjectGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    //    auto* in = context.Input<LoDTensor>("X");
    auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    in_g->mutable_data<T>(context.GetPlace());
    auto place = context.GetEigenDevice<Place>();

    int context_start = context.Attr<int>("context_start");
    int context_length = context.Attr<int>("context_length");
    bool padding_trainable = context.Attr<bool>("padding_trainable");
    int context_stride = context.Attr<bool>("context_stride");

    // InferShape by in_lod
    PADDLE_ENFORCE_EQ(in_g->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_g_level_0 = in_g->lod()[0];
    int64_t input_width = in_g->dims()[1];
    int64_t output_width = out_g->dims()[1];
    int64_t padding_width = 0;
    PADDLE_ENFORCE(input_width * context_length == output_width,
                   "Input size and pooling size should be consistent.");

    LoDTensor* padding_data = nullptr;
    if (padding_trainable) {
      padding_data = context.Output<LoDTensor>("PaddingData");
      padding_data->mutable_data<T>(context.GetPlace());
      PADDLE_ENFORCE_EQ(padding_data->dims().size(), 2UL,
                        "Only support one level sequence now.");
      padding_width = padding_data->dims()[1];
      PADDLE_ENFORCE(padding_width == input_width,
                     "Input size and pooling size should be consistent.");
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);

    paddle::operators::math::Col2ImFunctor<
        paddle::operators::math::ColFormat::kOCF, Place, float>
        col2im_ocf;

    for (int i = 0; i < static_cast<int>(lod_g_level_0.size()) - 1; ++i) {
      Tensor in_g_t = in_g->Slice<T>(static_cast<int>(lod_g_level_0[i]),
                                     static_cast<int>(lod_g_level_0[i + 1]));
      Tensor out_g_t = out_g->Slice<T>(static_cast<int>(lod_g_level_0[i]),
                                       static_cast<int>(lod_g_level_0[i + 1]));

      int sequence_height = in_g_t.dims()[0];
      int sequence_width = in_g_t.dims()[1];

      for (int j = 0; j < context_length; ++j) {
        if (padding_trainable) {
          out_g_t.Resize(framework::make_ddim(
              {sequence_height * context_length, sequence_width}));
          if (up_pad != 0) {
            for (int k = 0; k < up_pad; ++k) {
              Tensor out_t_sub = out_g_t.Slice<T>(
                  k * context_length, k * context_length + (up_pad - k));
              Tensor w_sub = padding_data->Slice<T>(k, context_length - k);
              auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
              auto w_sub_e = EigenMatrix<T>::From(w_sub);
              w_sub_e.device(place) = w_sub_e + out_t_sub_e;
              // out_t_sub_e.device(place) = 0;
            }
          }
          if (down_pad != 0) {
            int k =
                (sequence_height + up_pad - context_length) / context_stride +
                1;
            for (int t = 0; t + k < sequence_height; ++t) {
              Tensor out_t_sub =
                  out_g_t.Slice<T>((k + t) * context_length * sequence_width -
                                       t * sequence_width,
                                   (k + t) * context_length * sequence_width);
              Tensor w_sub = padding_data->Slice<T>(up_pad + 1, up_pad + 1 + t);
              auto out_t_sub_e = EigenMatrix<T>::From(out_t_sub);
              auto w_sub_e = EigenMatrix<T>::From(w_sub);
              w_sub_e.device(place) = w_sub_e + out_t_sub_e;
              // out_t_sub_e.device(place) = 0;
            }
          }
        }
        out_g_t.Resize(framework::make_ddim(
            {sequence_height, 1, 1, context_length, sequence_width}));

        int pad;
        int row_start;

        if (up_pad != 0) {
          pad = up_pad;
          row_start = 0;
        } else if (down_pad != 0) {
          pad = down_pad;
          row_start = down_pad;
        } else {
          pad = 0;
          row_start = 0;
        }
        col2im_ocf(context.device_context(), in_g_t, out_g_t,
                   /*stride*/ context_stride, /*pad*/ pad,
                   /*row_start*/ row_start,
                   /*row_end*/ row_start + sequence_height);

        // out_g_t back to orign size
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
