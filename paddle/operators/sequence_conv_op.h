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
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/sequence_project.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
// template <typename T, int MajorType = Eigen::RowMajor,
//          typename IndexType = Eigen::DenseIndex>
// using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class SequenceConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    auto filter = *context.Input<Tensor>("Filter");

    out->mutable_data<T>(context.GetPlace());
    //  out->set_lod(in->lod());

    int context_start = context.Attr<int>("context_start");
    int context_length = context.Attr<int>("context_length");
    int context_stride = context.Attr<int>("context_stride");
    bool padding_trainable = context.Attr<bool>("padding_trainable");

    // InferShape by in_lod
    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");

    const Tensor* padding_data = nullptr;
    if (padding_trainable) {
      padding_data = context.Input<Tensor>("PaddingData");
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_width;
    sequence_width = static_cast<int>(in->dims()[1]);

    // use col_shape in the im2col calculation
    framework::DDim col_shape = {in->dims()[0],
                                 sequence_width * context_length};
    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // Because if padding_trainable is false, padding data should be zeros.
    auto temp = framework::EigenVector<T>::Flatten(col);
    temp.device(context.GetEigenDevice<Place>()) =
        temp.constant(static_cast<T>(0));

    paddle::operators::math::SequenceProjectFunctor<Place, T>
        seq_project_functor;
    LoDTensor* input = const_cast<LoDTensor*>(in);
    Tensor* pad_data = const_cast<Tensor*>(padding_data);

    seq_project_functor(context.device_context(), *input, *pad_data, col,
                        padding_trainable, context_start, context_length,
                        context_stride, up_pad, down_pad, false, false, false);

    filter.Resize(framework::make_ddim({context_length * sequence_width, 1}));
    math::matmul<Place, T>(context.device_context(), col, false, filter, false,
                           T(1.0), out, T(0.0));
  }
};

template <typename Place, typename T>
class SequenceConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* filter_g = context.Output<Tensor>(framework::GradVarName("Filter"));
    auto* padding_data_g =
        context.Output<Tensor>(framework::GradVarName("PaddingData"));
    auto* in = context.Input<LoDTensor>("X");
    auto* filter = context.Input<Tensor>("Filter");

    int context_start = context.Attr<int>("context_start");
    int context_length = context.Attr<int>("context_length");
    int context_stride = context.Attr<int>("context_stride");
    bool padding_trainable = context.Attr<bool>("padding_trainable");

    // InferShape by in_lod
    PADDLE_ENFORCE_EQ(in->lod().size(), 1UL,
                      "Only support one level sequence now.");
    auto lod_g_level_0 = in->lod()[0];

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int sequence_width = static_cast<int>(in->dims()[1]);

    // use col_shape in the im2col calculation
    framework::DDim col_shape = {in->dims()[0],
                                 sequence_width * context_length};
    Tensor col;

    if (in_g || filter_g || (padding_trainable && padding_data_g)) {
      col.mutable_data<T>(col_shape, context.GetPlace());
      // Because if padding_trainable is false, padding data should be zeros.
      auto temp = framework::EigenVector<T>::Flatten(col);
      temp.device(context.GetEigenDevice<Place>()) =
          temp.constant(static_cast<T>(0));

      math::matmul<Place, T>(context.device_context(), *out_g, false, *filter,
                             true, T(1.0), &col, T(1.0));
    }
    paddle::operators::math::SequenceProjectFunctor<Place, T>
        seq_project_functor;

    if (in_g) {
      in_g->mutable_data<T>(context.GetPlace());
      in_g->set_lod(in->lod());

      math::SetConstant<Place, T> functor;
      functor(context.device_context(), in_g, 0);

      seq_project_functor(context.device_context(), *in_g, *padding_data_g, col,
                          padding_trainable, context_start, context_length,
                          context_stride, up_pad, down_pad, true, true, false);
    }

    if (padding_trainable && padding_data_g) {
      padding_data_g->mutable_data<T>(context.GetPlace());

      math::SetConstant<Place, T> functor;
      functor(context.device_context(), padding_data_g, 0);

      LoDTensor* input = const_cast<LoDTensor*>(in);
      seq_project_functor(context.device_context(), *input, *padding_data_g,
                          col, padding_trainable, context_start, context_length,
                          context_stride, up_pad, down_pad, true, false, true);
    }

    if (filter_g) {
      filter_g->mutable_data<T>(context.GetPlace());

      math::SetConstant<Place, T> functor;
      functor(context.device_context(), filter_g, 0);

      Tensor filter_grad_ = *filter_g;
      LoDTensor out_grad_ = *out_g;

      const Tensor* padding_data = nullptr;
      if (padding_trainable) {
        padding_data = context.Input<Tensor>("PaddingData");
      }

      sequence_width = static_cast<int>(in->dims()[1]);

      LoDTensor* input = const_cast<LoDTensor*>(in);
      Tensor* pad_data = const_cast<Tensor*>(padding_data);

      seq_project_functor(context.device_context(), *input, *pad_data, col,
                          padding_trainable, context_start, context_length,
                          context_stride, up_pad, down_pad, false, false,
                          false);

      filter_grad_.Resize(
          framework::make_ddim({context_length * sequence_width, 1}));

      math::matmul<Place, T>(context.device_context(), col, true, out_grad_,
                             false, T(1.0), &filter_grad_, T(1.0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
