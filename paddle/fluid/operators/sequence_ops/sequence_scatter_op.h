/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/scatter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
class SequenceScatterOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* ids = ctx.Input<LoDTensor>("Ids");
    auto* updates = ctx.Input<LoDTensor>("Updates");
    auto* out = ctx.Output<Tensor>("Out");

    auto& ids_lod = ids->lod();
    PADDLE_ENFORCE_EQ(ids_lod.empty(), false,
                      platform::errors::InvalidArgument(
                          "Input(Ids) Tensor of SequenceScatter operator does "
                          "not contain LoD information."));

    // Initialize out as same as x
    out->mutable_data<T>(ctx.GetPlace());
    framework::TensorCopySync(*x, ctx.GetPlace(), out);

    auto x_dims = x->dims();
    auto out_dims = out->dims();

    for (int i = 0; i < x_dims.size(); ++i)
      PADDLE_ENFORCE_EQ(x_dims[i], out_dims[i],
                        platform::errors::InvalidArgument(
                            "Input(X) and output(Out) shape of SequenceScatter "
                            "operator do not match. Received input(X)'s shape "
                            "is [%s], output(Out)'s shape is [%s].",
                            x_dims, out_dims));

    size_t slice_size = 1;
    for (int i = 1; i < x_dims.size(); ++i) slice_size *= x_dims[i];

    auto lod_vec = ids_lod[0];
    unsigned int seg = 0;
    for (int i = 0; i < ids->dims()[0]; ++i) {
      PADDLE_ENFORCE_LT(
          seg, lod_vec.size() - 1,
          platform::errors::OutOfRange("The segment index is out of bound in "
                                       "SequenceScatter operator, it must be "
                                       "less than batch size. The segment "
                                       "index is %d, the batch size is %d.",
                                       seg, lod_vec.size()));
      int lower_bound = lod_vec[seg];
      int upper_bound = lod_vec[seg + 1];
      if (i >= lower_bound && i < upper_bound) {
        T* p_out = out->data<T>();
        const T* p_updates = updates->data<T>();
        const int64_t* p_index = ids->data<int64_t>();
        p_out[seg * slice_size + p_index[i]] += p_updates[i];
      } else {
        ++seg;
        --i;
      }
    }
  }
};

template <typename T>
class SequenceScatterGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::Unimplemented("Device dose not match. The "
                                        "SequenceScatterGradientOpKernel can "
                                        "only run on CPU device."));
    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dUpdates = ctx.Output<LoDTensor>(framework::GradVarName("Updates"));
    auto* ids = ctx.Input<LoDTensor>("Ids");
    auto* dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto& ids_lod = ids->lod();

    dX->mutable_data<T>(ctx.GetPlace());
    framework::TensorCopySync(*dOut, ctx.GetPlace(), dX);
    dUpdates->mutable_data<T>(ctx.GetPlace());

    auto dx_dims = dX->dims();
    auto dout_dims = dOut->dims();

    for (int i = 0; i < dx_dims.size(); ++i)
      PADDLE_ENFORCE_EQ(dx_dims[i], dout_dims[i],
                        platform::errors::InvalidArgument(
                            "Input(Out@GRAD) and output(X@GRAD) shape of "
                            "SequenceScatterGradient operator do not match. "
                            "Received input(Out@GRAD)'s shape is [%s], "
                            "output(X@GRAD)'s shape is [%s].",
                            dout_dims, dx_dims));

    size_t slice_size = 1;
    for (int i = 1; i < dx_dims.size(); ++i) slice_size *= dx_dims[i];

    auto lod_vec = ids_lod[0];
    unsigned int seg = 0;

    for (int i = 0; i < ids->dims()[0]; ++i) {
      PADDLE_ENFORCE_LT(
          seg, lod_vec.size() - 1,
          platform::errors::OutOfRange(
              "The segment index is out of bound in SequenceScatterGradient "
              "operator, it must be less than batch size. The segment index is "
              "%d, the batch size is %d.",
              seg, lod_vec.size()));
      int lower_bound = lod_vec[seg];
      int upper_bound = lod_vec[seg + 1];
      if (i >= lower_bound && i < upper_bound) {
        const T* p_dOut = dOut->data<T>();
        const int64_t* p_index = ids->data<int64_t>();
        T* p_dUpdates = dUpdates->data<T>();
        p_dUpdates[i] = p_dOut[seg * slice_size + p_index[i]];
      } else {
        ++seg;
        --i;
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
