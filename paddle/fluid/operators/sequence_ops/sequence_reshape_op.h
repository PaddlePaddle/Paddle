//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
template <typename DeviceContext, typename T>
class SequenceReshapeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int out_width = context.Attr<int>("new_dim");

    auto in_dims = in->dims();
    int64_t in_width = in_dims[1];
    auto& in_lod = in->lod();

    PADDLE_ENFORCE_EQ(in_lod.empty(), false,
                      platform::errors::NotFound(
                          "Input(X) Tensor of SequenceReshapeOp does not "
                          "contain LoD information."));
    PADDLE_ENFORCE_EQ(in_lod.size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Input(X) Tensor of SequenceReshapeOp Only support "
                          "one level sequence now. But lod size "
                          "of Input(X) is %d",
                          in_lod.size()));
    PADDLE_ENFORCE_EQ(
        (uint64_t)in_dims[0], in_lod[0].back(),
        platform::errors::InvalidArgument(
            "The size of SequenceReshapeOp X.shape[0] and X.lod()[0].back() "
            "should "
            "be same. But X.shape[0] = %d, X.lod()[0].back() = %d",
            (uint64_t)in_dims[0], in_lod[0].back()));

    auto in_lod_l0 = in_lod[0];
    int seq_num = in_lod_l0.size() - 1;

    if (in_width == out_width) {
      out->set_lod(in->lod());
    } else {
      auto& out_lod = *out->mutable_lod();
      out_lod.resize(1);
      out_lod[0].resize(seq_num + 1);
      out_lod[0][0] = 0;
      for (int i = 0; i < seq_num; ++i) {
        size_t seq_len = in_lod_l0[i + 1] - in_lod_l0[i];
        size_t offset = 0;
        offset = (seq_len * in_width) / out_width;
        PADDLE_ENFORCE_EQ(
            offset * out_width, seq_len * in_width,
            platform::errors::InvalidArgument(
                "Please make sure (sequence_length * dimension) "
                "can be divided by context Attr(new_dim) with no remainder for "
                "each sequence. But the %dth sequence is invalid.",
                i + 1));
        out_lod[0][i + 1] = out_lod[0][i] + offset;
      }
    }

    framework::TensorCopy(*in, context.GetPlace(), out);
    out->Resize({static_cast<int64_t>(out->lod()[0].back()), out_width});
  }
};

template <typename DeviceContext, typename T>
class SequenceReshapeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x_tensor_ptr = context.Input<LoDTensor>("X");
    auto* outg_tensor_ptr =
        context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* xg_tensor_ptr =
        context.Output<LoDTensor>(framework::GradVarName("X"));

    xg_tensor_ptr->mutable_data<T>(context.GetPlace());
    framework::TensorCopy(*outg_tensor_ptr, context.GetPlace(), xg_tensor_ptr);
    xg_tensor_ptr->Resize(x_tensor_ptr->dims());
  }
};

}  // namespace operators
}  // namespace paddle
