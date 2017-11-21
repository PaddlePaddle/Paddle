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

#include "paddle/framework/op_registry.h"
#include "paddle/memory/memcpy.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename Place, typename T>
class SeqExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    const T* x_data = x->data<T>();
    auto x_dims = x->dims();
    auto* y = context.Input<LoDTensor>("Y");
    PADDLE_ENFORCE_EQ(static_cast<size_t>(x_dims[0]),
                      y->lod().back().size() - 1,
                      "The size of last lod level in Input(Y)"
                      "must be equal to dims[0] of Input(X).");
    out->set_lod(y->lod());
    auto place = context.GetEigenDevice<Place>();
    size_t element_len = framework::product(x_dims) / x_dims[0];
    T* out_data = out->mutable_data<T>(context.GetPlace());
    auto out_starts = out->lod().back();

    for (size_t i = 0; i < out_starts.size() - 1; i++) {
      int scale = out_starts[i + 1] - out_starts[i];
      Eigen::TensorMap<
          Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>>
          x_t(x_data, 1, element_len);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>>
          out_t(out_data, scale, element_len);
      Eigen::array<int, 2> cast({{scale, 1}});
      out_t.device(place) = x_t.broadcast(cast);
      x_data += element_len;
      out_data += element_len * scale;
    }
  }
};

/*
 *Given Grad(Out)
 *
 *    Grad(Out).lod = [[0,                            2],
 *                     [0,              3,            6]]
 *    Grad(Out).data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
 * Then
 *    Grad(X).data = [(0.1 + 0.2 + 0.3), (0.4 + 0.5 + 0.6)]
 *                 = [0.6, 1.5]
 *    Grad(X).lod = Input(X).lod
 *
 * */
template <typename Place, typename T>
class SeqExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* d_out = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x = context.Input<LoDTensor>("X");
    auto* out = context.Input<LoDTensor>("Out");
    auto* d_x = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto out_last_level = out->lod().back();
    d_x->set_lod(x->lod());
    const T* d_out_data = d_out->data<T>();
    T* d_x_data = d_x->mutable_data<T>(context.GetPlace());
    size_t element_len = d_out->numel() / d_out->dims()[0];
    for (size_t i = 0; i < out_last_level.size() - 1; ++i) {
      size_t repeat = out_last_level[i + 1] - out_last_level[i];
      Eigen::TensorMap<
          Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>>
      d_out_t(d_out_data, static_cast<int>(repeat), element_len);
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>
      d_x_t(d_x_data, static_cast<int>(element_len));
      auto place = context.GetEigenDevice<Place>();
      d_x_t.device(place) = d_out_t.sum(Eigen::array<int, 1>({{0}}));
      d_out_data += (repeat * element_len);
      d_x_data += element_len;
    }
  }
};

}  // namespace operators
}  // namespace paddle
