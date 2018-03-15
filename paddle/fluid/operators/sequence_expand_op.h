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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<LoDTensor>("X");
    auto* y = context.Input<LoDTensor>("Y");
    auto* out = context.Output<LoDTensor>("Out");
    int ref_level = context.Attr<int>("ref_level");

    auto& x_lod = x->lod();
    auto& y_lod = y->lod();

    PADDLE_ENFORCE_GE(ref_level, 0,
                      "Value of attribute `ref_level` should be greater or "
                      "equal to 0.");

    PADDLE_ENFORCE_LT(ref_level, y_lod.size(),
                      "Value of attribute `ref_level` should be smaller than "
                      "level number of Y's lod.");

    if (y_lod[ref_level].size() < 1) {
      framework::TensorCopy(*x, context.GetPlace(), out);
      return;
    }

    if (x_lod.size() == 0) {
      int out_start = 0;
      for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
        int repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
        auto x_sub_tensor = x->Slice(i - 1, i);
        for (size_t j = 0; j < repeat_num; ++j) {
          auto out_sub_tensor = out->Slice(out_start, out_start + 1);
          framework::TensorCopy(x_sub_tensor, context.GetPlace(),
                                &out_sub_tensor);
          out_start++;
        }
      }
    } else {
      auto& out_lod = *out->mutable_lod();
      out_lod.resize(1);
      out_lod[0].resize(1);
      out_lod[0][0] = 0;
      int out_idx = 0;
      for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
        int repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
        int x_seq_len = x_lod[0][i] - x_lod[0][i - 1];
        auto x_sub_tensor = x->Slice(x_lod[0][i], x_lod[0][i - 1]);
        for (size_t j = 0; j < repeat_num; ++j) {
          auto out_sub_tensor =
              out->Slice(out_lod[0][out_idx], out_lod[0][out_idx] + x_seq_len);
          framework::TensorCopy(x_sub_tensor, context.GetPlace(),
                                &out_sub_tensor);
          out_lod[0].push_back(out_lod[0][out_idx] + x_seq_len);
          out_idx++;
        }
      }
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
template <typename DeviceContext, typename T>
class SequenceExpandGradKernel : public framework::OpKernel<T> {
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
      auto place =
          context.template device_context<DeviceContext>().eigen_device();
      d_x_t.device(*place) = d_out_t.sum(Eigen::array<int, 1>({{0}}));
      d_out_data += (repeat * element_len);
      d_x_data += element_len;
    }
  }
};

}  // namespace operators
}  // namespace paddle
