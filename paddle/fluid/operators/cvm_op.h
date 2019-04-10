/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
class CVMOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const LoDTensor* x = context.Input<LoDTensor>("X");
    const T* x_data = x->data<T>();
    auto lod = x->lod()[0];
    int64_t item_size = x->numel() / x->dims()[0];
    int offset = 2;
    if (!context.Attr<bool>("use_cvm")) {
      item_size -= offset;
    }
    LoDTensor* y = context.Output<LoDTensor>("Y");
    T* y_data = y->mutable_data<T>(context.GetPlace());

    int seq_num = static_cast<int>(lod.size()) - 1;
    for (int i = 0; i < seq_num; ++i) {
      int64_t seq_len = static_cast<int64_t>(lod[i + 1] - lod[i]);

      for (int j = 0; j < seq_len; ++j) {
        if (context.Attr<bool>("use_cvm")) {
          std::memcpy(y_data, x_data, item_size * sizeof(T));
          y_data[0] = log(y_data[0] + 1);
          y_data[1] = log(y_data[1] + 1) - y_data[0];
          x_data += item_size;
          y_data += item_size;
        } else {
          std::memcpy(y_data, x_data + offset, item_size * sizeof(T));
          x_data += item_size + offset;
          y_data += item_size;
        }
      }
    }
  }
};

template <typename T>
class CVMGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    LoDTensor* dx = context.Output<LoDTensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    const Tensor* cvm = context.Input<Tensor>("CVM");
    const T* cvm_data = cvm->data<T>();
    int offset = 2;
    const framework::LoDTensor* dOut =
        context.Input<framework::LoDTensor>(framework::GradVarName("Y"));
    const T* dout_data = dOut->data<T>();

    auto lod = dx->lod()[0];
    int64_t item_size = dx->numel() / dx->dims()[0];
    if (!context.Attr<bool>("use_cvm")) {
      item_size -= offset;
    }

    int seq_num = static_cast<int>(lod.size()) - 1;
    for (int i = 0; i < seq_num; ++i) {
      int64_t seq_len = static_cast<int64_t>(lod[i + 1] - lod[i]);

      for (int j = 0; j < seq_len; ++j) {
        if (context.Attr<bool>("use_cvm")) {
          std::memcpy(dx_data, dout_data, item_size * sizeof(T));
          dx_data[0] = cvm_data[0];
          dx_data[1] = cvm_data[1];
          dx_data += item_size;
          dout_data += item_size;
        } else {
          std::memcpy(dx_data + offset, dout_data, item_size * sizeof(T));
          dx_data[0] = cvm_data[0];
          dx_data[1] = cvm_data[1];
          dx_data += item_size + offset;
          dout_data += item_size;
        }
      }
      cvm_data += offset;
    }
  }
};
}  // namespace operators
}  // namespace paddle
