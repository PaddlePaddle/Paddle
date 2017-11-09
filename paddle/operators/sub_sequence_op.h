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
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename T>
LoD subsequenceLoD(const T* in, const std::vector<int> offsets,
                   const std::vector<int> sizes) {
  auto out_lod = in->lod();
  size_t lod_offset = 0;

  auto n = in->lod()[0].size() - 1;
  out_lod[0][0] = 0;
  for (size_t i = 0; i < n; ++i) {
    lod_offset += sizes[i];
    out_lod[0][i+1] = lod_offset;
  }
  return out_lod;
}

template <typename Place, typename T>
class SubSequenceOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<LoDTensor>("X");
    std::vector<int> offsets = ctx.Attr<std::vector<int>>("offset");
    std::vector<int> sizes = ctx.Attr<std::vector<int>>("size");
    auto* out = ctx.Output<LoDTensor>("Out");

    auto offset_len = offsets.size();
    auto size_len = sizes.size();

    auto lod = in->lod();
    auto n = lod[0].size() - 1;

    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(n, offset_len,
                      "The length of input and offset should be the same")
    PADDLE_ENFORCE_EQ(n, size_len,
                      "The length of input and size should be the same")

    for (size_t i = 0; i < n; ++i) {
      auto offset = offsets[i];
      auto size = sizes[i];
      PADDLE_ENFORCE_LT(lod[0][i] + offset + size, lod[0][i + 1],
                        "The target tensor's length overflow")
    }

    out->mutable_data<T>(ctx.GetPlace());
    auto out_lod = subsequenceLoD(in, offsets, sizes);
    out->set_lod(out_lod);

    auto in_stride = framework::stride(in->dims());
    auto out_stride = framework::stride(out->dims());

    size_t out_offset = 0;
    for (size_t i = 0; i < n; ++i) {
      auto offset = offsets[i];
      auto size = sizes[i];

      Tensor in_t = in->Slice(static_cast<int>(lod[0][i] + offset),
                               static_cast<int>(lod[0][i] + offset + size));

      StridedMemcpy<T>(ctx.device_context(), in_t.data<T>(),
                       in_stride, in_t.dims(), out_stride,
                       out->data<T>() + out_offset);
      out_offset += size * in_stride[0];
    }
  }
};

template <typename Place, typename T>
class SubSequenceGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<LoDTensor>("X");
    std::vector<int> offsets = ctx.Attr<std::vector<int>>("offset");
    std::vector<int> sizes = ctx.Attr<std::vector<int>>("size");
    auto* out_grad =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));

    auto offset_len = offsets.size();
    auto size_len = sizes.size();

    auto lod = in->lod();
    auto n = lod[0].size() - 1;

    // check input data format
    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(n, offset_len,
                      "The length of input and offset should be the same")
    PADDLE_ENFORCE_EQ(n, size_len,
                      "The length of input and size should be the same")

    for (size_t i = 0; i < n; ++i) {
      auto offset = offsets[i];
      auto size = sizes[i];
      PADDLE_ENFORCE_LT(lod[0][i] + offset + size, lod[0][i + 1],
                        "The target tensor's length overflow")
    }

    auto out_lod = subsequenceLoD(in, offsets, sizes);

    x_grad->set_lod(lod);
    x_grad->mutable_data<T>(ctx.GetPlace());
    auto temp = framework::EigenVector<T>::Flatten(*x_grad);
    temp.device(ctx.GetEigenDevice<Place>()) = temp.constant(static_cast<T>(0));

    auto out_grad_stride = framework::stride(out_grad->dims());

    for (size_t i = 0; i < out_lod[0].size() - 1; ++i) {
      Tensor out_grad_t =
          out_grad->Slice(static_cast<int>(out_lod[0][i]),
                          static_cast<int>(out_lod[0][i + 1]));
      auto out_grad_stride = framework::stride(out_grad_t.dims());

      auto x_grad_stride = framework::stride(x_grad->dims());

      auto offset = offsets[i];
      auto size = sizes[i];

      Tensor x_grad_t = x_grad->Slice(static_cast<int>(lod[0][i] + offset),
                         static_cast<int>(lod[0][i] + offset + size));

      StridedMemcpy<T>(ctx.device_context(), out_grad_t.data<T>(),
                       out_grad_stride, out_grad_t.dims(), x_grad_stride,
                       x_grad_t.data<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle
