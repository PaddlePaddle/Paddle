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
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename T>
LoD SequenceSliceLoD(const T& in, const int64_t* offset_data,
                     const int64_t* length_data) {
  auto out_lod = in.lod();
  size_t lod_offset = 0;

  auto n = in.lod()[0].size() - 1;
  out_lod[0][0] = 0;
  for (size_t i = 0; i < n; ++i) {
    lod_offset += length_data[i];
    out_lod[0][i+1] = lod_offset;
  }
  return out_lod;
}

template <typename Place, typename T>
class SequenceSliceOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<LoDTensor>("X");
    auto* offset = ctx.Input<Tensor>("Offset");
    auto* length = ctx.Input<Tensor>("Length");
    auto* out = ctx.Output<LoDTensor>("Out");

    const int64_t* offset_data = offset->data<int64_t>();
    const int64_t* length_data = length->data<int64_t>();

    if (platform::is_gpu_place(ctx.GetPlace())) {
      framework::Tensor offset_cpu;
      offset_cpu.mutable_data<T>(offset->dims(), platform::CPUPlace());
      offset_cpu.CopyFrom(*offset, platform::CPUPlace(), ctx.device_context());
      offset_data = offset_cpu.data<int64_t>();

      framework::Tensor length_cpu;
      length_cpu.mutable_data<T>(length->dims(), platform::CPUPlace());
      length_cpu.CopyFrom(*length, platform::CPUPlace(), ctx.device_context());
      length_data = length_cpu.data<int64_t>();
    }

    auto lod = in->lod();
    auto n = lod[0].size() - 1;

    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(offset->dims().size(), 1UL,
                      "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(length->dims().size(), 1UL,
                      "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(
        n, length->dims()[0],
        "The size of input-sequence and length-array should be the same")
    PADDLE_ENFORCE_EQ(
        n, offset->dims()[0],
        "The size of input-sequence and offset-array should be the same")

    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_LT(0, offset_data[i], "The offset must greater than zero")
      PADDLE_ENFORCE_LT(0, length_data[i], "The length must greater than zero")
      PADDLE_ENFORCE_LT(lod[0][i] + offset_data[i] + length_data[i],
                        lod[0][i + 1], "The target tensor's length overflow")
    }

    out->mutable_data<T>(ctx.GetPlace());
    auto out_lod = SequenceSliceLoD(*in, offset_data, length_data);
    out->set_lod(out_lod);
    math::SetConstant<Place, T> set_zero;
    set_zero(ctx.device_context(), out, static_cast<T>(0));

    auto in_stride = framework::stride(in->dims());
    auto out_stride = framework::stride(out->dims());

    size_t out_offset = 0;
    for (size_t i = 0; i < n; ++i) {
      Tensor in_t =
          in->Slice(static_cast<int>(lod[0][i] + offset_data[i]),
                    static_cast<int>(lod[0][i] + offset_data[i] +
                    length_data[i]));

      StridedMemcpy<T>(ctx.device_context(), in_t.data<T>(),
                       in_stride, in_t.dims(), out_stride,
                       out->data<T>() + out_offset);
      out_offset += length_data[i] * in_stride[0];
    }
  }
};

template <typename Place, typename T>
class SequenceSliceGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<LoDTensor>("X");
    auto* offset = ctx.Input<Tensor>("Offset");
    auto* length = ctx.Input<Tensor>("Length");
    auto* out_grad =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));

    const int64_t* offset_data = offset->data<int64_t>();
    const int64_t* length_data = length->data<int64_t>();

    if (platform::is_gpu_place(ctx.GetPlace())) {
      framework::Tensor offset_cpu;
      offset_cpu.mutable_data<T>(offset->dims(), platform::CPUPlace());
      offset_cpu.CopyFrom(*offset, platform::CPUPlace(), ctx.device_context());
      offset_data = offset_cpu.data<int64_t>();

      framework::Tensor length_cpu;
      length_cpu.mutable_data<T>(length->dims(), platform::CPUPlace());
      length_cpu.CopyFrom(*length, platform::CPUPlace(), ctx.device_context());
      length_data = length_cpu.data<int64_t>();
    }

    auto lod = in->lod();
    auto out_lod = SequenceSliceLoD(*in, offset_data, length_data);

    x_grad->set_lod(lod);
    x_grad->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<Place, T> set_zero;
    set_zero(ctx.device_context(), x_grad, static_cast<T>(0));

    auto out_grad_stride = framework::stride(out_grad->dims());

    for (size_t i = 0; i < out_lod[0].size() - 1; ++i) {
      Tensor out_grad_t =
          out_grad->Slice(static_cast<int>(out_lod[0][i]),
                          static_cast<int>(out_lod[0][i + 1]));
      auto out_grad_stride = framework::stride(out_grad_t.dims());

      auto x_grad_stride = framework::stride(x_grad->dims());

      Tensor x_grad_t = x_grad->Slice(
          static_cast<int>(lod[0][i] + offset_data[i]),
          static_cast<int>(lod[0][i] + offset_data[i] + length_data[i]));

      StridedMemcpy<T>(ctx.device_context(), out_grad_t.data<T>(),
                       out_grad_stride, out_grad_t.dims(), x_grad_stride,
                       x_grad_t.data<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle
