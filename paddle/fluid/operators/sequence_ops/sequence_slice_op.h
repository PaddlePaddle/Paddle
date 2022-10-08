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
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename T>
inline LoD SequenceSliceLoD(const T& in,
                            const int64_t* offset_data,
                            const int64_t* length_data) {
  auto out_lod = in.lod();
  size_t lod_offset = 0;

  auto n = in.lod()[0].size() - 1;
  out_lod[0][0] = 0;
  for (size_t i = 0; i < n; ++i) {
    lod_offset += length_data[i];
    out_lod[0][i + 1] = lod_offset;
  }
  return out_lod;
}

template <typename DeviceContext, typename T>
class SequenceSliceOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<LoDTensor>("X");
    auto* offset = ctx.Input<phi::DenseTensor>("Offset");
    auto* length = ctx.Input<phi::DenseTensor>("Length");
    auto* out = ctx.Output<LoDTensor>("Out");

    auto lod = in->lod();
    PADDLE_ENFORCE_EQ(lod.empty(),
                      false,
                      platform::errors::InvalidArgument(
                          "Input(X) Tensor of SequenceSlice operator does not "
                          "contain LoD information."));

    PADDLE_ENFORCE_EQ(
        lod.size(),
        1UL,
        platform::errors::InvalidArgument(
            "LoD information error. SequenceSlice operator only support one "
            "level sequence now, but received LoD level is %d.",
            lod.size()));
    auto n = lod[0].size() - 1;
    PADDLE_ENFORCE_EQ(
        n,
        static_cast<size_t>(length->dims()[0]),
        platform::errors::InvalidArgument(
            "Input length shape error. The length of input LoD sequence and "
            "input length-array‘s first dimension should be equal, but the LoD "
            "sequence length is %d, the length-array‘s first dimension is %d.",
            n,
            static_cast<size_t>(length->dims()[0])));
    PADDLE_ENFORCE_EQ(
        n,
        static_cast<size_t>(offset->dims()[0]),
        platform::errors::InvalidArgument(
            "Input offset shape error. The length of input LoD sequence and "
            "input offset-array‘s first dimension should be equal, but the LoD "
            "sequence length is %d, the offset-array‘s first dimension is %d.",
            n,
            static_cast<size_t>(offset->dims()[0])));

    const int64_t* offset_data = offset->data<int64_t>();
    const int64_t* length_data = length->data<int64_t>();
    phi::DenseTensor offset_cpu;
    phi::DenseTensor length_cpu;

    if (platform::is_gpu_place(ctx.GetPlace())) {
      offset_cpu.mutable_data<T>(offset->dims(), platform::CPUPlace());
      framework::TensorCopySync(*offset, platform::CPUPlace(), &offset_cpu);
      offset_data = offset_cpu.data<int64_t>();

      length_cpu.mutable_data<T>(length->dims(), platform::CPUPlace());
      framework::TensorCopySync(*length, platform::CPUPlace(), &length_cpu);
      length_data = length_cpu.data<int64_t>();
    }

    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_LE(0,
                        offset_data[i],
                        platform::errors::InvalidArgument(
                            "The input offset[%d]'s value is negative, its "
                            "value is %d, expect it to be non-negative.",
                            i,
                            offset_data[i]));
      PADDLE_ENFORCE_LE(0,
                        length_data[i],
                        platform::errors::InvalidArgument(
                            "The input length[%d]'s value is negative, its "
                            "value is %d, expect it to be non-negative.",
                            i,
                            offset_data[i]));
      PADDLE_ENFORCE_LE(
          lod[0][i] + offset_data[i] + length_data[i],
          lod[0][i + 1],
          platform::errors::OutOfRange(
              "The slice end index of target tensor is out of range. expect it "
              "less than or equal to %d, but the actual slice end index is %d.",
              lod[0][i + 1],
              lod[0][i] + offset_data[i] + length_data[i]));
    }

    out->mutable_data<T>(ctx.GetPlace());
    auto out_lod = SequenceSliceLoD(*in, offset_data, length_data);
    auto out_dims = in->dims();
    out_dims[0] = out_lod[0][out_lod[0].size() - 1];
    out->Resize(out_dims);
    out->set_lod(out_lod);

    auto in_stride = phi::stride(in->dims());
    auto out_stride = phi::stride(out->dims());

    size_t out_offset = 0;
    for (size_t i = 0; i < n; ++i) {
      if (length_data[i] == 0) continue;
      Tensor in_t = in->Slice(
          static_cast<int>(lod[0][i] + offset_data[i]),
          static_cast<int>(lod[0][i] + offset_data[i] + length_data[i]));

      StridedMemcpy<T>(ctx.device_context(),
                       in_t.data<T>(),
                       in_stride,
                       in_t.dims(),
                       out_stride,
                       out->data<T>() + out_offset);
      out_offset += length_data[i] * in_stride[0];
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceSliceGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<LoDTensor>("X");
    auto* offset = ctx.Input<phi::DenseTensor>("Offset");
    auto* length = ctx.Input<phi::DenseTensor>("Length");
    auto* out_grad =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));

    const int64_t* offset_data = offset->data<int64_t>();
    const int64_t* length_data = length->data<int64_t>();
    phi::DenseTensor offset_cpu;
    phi::DenseTensor length_cpu;

    if (platform::is_gpu_place(ctx.GetPlace())) {
      offset_cpu.mutable_data<T>(offset->dims(), platform::CPUPlace());
      framework::TensorCopySync(*offset, platform::CPUPlace(), &offset_cpu);
      offset_data = offset_cpu.data<int64_t>();

      length_cpu.mutable_data<T>(length->dims(), platform::CPUPlace());
      framework::TensorCopySync(*length, platform::CPUPlace(), &length_cpu);
      length_data = length_cpu.data<int64_t>();
    }

    auto lod = in->lod();
    // to avoid out_grad missing lod, compute lod again
    auto out_lod = SequenceSliceLoD(*in, offset_data, length_data);

    if (x_grad) {
      x_grad->mutable_data<T>(ctx.GetPlace());
      x_grad->set_lod(in->lod());
      phi::funcs::SetConstant<DeviceContext, T> set_zero;
      set_zero(ctx.template device_context<DeviceContext>(),
               x_grad,
               static_cast<T>(0));

      for (size_t i = 0; i < out_lod[0].size() - 1; ++i) {
        if (length_data[i] == 0) continue;
        Tensor out_grad_t =
            out_grad->Slice(static_cast<int>(out_lod[0][i]),
                            static_cast<int>(out_lod[0][i + 1]));
        auto out_grad_stride = phi::stride(out_grad_t.dims());

        auto x_grad_stride = phi::stride(x_grad->dims());

        Tensor x_grad_t = x_grad->Slice(
            static_cast<int>(lod[0][i] + offset_data[i]),
            static_cast<int>(lod[0][i] + offset_data[i] + length_data[i]));

        StridedMemcpy<T>(ctx.device_context(),
                         out_grad_t.data<T>(),
                         out_grad_stride,
                         out_grad_t.dims(),
                         x_grad_stride,
                         x_grad_t.data<T>());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
