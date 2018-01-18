/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

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

    const T* p_in_data = in->data<T>();

    auto in_dims = in->dims();
    int64_t in_width = in_dims[1];
    auto& in_lod = in->lod();

    PADDLE_ENFORCE_EQ(in_lod.size(), 1UL,
                      "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(
        in_dims[0], in_lod[0].back(),
        "Inconsistent size between X.shape[0] and X.lod()[0].back().");

    auto in_lod_l0 = in_lod[0];
    int seq_num = in_lod_l0.size() - 1;

    auto& out_lod = *out->mutable_lod();
    out_lod.resize(1);
    out_lod[0].clear();
    out_lod[0].push_back(0);
    for (int i = 0; i < seq_num; ++i) {
      size_t seq_len = in_lod_l0[i + 1] - in_lod_l0[i];
      size_t offset = 0;
      offset = (seq_len * in_width) / out_width;
      PADDLE_ENFORCE_EQ(offset * out_width, seq_len * in_width,
                        "Please make sure (sequence_length * dimension) can be "
                        "divided by new_dim with no remainder for each "
                        "sequence. The %dth sequence is invalid.",
                        i + 1);
      PADDLE_ENFORCE_GT(offset, 0,
                        "Illegal operation, length of the %dth sequence become "
                        "to 0 after reshaped.",
                        i + 1);
      out_lod[0].push_back(out_lod[0].back() + offset);
    }

    out->mutable_data<T>(context.GetPlace());
    out->Resize({static_cast<int64_t>(out_lod[0].back()), out_width});
    T* p_out_data = out->mutable_data<T>(context.GetPlace());
    math::set_constant(context.device_context(), out, 0.0f);

    for (int i = 0; i < seq_num; ++i) {
      size_t in_offset = in_lod_l0[i] * in_width;
      size_t out_offset = out_lod[0][i] * out_width;
      size_t in_count = (in_lod_l0[i + 1] - in_lod_l0[i]) * in_width;
      size_t out_count = (out_lod[0][i + 1] - out_lod[0][i]) * out_width;
      size_t bytes = sizeof(T) * std::min(in_count, out_count);
      if (platform::is_cpu_place(context.GetPlace())) {
        memory::Copy(boost::get<platform::CPUPlace>(context.GetPlace()),
                     p_out_data + out_offset,
                     boost::get<platform::CPUPlace>(context.GetPlace()),
                     p_in_data + in_offset, bytes);
      } else {
#ifdef PADDLE_WITH_CUDA
        auto& dev_ctx =
            context.template device_context<platform::CUDADeviceContext>();
        memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                     p_out_data + out_offset,
                     boost::get<platform::CUDAPlace>(context.GetPlace()),
                     p_in_data + in_offset, bytes, dev_ctx.stream());
#endif
      }
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceReshapeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x_tensor_ptr = context.Input<LoDTensor>("X");
    auto* out_tensor_ptr = context.Input<LoDTensor>("Out");
    auto* out_grad_tensor_ptr =
        context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x_grad_tensor_ptr =
        context.Output<LoDTensor>(framework::GradVarName("X"));

    T* p_x_grad_data = x_grad_tensor_ptr->mutable_data<T>(context.GetPlace());
    const T* p_out_grad_data = out_grad_tensor_ptr->data<T>();

    auto& x_lod = x_tensor_ptr->lod();
    int seq_num = x_lod[0].size() - 1;
    int x_width = x_tensor_ptr->dims()[1];
    auto& out_lod = out_tensor_ptr->lod();
    int out_width = out_tensor_ptr->dims()[1];

    math::set_constant(context.device_context(), x_grad_tensor_ptr, 0.0f);

    for (int i = 0; i < seq_num; ++i) {
      size_t src_offset = out_lod[0][i] * out_width;
      size_t dst_offset = x_lod[0][i] * x_width;
      size_t src_count = (out_lod[0][i + 1] - out_lod[0][i]) * out_width;
      size_t dst_count = (x_lod[0][i + 1] - x_lod[0][i]) * x_width;
      size_t bytes = sizeof(T) * std::min(src_count, dst_count);
      if (platform::is_cpu_place(context.GetPlace())) {
        memory::Copy(boost::get<platform::CPUPlace>(context.GetPlace()),
                     p_x_grad_data + dst_offset,
                     boost::get<platform::CPUPlace>(context.GetPlace()),
                     p_out_grad_data + src_offset, bytes);
      } else {
#ifdef PADDLE_WITH_CUDA
        auto& dev_ctx =
            context.template device_context<platform::CUDADeviceContext>();
        memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                     p_x_grad_data + dst_offset,
                     boost::get<platform::CUDAPlace>(context.GetPlace()),
                     p_out_grad_data + src_offset, bytes, dev_ctx.stream());
#endif
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
