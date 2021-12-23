/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include <memory>

#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
template <typename T>
class TopkV2XPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");
    const auto& in_dims = input->dims();
    const T* in_data = input->data<T>();
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    const auto& out_dims = output->dims();

    const auto& sorted = static_cast<bool>(ctx.Attr<bool>("sorted"));
    const auto& largest = static_cast<bool>(ctx.Attr<bool>("largest"));
    PADDLE_ENFORCE_EQ(
        sorted, true,
        platform::errors::External(
            "XPU API does not support unsorted topk operation currently."
            " Operator will be supported in future update."));
    PADDLE_ENFORCE_EQ(
        largest, true,
        platform::errors::External(
            "XPU API does not support smallest topk operation currently."
            " Operator will be supported in future update."));

    int axis = static_cast<int>(ctx.Attr<int>("axis"));
    if (axis < 0) axis += in_dims.size();

    size_t k = static_cast<int>(ctx.Attr<int>("k"));
    auto* k_t = ctx.Input<Tensor>("K");
    if (k_t) {
      k = k_t->data<int>()[0];
      framework::DDim output_dims = output->dims();
      output_dims[axis] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }
    if (axis + 1 == in_dims.size()) {
      Tensor indices_32_data_tensor;
      int32_t* indices_int_data = indices_32_data_tensor.mutable_data<int32_t>(
          ctx.GetPlace(), indices->numel());

      const size_t row = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const size_t col = in_dims[in_dims.size() - 1];
      auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
      int r = xpu::sorted_topk<T>(dev_ctx.x_context(), in_data, output_data,
                                  indices_int_data, row, col, k);
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External(
              "XPU API return wrong value[%d] in call kernel name "
              "[%s], please check "
              "where Baidu Kunlun Card is properly installed.",
              r, "sorted_topk"));
      r = xpu::cast_v2<int32_t, int64_t>(dev_ctx.x_context(),
                                         (const int32_t*)indices_int_data,
                                         indices_data, indices->numel());
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External(
              "XPU API return wrong value[%d] in call kernel name "
              "[%s], please check "
              "where Baidu Kunlun Card is properly installed.",
              r, "cast_v2"));

    } else {
      // do transpose if axis is not the last dim of input
      std::vector<int> trans_axes;
      for (int i = 0; i < axis; i++) {
        trans_axes.emplace_back(i);
      }
      for (int i = axis + 1; i < in_dims.size(); i++) {
        trans_axes.emplace_back(i);
      }
      trans_axes.emplace_back(axis);
      // Get input and output dims for transpose
      framework::DDim trans_dims(in_dims);
      framework::DDim trans_out_dims(output->dims());
      for (size_t i = 0; i < trans_axes.size(); i++) {
        trans_dims[i] = in_dims[trans_axes[i]];
      }
      for (size_t i = 0; i < trans_axes.size(); i++) {
        trans_out_dims[i] = out_dims[trans_axes[i]];
      }

      std::vector<int> x_shape_host(in_dims.size(), 0);
      for (int i = 0; i < in_dims.size(); ++i) {
        x_shape_host[i] = in_dims[i];
      }

      Tensor trans_in;
      trans_in.mutable_data<T>(trans_dims, ctx.GetPlace());
      auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

      // Transpose and save interval output to trans_in
      int r = xpu::transpose<T>(dev_ctx.x_context(), in_data,
                                trans_in.data<T>(), x_shape_host, trans_axes);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External("XPU API 1st Transpose kernel"
                                     " returns wrong value[%d]!",
                                     r));

      Tensor trans_out;
      T* trans_out_data =
          trans_out.mutable_data<T>(trans_out_dims, ctx.GetPlace());
      Tensor trans_idx;
      int64_t* trans_idx_data =
          trans_idx.mutable_data<int64_t>(trans_out_dims, ctx.GetPlace());
      Tensor trans_idx_int32;
      int32_t* trans_idx_int32_data =
          trans_idx_int32.mutable_data<int32_t>(trans_out_dims, ctx.GetPlace());
      const size_t row = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const size_t col = trans_dims[trans_dims.size() - 1];

      // Do top k on transposed input
      r = xpu::sorted_topk<T>(dev_ctx.x_context(), trans_in.data<T>(),
                              trans_out_data, trans_idx_int32_data, row, col,
                              k);
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External(
              "XPU API return wrong value[%d] in call kernel name "
              "[%s], please check "
              "where Baidu Kunlun Card is properly installed.",
              r, "sorted_topk"));

      r = xpu::cast_v2<int32_t, int64_t>(dev_ctx.x_context(),
                                         (const int32_t*)trans_idx_int32_data,
                                         trans_idx_data, indices->numel());
      PADDLE_ENFORCE_EQ(
          r, XPU_SUCCESS,
          platform::errors::External(
              "XPU API return wrong value[%d] in call kernel name "
              "[%s], please check "
              "where Baidu Kunlun Card is properly installed.",
              r, "cast_v2"));

      // Transpose back to original dims
      std::vector<int> trans_back_axes;
      for (int i = 0; i < axis; i++) {
        trans_axes.emplace_back(i);
      }
      trans_axes.emplace_back(trans_out_dims.size() - 1);
      for (int i = axis; i < trans_out_dims.size() - 1; i++) {
        trans_axes.emplace_back(i);
      }

      std::vector<int> trans_out_shape_host(trans_back_axes.size(), 0);
      for (size_t i = 0; i < trans_back_axes.size(); ++i) {
        trans_out_shape_host[i] = trans_out_dims[i];
      }
      r = xpu::transpose<T>(dev_ctx.x_context(), trans_out_data, output_data,
                            trans_out_shape_host, trans_back_axes);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External("XPU API 2nd Transpose kernel"
                                     " returns wrong value[%d]",
                                     r));
      r = xpu::transpose<int64_t>(dev_ctx.x_context(), trans_idx_data,
                                  indices_data, trans_out_shape_host,
                                  trans_back_axes);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External("XPU API 3rd Transpose kernel"
                                     " returns wrong value[%d]",
                                     r));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(top_k_v2, ops::TopkV2XPUKernel<float>);
#endif
