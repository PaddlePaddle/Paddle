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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using framework::DataLayout;

template <typename T>
class TransposeMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    const T* input_data = input->data<T>();

    if (ndims == 1) {
      output->ShareDataWith(*input);
      return;
    }

    std::vector<int> nchw_tz = paddle::framework::vectorize2int(input->dims());

    const std::string key = platform::TransposeMKLDNNHandler::GetHash(
        nchw_tz, axis, ctx.op().Output("Out"));

    platform::TransposeMKLDNNHandler handler(nchw_tz, axis, dev_ctx,
                                             mkldnn_engine, key);

    auto transpose_src_memory_p = handler.AcquireSrcMemory(
        input->get_mkldnn_prim_desc(), platform::to_void_cast<T>(input_data));
    auto transpose_dst_memory_p =
        handler.AcquireDstMemory(output, ctx.GetPlace());
    auto transpose_p = handler.AcquireTranspose(transpose_dst_memory_p,
                                                transpose_src_memory_p);

    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(*transpose_p);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    // Transpose did change logical dimensions of Tensor, but reorder does not.
    // Reorder does change only physical layout eg. format , strides
    // so we need to create new primitive descriptor with changed logical layout
    // so it match output shape
    auto output_mem_pd = paddle::platform::create_prim_desc_from_dims(
        paddle::framework::vectorize2int(output->dims()),
        mkldnn::memory::format::blocked);
    output->set_mkldnn_prim_desc(*output_mem_pd);
  }
};

template <typename T>
class TransposeINT8MKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    std::vector<int> axis_int8 = {0, 2, 3, 1};
    if (axis.size() != 1) {
      PADDLE_ENFORCE_EQ(axis.size(), axis_int8.size());
      for (size_t i = 0; i < axis.size(); i++) {
        PADDLE_ENFORCE_EQ(axis[i], axis_int8[i],
                          "Current INT8 MKLDNN Transpose kernel only surpport "
                          "axis with [0, 2, 3, 1] due to MKL-DNN kernel "
                          "implementation.");
      }
    }
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    output->ShareDataWith(*input);
  }
};

template <typename T>
class TransposeMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    if (!x_grad) return;

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);
    int ndims = axis.size();
    if (ndims == 1) {
      x_grad->ShareDataWith(*out_grad);
      return;
    }

    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    const T* out_grad_data = out_grad->data<T>();
    x_grad->mutable_data<T>(ctx.GetPlace());

    std::vector<int> nchw_tz =
        paddle::framework::vectorize2int(out_grad->dims());

    const std::string key = platform::TransposeMKLDNNHandler::GetHash(
        nchw_tz, axis, ctx.op().Output(framework::GradVarName("X")));

    platform::TransposeMKLDNNHandler handler(nchw_tz, reversed_axis, dev_ctx,
                                             mkldnn_engine, key);

    auto transpose_src_memory_p =
        handler.AcquireSrcMemory(out_grad->get_mkldnn_prim_desc(),
                                 platform::to_void_cast<T>(out_grad_data));
    auto transpose_dst_memory_p =
        handler.AcquireDstMemory(x_grad, ctx.GetPlace());
    auto transpose_p = handler.AcquireTranspose(transpose_dst_memory_p,
                                                transpose_src_memory_p);

    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(*transpose_p);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    // Transpose did change logical dimensions of Tensor, but reorder does not.
    // Reorder does change only physical layout eg. format , strides
    // so we need to create new primitive descriptor with changed logical layout
    // so it match output shape
    auto x_grad_mem_pd = paddle::platform::create_prim_desc_from_dims(
        paddle::framework::vectorize2int(x_grad->dims()),
        mkldnn::memory::format::blocked);
    x_grad->set_mkldnn_prim_desc(*x_grad_mem_pd);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(transpose2, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>,
                   ops::TransposeINT8MKLDNNOpKernel<uint8_t>,
                   ops::TransposeINT8MKLDNNOpKernel<int8_t>);

REGISTER_OP_KERNEL(transpose, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(transpose_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);
REGISTER_OP_KERNEL(transpose2_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::TransposeMKLDNNGradOpKernel<float>);
