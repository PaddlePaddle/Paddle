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
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;

template <typename T>
class PoolMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");

    PADDLE_ENFORCE_EQ(input->layout(), DataLayout::kMKLDNN,
                      "Wrong layout set for Input tensor");
    PADDLE_ENFORCE_NE(input->format(), MKLDNNMemoryFormat::format_undef,
                      "Wrong format set for Input tensor");

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    if (ctx.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(input->dims()[i + 2]);
      }
    }

    // Only 2D pooling is supported now
    PADDLE_ENFORCE(ksize.size() == 2, "ksize must be 2D, i.e. 2D pooling");
    PADDLE_ENFORCE(pooling_type == "max" || pooling_type == "avg",
                   "pooling_type must be 'max' or 'avg'");
    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input dim must be with 4, i.e. NCHW");

    auto src_tz = paddle::framework::vectorize<int>(input->dims());
    auto dst_tz = paddle::framework::vectorize<int>(output->dims());

    auto is_test = ctx.Attr<bool>("is_test");

    platform::PoolingMKLDNNHandler<T> handler(
        src_tz, dst_tz, ksize, strides, paddings, pooling_type,
        ctx.Attr<bool>("ceil_mode"), input->format(),
        paddle::framework::ToMKLDNNDataType(input->type()), is_test, dev_ctx,
        ctx.GetPlace(), ctx.op().Output("Out"), ctx.Attr<bool>("exclusive"));

    auto src_memory = handler.AcquireSrcMemory(input);
    auto dst_memory = handler.AcquireDstMemory(output);

    std::shared_ptr<mkldnn::pooling_forward> pool_p;
    std::shared_ptr<mkldnn::memory> workspace_memory;
    if ((is_test == false) && (pooling_type == "max")) {
      // Training
      workspace_memory = handler.AcquireWorkspaceMemory();
      pool_p = handler.AcquireForwardPrimitive(*src_memory, *dst_memory,
                                               *workspace_memory);
    } else {
      // Inference
      pool_p = handler.AcquireForwardPrimitive(*src_memory, *dst_memory);
    }

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{*pool_p};
    stream(stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

template <typename T>
class PoolMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    const Tensor* in_x = ctx.Input<Tensor>("X");
    const Tensor* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(in_x->layout(), DataLayout::kMKLDNN,
                      "Wrong layout set for Input tensor");
    PADDLE_ENFORCE_NE(in_x->format(), MKLDNNMemoryFormat::format_undef,
                      "Wrong format set for Input tensor");

    PADDLE_ENFORCE_EQ(out_grad->layout(), DataLayout::kMKLDNN,
                      "Wrong layout set for Input output_grad tensor");
    PADDLE_ENFORCE_NE(out_grad->format(), MKLDNNMemoryFormat::format_undef,
                      "Wrong format set for Input output_grad tensor");

    PADDLE_ENFORCE_EQ(
        ctx.Attr<bool>("is_test"), false,
        "is_test attribute should be set to False in training phase.");

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    if (ctx.Attr<bool>("global_pooling")) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    std::vector<mkldnn::primitive> pipeline;

    auto diff_src_tz = paddle::framework::vectorize<int>(in_x_grad->dims());
    auto diff_dst_tz = paddle::framework::vectorize<int>(out_grad->dims());

    // Get an unique name from "argument" name of "Out" variable
    // This name will be used as key when referring info from device context
    const std::string key = platform::CreateKey(
        diff_src_tz, pooling_type, ksize, strides, paddings,
        memory::data_type::f32, in_x->format(), ctx.op().Input("Out"));

    platform::PoolingMKLDNNHandler<T> handler(
        diff_dst_tz, diff_src_tz, ksize, strides, paddings, pooling_type,
        ctx.Attr<bool>("ceil_mode"), in_x->format(), out_grad->format(),
        paddle::framework::ToMKLDNNDataType(out_grad->type()), dev_ctx,
        ctx.GetPlace(), ctx.op().Input("Out"), ctx.Attr<bool>("exclusive"));

    auto diff_dst_memory = handler.AcquireDiffDstMemory(out_grad);
    auto diff_src_memory = handler.AcquireDiffSrcMemory(in_x_grad);

    std::shared_ptr<mkldnn::pooling_backward> pool_bwd_p;
    std::shared_ptr<mkldnn::memory> workspace_memory;
    if (pooling_type == "max") {
      // Max - pooling needs Workspace
      workspace_memory = handler.AcquireWorkspaceMemory();
      pool_bwd_p = handler.AcquireBackwardPrimitive(
          *diff_dst_memory, *workspace_memory, *diff_src_memory);
    } else {
      // Average Pooling
      pool_bwd_p =
          handler.AcquireBackwardPrimitive(*diff_dst_memory, *diff_src_memory);
    }

    pipeline.push_back(*pool_bwd_p);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    in_x_grad->set_layout(DataLayout::kMKLDNN);
    in_x_grad->set_format(platform::GetMKLDNNFormat(*diff_src_memory));
  }  // Compute()
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(pool2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNOpKernel<float>,
                   ops::PoolMKLDNNOpKernel<int8_t>,
                   ops::PoolMKLDNNOpKernel<uint8_t>);

REGISTER_OP_KERNEL(pool2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNGradOpKernel<float>);
