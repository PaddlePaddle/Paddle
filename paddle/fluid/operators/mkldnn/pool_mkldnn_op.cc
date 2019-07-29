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
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input tensor");

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

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

    auto input_format = input->format();
    memory::format output_format{memory::format::format_undef};

    mkldnn::memory::data_type dt =
        paddle::framework::ToMKLDNNDataType(input->type());
    auto fmt = input->format();

    const std::string key = platform::PoolingMKLDNNHandler::GetHash(
        src_tz, pooling_type, ksize, strides, paddings, dt, fmt,
        ctx.op().Output("Out"));

    platform::PoolingMKLDNNHandler handler(pooling_type, dt,
                                           ctx.Attr<bool>("is_test"), dev_ctx,
                                           mkldnn_engine, key);

    auto src_md = platform::MKLDNNMemDesc(src_tz, dt, input_format);

    auto src_memory =
        handler.AcquireSrcMemory(src_md, to_void_cast<T>(input_data));

    /* create memory descriptor for pooling without specified format
     * ('any') which lets a primitive (pooling in this case) choose
     * the memory format preferred for best performance
     */
    auto dst_md =
        platform::MKLDNNMemDesc(dst_tz, dt, mkldnn::memory::format::any);

    auto pooling_pd = handler.AcquirePoolingPrimitiveDescriptor(
        src_tz, dst_tz, src_md, dst_md, ksize, strides, paddings,
        ctx.Attr<bool>("ceil_mode"));

    auto dst_memory =
        handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));

    auto pool_p = handler.AcquirePooling(dst_memory, src_memory);

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{*pool_p};
    stream(stream::kind::eager).submit(pipeline).wait();

    output_format =
        (memory::format)dst_memory->get_primitive_desc().desc().data.format;

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(output_format);
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

    PADDLE_ENFORCE(in_x->layout() == DataLayout::kMKLDNN &&
                       in_x->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input X tensor");
    PADDLE_ENFORCE(out_grad->layout() == DataLayout::kMKLDNN &&
                       out_grad->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input output_grad tensor");

    PADDLE_ENFORCE(
        !ctx.Attr<bool>("is_test"),
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
    const mkldnn::engine& mkldnn_engine = dev_ctx.GetEngine();

    std::vector<mkldnn::primitive> pipeline;

    const T* out_grad_data = out_grad->data<T>();
    T* in_x_grad_data = in_x_grad->mutable_data<T>(ctx.GetPlace());
    memory::format in_x_grad_format{memory::format::format_undef};

    std::vector<int> diff_src_tz =
        paddle::framework::vectorize2int(in_x_grad->dims());
    std::vector<int> diff_dst_tz =
        paddle::framework::vectorize2int(out_grad->dims());

    // Get an unique name from "argument" name of "Out" variable
    // This name will be used as key when referring info from device context
    const std::string key = platform::PoolingMKLDNNHandler::GetHash(
        diff_src_tz, pooling_type, ksize, strides, paddings,
        memory::data_type::f32, in_x->format(), ctx.op().Input("Out"));

    platform::PoolingMKLDNNHandler handler(
        pooling_type, paddle::framework::ToMKLDNNDataType(in_x_grad->type()),
        false, dev_ctx, mkldnn_engine, key);

    auto workspace = handler.AcquireWorkspaceMemory();

    auto diff_dst_md = platform::MKLDNNMemDesc(
        {diff_dst_tz}, platform::MKLDNNGetDataType<T>(), out_grad->format());

    auto diff_dst_memory = handler.AcquireDiffDstMemory(
        diff_dst_md, to_void_cast<T>(out_grad_data));

    auto diff_src_md =
        platform::MKLDNNMemDesc(diff_src_tz, platform::MKLDNNGetDataType<T>(),
                                mkldnn::memory::format::any);

    auto bwd_pd = handler.AcquirePoolingBackwardPrimitiveDescriptor(
        diff_dst_md, diff_src_md, ksize, strides, paddings);

    auto diff_src_memory = handler.AcquireDiffSrcMemoryFromPrimitive(
        reinterpret_cast<void*>(in_x_grad_data));

    auto pool_bwd_p = handler.AcquirePoolingBackward(diff_dst_memory, workspace,
                                                     diff_src_memory);

    pipeline.push_back(*pool_bwd_p);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    in_x_grad_format = (memory::format)diff_src_memory->get_primitive_desc()
                           .desc()
                           .data.format;
    in_x_grad->set_layout(DataLayout::kMKLDNN);
    in_x_grad->set_format(in_x_grad_format);
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
