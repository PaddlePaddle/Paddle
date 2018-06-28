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

#include <iostream>
#include "mkldnn.hpp"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;

using mkldnn::memory;  // Note: paddle has also "memory" namespace
using mkldnn::primitive;
using mkldnn::softmax_forward;
using mkldnn::softmax_backward;
using mkldnn::prop_kind;
using mkldnn::stream;
using platform::to_void_cast;

class SoftmaxMKLDNNHandler : public platform::MKLDNNHandler {
 public:
  SoftmaxMKLDNNHandler(
      std::shared_ptr<mkldnn::softmax_forward::primitive_desc> softmax_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        softmax_pd_(softmax_pd) {}

  SoftmaxMKLDNNHandler(
      std::shared_ptr<mkldnn::softmax_forward::primitive_desc> softmax_pd,
      std::shared_ptr<mkldnn::softmax_backward::primitive_desc> softmax_bwd_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        softmax_pd_(softmax_pd),
        softmax_bwd_pd_(softmax_bwd_pd) {
    // If we are in Grad operatgor then update a key with BWD suffix to
    // distinguish from FWD memory primitives
    key_ += "-BWD";
  }

  std::shared_ptr<mkldnn::softmax_forward> AcquireSoftmax(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p) {
    /*Generate key*/
    auto prim_key = key_ + "@softmax_p";

    auto softmax_p = std::static_pointer_cast<mkldnn::softmax_forward>(
        dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE((softmax_p != nullptr) || (is_reusing_ == false),
                   "Fail to find softmax primitive in device context");
    if (softmax_p == nullptr) {
      softmax_p = std::make_shared<mkldnn::softmax_forward>(
          *(softmax_pd_.get()),
          *(static_cast<mkldnn::memory*>(src_memory_p.get())),
          *(static_cast<mkldnn::memory*>(dst_memory_p.get())));
      dev_ctx_.SetBlob(prim_key, softmax_p);
    } else {
      is_reusing_ = true;
    }

    return softmax_p;
  }

  std::shared_ptr<mkldnn::softmax_backward> AcquireSoftmaxBackward(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> diff_dst_memory_p,
      std::shared_ptr<mkldnn::memory> diff_src_memory_p) {
    auto prim_key = key_ + "@softmax_bwd_p";
    auto softmax_bwd_p = std::static_pointer_cast<mkldnn::softmax_backward>(
        dev_ctx_.GetBlob(prim_key));
    PADDLE_ENFORCE((softmax_bwd_p != nullptr) || (is_reusing_ == false),
                   "Fail to find softmax backward primitive in device context");
    if (softmax_bwd_p == nullptr) {
      softmax_bwd_p = std::make_shared<mkldnn::softmax_backward>(
          *softmax_bwd_pd_, *(dst_memory_p.get()), *(diff_dst_memory_p.get()),
          *(diff_src_memory_p.get()));
      dev_ctx_.SetBlob(prim_key, softmax_bwd_p);
    } else {
      is_reusing_ = true;
    }

    return softmax_bwd_p;
  }

 private:
  std::shared_ptr<mkldnn::softmax_forward::primitive_desc> softmax_pd_;
  std::shared_ptr<mkldnn::softmax_backward::primitive_desc> softmax_bwd_pd_;
};

template <typename T>
class SoftmaxMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();
    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");
    PADDLE_ENFORCE(input->dims().size() == 2UL,
                   "The input of softmax op must be a 2D matrix.");
    const T* input_data = input->data<T>();
    // allocate memory for output
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());
    // MKL-DNN does support softmax over selected axis. Having 2D Tensor,
    // we will make normalization after final eg. axis: 1
    PADDLE_ENFORCE(((src_tz[0] == dst_tz[0]) && (src_tz[1] == dst_tz[1])),
                   "Softmax input and output dimensions should match");
    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};
    // Generate keys for storing/retriving primitives for this operator
    const std::string key =
        platform::MKLDNNHandler::GetHash(softmax_tz, ctx.op().Output("Out"));
    const std::string key_softmax_pd = key + "@softmax_pd";

    // Currently only NC data format is supported
    auto softmax_md = MKLDNNMemDesc(
        {softmax_tz}, platform::MKLDNNGetDataType<T>(), memory::format::nc);
    // Normalization is made after innermost dimension eg. C out of NC
    auto softmax_desc = softmax_forward::desc(prop_kind::forward_scoring,
                                              softmax_md, 1 /*dim: C*/);
    auto softmax_pd = std::make_shared<mkldnn::softmax_forward::primitive_desc>(
        softmax_desc, mkldnn_engine);
    dev_ctx.SetBlob(key_softmax_pd, softmax_pd);

    SoftmaxMKLDNNHandler handler(softmax_pd, dev_ctx, mkldnn_engine, key);
    auto softmax_src_memory_p =
        handler.AcquireSrcMemory(softmax_md, to_void_cast<T>(input_data));
    auto softmax_dst_memory_p =
        handler.AcquireDstMemory(softmax_md, to_void_cast<T>(output_data));
    auto softmax_p =
        handler.AcquireSoftmax(softmax_dst_memory_p, softmax_src_memory_p);

    std::vector<primitive> pipeline{
        *(static_cast<softmax_forward::primitive*>(softmax_p.get()))};
    stream(stream::kind::eager).submit(pipeline).wait();

    const bool is_test = ctx.Attr<bool>("is_test");
    if (!is_test) {
      T threshold = exp(-64);
      for (int i = 0; i < dst_tz[0] * dst_tz[1]; ++i) {
        output_data[i] =
            output_data[i] < threshold ? threshold : output_data[i];
      }
    }
  }
};

template <typename T>
class SoftmaxMKLDNNGradKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();
    const Tensor* output = ctx.Input<Tensor>("Out");
    const T* dst_data = output->data<T>();

    auto* dout = ctx.template Input<Tensor>(framework::GradVarName("Out"));
    const auto* diff_dst_ptr = dout->template data<T>();

    auto* dx =
        ctx.template Output<framework::Tensor>(framework::GradVarName("X"));
    T* diff_src_ptr = dx->template mutable_data<T>(ctx.GetPlace());

    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());
    std::vector<int> src_tz(dst_tz);
    PADDLE_ENFORCE(output->dims().size() == 2UL,
                   "The input of softmax op must be a 2D matrix.");
    // MKL-DNN does support softmax over selected axis. Having 2D Tensor,
    // we will make normalization after final eg. axis: 1
    PADDLE_ENFORCE(((src_tz[0] == dst_tz[0]) && (src_tz[1] == dst_tz[1])),
                   "Softmax input and output dimensions should match");
    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};
    // Currently only supports NC data format
    // retrieve eltwise primitive desc from device context
    const std::string key =
        platform::MKLDNNHandler::GetHash(softmax_tz, ctx.op().Input("Out"));
    const std::string key_softmax_pd = key + "@softmax_pd";

    auto softmax_pd =
        std::static_pointer_cast<mkldnn::softmax_forward::primitive_desc>(
            dev_ctx.GetBlob(key_softmax_pd));
    PADDLE_ENFORCE(softmax_pd != nullptr,
                   "Fail to find softmax_pd in device context");

    // TODO(jczaja): Add layouts support when there is a need to do so
    // Two dimensional softmax does support NC format
    auto data_softmax_md = MKLDNNMemDesc(
        {softmax_tz}, platform::MKLDNNGetDataType<T>(), memory::format::nc);
    auto diff_softmax_md = MKLDNNMemDesc(
        {softmax_tz}, platform::MKLDNNGetDataType<T>(), memory::format::nc);
    // Normalization is made after innermost dimension eg. C out of NC
    auto softmax_bwd_desc =
        softmax_backward::desc(diff_softmax_md, data_softmax_md, 1 /* dim: C*/);
    auto softmax_bwd_pd =
        std::make_shared<mkldnn::softmax_backward::primitive_desc>(
            softmax_bwd_desc, mkldnn_engine, *softmax_pd);

    SoftmaxMKLDNNHandler handler(softmax_pd, softmax_bwd_pd, dev_ctx,
                                 mkldnn_engine, key);
    auto dst_memory_p =
        handler.AcquireDstMemory(data_softmax_md, to_void_cast<T>(dst_data));
    auto diff_dst_memory_p = handler.AcquireDiffDstMemory(
        diff_softmax_md, to_void_cast<T>(diff_dst_ptr));
    auto diff_src_memory_p = handler.AcquireDiffSrcMemory(
        diff_softmax_md, to_void_cast<T>(diff_src_ptr));

    // Get primitve from device context
    auto softmax_bwd_p = handler.AcquireSoftmaxBackward(
        dst_memory_p, diff_dst_memory_p, diff_src_memory_p);

    std::vector<primitive> pipeline{*softmax_bwd_p};
    stream(stream::kind::eager).submit(pipeline).wait();
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(softmax, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNKernel<float>);
REGISTER_OP_KERNEL(softmax_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNGradKernel<float>);
