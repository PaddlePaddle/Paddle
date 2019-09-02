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
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;

using mkldnn::memory;  // Note: paddle has also "memory" namespace
using mkldnn::primitive;
using mkldnn::prop_kind;
using mkldnn::softmax_backward;
using mkldnn::softmax_forward;
using mkldnn::stream;
using platform::to_void_cast;

template <typename T>
class SoftmaxMKLDNNHandler : public platform::MKLDNNHandler {
 public:
  SoftmaxMKLDNNHandler(const std::vector<int>& dims,
                       const mkldnn::memory::format fmt,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        dims_(dims),
        fmt_(fmt) {}

  SoftmaxMKLDNNHandler(const std::vector<int>& dims,
                       const mkldnn::memory::format fmt,
                       const mkldnn::memory::format diff_fmt,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        dims_(dims),
        fmt_(fmt),
        diff_fmt_(diff_fmt) {
    // If we are in Grad operatgor then update a key with BWD suffix to
    // distinguish from FWD memory primitives
    // Key_common will allow to access FWD_PD from cache
    key_ += "-BWD";
  }

  // TODO(jczaja): Once fwd_pd_ are moved to MKLDNNHandler then this function
  // should be moved as well eg. SoftmaxMKLDNNHandler -> MKLDNNHandler<softmax_>
  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(void* ptr) {
    return this->AcquireMemory(dims_, platform::MKLDNNGetDataType<T>(), fmt_,
                               ptr, "@user_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(void* ptr) {
    return this->AcquireMemory(dims_, platform::MKLDNNGetDataType<T>(), fmt_,
                               ptr, "@user_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemory(void* ptr) {
    return this->AcquireMemory(dims_, platform::MKLDNNGetDataType<T>(),
                               diff_fmt_, ptr, "@user_diff_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemory(void* ptr) {
    return this->AcquireMemory(dims_, platform::MKLDNNGetDataType<T>(),
                               diff_fmt_, ptr, "@user_diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromPrimitive(void* ptr) {
    this->AcquireSoftmaxPrimitiveDescriptor();
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_primitive_desc(), ptr,
                                            "@dst_mem_p");
  }

  std::shared_ptr<mkldnn::softmax_forward> AcquireSoftmax(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p) {
    /*Generate key*/
    auto prim_key = key_ + "@softmax_p";

    auto softmax_p = std::static_pointer_cast<mkldnn::softmax_forward>(
        dev_ctx_.GetBlob(prim_key));
    if (softmax_p == nullptr) {
      this->AcquireSoftmaxPrimitiveDescriptor();
      softmax_p = std::make_shared<mkldnn::softmax_forward>(
          *fwd_pd_, *(static_cast<mkldnn::memory*>(src_memory_p.get())),
          *(static_cast<mkldnn::memory*>(dst_memory_p.get())));
      dev_ctx_.SetBlob(prim_key, softmax_p);
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
    if (softmax_bwd_p == nullptr) {
      auto data_softmax_md =
          mkldnn::memory::desc(dims_, platform::MKLDNNGetDataType<T>(), fmt_);
      auto diff_softmax_md = mkldnn::memory::desc(
          dims_, platform::MKLDNNGetDataType<T>(), diff_fmt_);
      // TODO(jczaja): Add support for other axes
      auto softmax_bwd_desc = softmax_backward::desc(
          diff_softmax_md, data_softmax_md, 1 /* dim: C*/);
      this->AcquireSoftmaxPrimitiveDescriptor();
      auto softmax_bwd_pd = mkldnn::softmax_backward::primitive_desc(
          softmax_bwd_desc, engine_, *fwd_pd_);

      softmax_bwd_p = std::make_shared<mkldnn::softmax_backward>(
          softmax_bwd_pd, *dst_memory_p, *diff_dst_memory_p,
          *diff_src_memory_p);
      dev_ctx_.SetBlob(prim_key, softmax_bwd_p);
    }

    return softmax_bwd_p;
  }

 protected:
  void AcquireSoftmaxPrimitiveDescriptor(void) {
    // Softmax PD has to be passed to Grad op that
    // may be executed by diffrent thread, hence
    // for that one we use key that does not contain TID
    const std::string key_softmax_pd = key_common_ + "@softmax_pd";

    fwd_pd_ = std::static_pointer_cast<softmax_forward::primitive_desc>(
        dev_ctx_.GetBlob(key_softmax_pd));
    if (fwd_pd_ == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);
      fwd_pd_ = std::static_pointer_cast<softmax_forward::primitive_desc>(
          dev_ctx_.GetBlob(key_softmax_pd));
      if (fwd_pd_ == nullptr) {
        // TODO(jczaja): Make it working along chosen axis and for
        // forward_training
        // Normalization is made after innermost dimension eg. C out of NC
        auto md =
            mkldnn::memory::desc(dims_, platform::MKLDNNGetDataType<T>(), fmt_);
        auto softmax_desc =
            softmax_forward::desc(prop_kind::forward_scoring, md, 1 /*dim: C*/);
        fwd_pd_.reset(
            new softmax_forward::primitive_desc(softmax_desc, engine_));
        dev_ctx_.SetBlob(key_softmax_pd, fwd_pd_);
      }
    }
  }

 private:
  std::vector<int> dims_;
  mkldnn::memory::format fmt_;
  mkldnn::memory::format diff_fmt_;
  std::shared_ptr<mkldnn::softmax_forward::primitive_desc> fwd_pd_;
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
    PADDLE_ENFORCE_EQ(
        input->dims(), output->dims(),
        "The shape of softmax's input and output must be identical.");

    // make sure 'output' holds memory, which will be shared by
    // 'flattened_output' later.
    output->mutable_data<T>(ctx.GetPlace());

    // flatten input and output to 2-D matrixs
    auto dims = input->dims();  // input and output share the same shape
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);
    framework::Tensor flattened_input;
    framework::Tensor flattened_output;
    flattened_input.ShareDataWith(*input).Resize(flattened_dims);
    flattened_output.ShareDataWith(*output).Resize(flattened_dims);

    const T* input_data = flattened_input.data<T>();
    T* output_data = flattened_output.mutable_data<T>(ctx.GetPlace());

    std::vector<int> src_tz = paddle::framework::vectorize2int(flattened_dims);
    std::vector<int> dst_tz = src_tz;
    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};
    // Generate keys for storing/retriving primitives for this operator
    const std::string key =
        platform::MKLDNNHandler::GetHash(softmax_tz, ctx.op().Output("Out"));

    SoftmaxMKLDNNHandler<T> handler(softmax_tz, mkldnn::memory::format::nc,
                                    dev_ctx, mkldnn_engine, key);

    // Currently only NC data format is supported
    auto softmax_src_memory_p =
        handler.AcquireSrcMemory(to_void_cast<T>(input_data));
    auto softmax_dst_memory_p =
        handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
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
    auto* dout = ctx.template Input<Tensor>(framework::GradVarName("Out"));
    auto* dx =
        ctx.template Output<framework::Tensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(
        dout->dims(), dx->dims(),
        "The shape of softmax_grad's input and output must be identical.");

    // make sure 'dx' holds memory, which will be shared by 'flattened_dx'
    // later.
    dx->template mutable_data<T>(ctx.GetPlace());

    auto dims = dout->dims();  // input and output share the same shape
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);
    framework::Tensor flattened_output;
    framework::Tensor flattened_dout;
    framework::Tensor flattened_dx;
    flattened_output.ShareDataWith(*output).Resize(flattened_dims);
    flattened_dout.ShareDataWith(*dout).Resize(flattened_dims);
    flattened_dx.ShareDataWith(*dx).Resize(flattened_dims);

    const T* dst_data = flattened_output.data<T>();
    const T* diff_dst_ptr = flattened_dout.template data<T>();
    T* diff_src_ptr = flattened_dx.template mutable_data<T>(ctx.GetPlace());

    std::vector<int> dst_tz = paddle::framework::vectorize2int(flattened_dims);
    std::vector<int> src_tz(dst_tz);

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
    // Normalization is made after innermost dimension eg. C out of NC
    SoftmaxMKLDNNHandler<T> handler(softmax_tz, mkldnn::memory::format::nc,
                                    mkldnn::memory::format::nc, dev_ctx,
                                    mkldnn_engine, key);

    auto dst_memory_p = handler.AcquireDstMemory(to_void_cast<T>(dst_data));
    auto diff_dst_memory_p =
        handler.AcquireDiffDstMemory(to_void_cast<T>(diff_dst_ptr));
    auto diff_src_memory_p =
        handler.AcquireDiffSrcMemory(to_void_cast<T>(diff_src_ptr));

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
