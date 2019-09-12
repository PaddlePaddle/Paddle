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
                       const MKLDNNMemoryFormat fmt,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       platform::Place cpu_place, const std::string& uniq_name)
      : platform::MKLDNNHandler(dev_ctx, dev_ctx.GetEngine(),
                                platform::GetHash(dims, uniq_name)),
        place_(cpu_place),
        fwd_pd_(nullptr),
        bwd_pd_(nullptr) {
    this->AcquireSoftmaxPrimitiveDescriptor(dims, fmt);
  }

  SoftmaxMKLDNNHandler(const std::vector<int>& dims,
                       const MKLDNNMemoryFormat fmt,
                       const MKLDNNMemoryFormat diff_fmt,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       platform::Place cpu_place, const std::string& uniq_name)
      : platform::MKLDNNHandler(dev_ctx, dev_ctx.GetEngine(),
                                platform::GetHash(dims, uniq_name)),
        place_(cpu_place),
        fwd_pd_(nullptr),
        bwd_pd_(nullptr) {
    // If we are in Grad operatgor then update a key with BWD suffix to
    // distinguish from FWD memory primitives
    // Key_common will allow to access FWD_PD from cache
    this->AcquireSoftmaxPrimitiveDescriptor(dims, fmt);
    this->AcquireSoftmaxBackwardPrimitiveDescriptor(dims, fmt, diff_fmt);
  }

  // TODO(jczaja): Once fwd_pd_ are moved to MKLDNNHandler then this function
  // should be moved as well eg. SoftmaxMKLDNNHandler -> MKLDNNHandler<softmax_>
  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(const Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(fwd_pd_->src_primitive_desc(),
                                            to_void_cast<T>(input_data),
                                            "@src_mem_p");
  }

  // TODO(jczaja): Move to MKLDNNHandler as common code
  std::shared_ptr<mkldnn::memory> AcquireDstMemory(framework::Tensor* output) {
    T* ptr = output->mutable_data<T>(place_,
                                     fwd_pd_->dst_primitive_desc().get_size());
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_primitive_desc(), ptr,
                                            "@dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(const Tensor* output) {
    const T* output_data = output->data<T>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_primitive_desc(),
                                            to_void_cast<T>(output_data),
                                            "@bwd-dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemory(const Tensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_dst_primitive_desc(),
                                            to_void_cast<T>(ptr),
                                            "@diff_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemory(
      framework::Tensor* diffsrc) {
    T* ptr = diffsrc->mutable_data<T>(
        place_, bwd_pd_->diff_src_primitive_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_src_primitive_desc(),
                                            ptr, "@diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::softmax_forward> AcquireSoftmax(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p) {
    /*Generate key*/
    auto prim_key = key_ + "@softmax_p";

    auto softmax_p = std::static_pointer_cast<mkldnn::softmax_forward>(
        dev_ctx_.GetBlob(prim_key));
    if (softmax_p == nullptr) {
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
      softmax_bwd_p = std::make_shared<mkldnn::softmax_backward>(
          *bwd_pd_, *dst_memory_p, *diff_dst_memory_p, *diff_src_memory_p);
      dev_ctx_.SetBlob(prim_key, softmax_bwd_p);
    }

    return softmax_bwd_p;
  }

 protected:
  void AcquireSoftmaxPrimitiveDescriptor(const std::vector<int>& dims,
                                         const mkldnn::memory::format fmt) {
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
            mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);
        auto softmax_desc =
            softmax_forward::desc(prop_kind::forward_scoring, md, 1 /*dim: C*/);
        fwd_pd_.reset(
            new softmax_forward::primitive_desc(softmax_desc, engine_));
        dev_ctx_.SetBlob(key_softmax_pd, fwd_pd_);
      }
    }
  }

  void AcquireSoftmaxBackwardPrimitiveDescriptor(
      const std::vector<int>& dims, const mkldnn::memory::format fmt,
      const mkldnn::memory::format diff_fmt) {
    // Fwd_PD_ has to exists when to create BWD_PD_
    PADDLE_ENFORCE_NOT_NULL(fwd_pd_);
    const std::string key_bwd_pd = key_ + "@softmax_bwd_pd";
    bwd_pd_ =
        std::static_pointer_cast<mkldnn::softmax_backward::primitive_desc>(
            dev_ctx_.GetBlob(key_bwd_pd));
    if (bwd_pd_ == nullptr) {
      auto data_softmax_md =
          mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);
      auto diff_softmax_md = mkldnn::memory::desc(
          dims, platform::MKLDNNGetDataType<T>(), diff_fmt);
      // TODO(jczaja): Add support for other axes
      auto backward_desc = softmax_backward::desc(
          diff_softmax_md, data_softmax_md, 1 /* dim: C*/);
      bwd_pd_.reset(new mkldnn::softmax_backward::primitive_desc(
          backward_desc, engine_, *fwd_pd_));
      dev_ctx_.SetBlob(key_bwd_pd, bwd_pd_);
    }
  }

 private:
  platform::Place place_;
  std::shared_ptr<mkldnn::softmax_forward::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::softmax_backward::primitive_desc> bwd_pd_;
};

template <typename T>
class SoftmaxMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");
    PADDLE_ENFORCE_EQ(
        input->dims(), output->dims(),
        "The shape of softmax's input and output must be identical.");

    // flatten input and output to 2-D matrixs
    auto dims = input->dims();  // input and output share the same shape
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);

    auto src_tz = paddle::framework::vectorize<int>(flattened_dims);
    auto dst_tz = src_tz;
    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};

    SoftmaxMKLDNNHandler<T> handler(softmax_tz, MKLDNNMemoryFormat::nc, dev_ctx,
                                    ctx.GetPlace(), ctx.op().Output("Out"));
    // Currently only NC data format is supported
    auto softmax_src_memory_p = handler.AcquireSrcMemory(input);
    auto softmax_dst_memory_p = handler.AcquireDstMemory(output);
    auto softmax_p =
        handler.AcquireSoftmax(softmax_dst_memory_p, softmax_src_memory_p);

    std::vector<primitive> pipeline{
        *(static_cast<softmax_forward::primitive*>(softmax_p.get()))};
    stream(stream::kind::eager).submit(pipeline).wait();

    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    const bool is_test = ctx.Attr<bool>("is_test");
    if (!is_test) {
      T threshold = exp(-64);
      for (int i = 0; i < dst_tz[0] * dst_tz[1]; ++i) {
        output_data[i] =
            output_data[i] < threshold ? threshold : output_data[i];
      }
    }

    output->set_layout(framework::DataLayout::kMKLDNN);
    // Softmax output format is the same as input one
    output->set_format(input->format());
  }
};

template <typename T>
class SoftmaxMKLDNNGradKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const Tensor* output = ctx.Input<Tensor>("Out");
    auto* dout = ctx.template Input<Tensor>(framework::GradVarName("Out"));
    auto* dx =
        ctx.template Output<framework::Tensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(
        dout->dims(), dx->dims(),
        "The shape of softmax_grad's input and output must be identical.");

    auto dims = dout->dims();  // input and output share the same shape
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);

    std::vector<int> dst_tz = paddle::framework::vectorize<int>(flattened_dims);
    std::vector<int> src_tz(dst_tz);

    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};

    // Two dimensional softmax does support NC format
    // Normalization is made after innermost dimension eg. C out of NC
    SoftmaxMKLDNNHandler<T> handler(softmax_tz, MKLDNNMemoryFormat::nc,
                                    MKLDNNMemoryFormat::nc, dev_ctx,
                                    ctx.GetPlace(), ctx.op().InputName("Out"));

    auto dst_memory_p = handler.AcquireDstMemory(output);
    auto diff_dst_memory_p = handler.AcquireDiffDstMemory(dout);
    auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);

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
