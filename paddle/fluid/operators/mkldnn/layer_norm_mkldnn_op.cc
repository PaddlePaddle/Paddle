/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace operators {

template <typename T>
class LayerNormOneDNNHandler
    : public phi::funcs::
          OneDNNHandlerNoCachingT<T, dnnl::layer_normalization_forward> {
 public:
  LayerNormOneDNNHandler(const std::vector<int64_t>& dims,
                         const float& epsilon,
                         const dnnl::normalization_flags& flags,
                         const bool& is_test,
                         const phi::DenseTensor* x,
                         const dnnl::engine engine,
                         platform::Place cpu_place)
      : phi::funcs::OneDNNHandlerNoCachingT<T,
                                            dnnl::layer_normalization_forward>(
            engine, cpu_place) {
    const auto fwd_prop_kind = is_test ? dnnl::prop_kind::forward_inference
                                       : dnnl::prop_kind::forward_training;
    this->AcquireForwardPrimitiveDescriptor(
        fwd_prop_kind, x->mem_desc(), epsilon, flags);
  }

  std::shared_ptr<dnnl::memory> AcquireScaleShiftMemory(
      const phi::DenseTensor* scale,
      const phi::DenseTensor* shift,
      const framework::ExecutionContext& ctx) {
    // OneDNN requires a single piece of memory for scale and shift data. During
    // inference both pieces of memory are merged inside
    // layer_norm_onednn_optimization_pass, but during training we have to
    // manually copy them into new memory buffer
    auto* scaleshift = ctx.Input<phi::DenseTensor>("ScaleShift");
    if (scaleshift) {
      return this->AcquireMemoryFromPrimitive(
          this->fwd_pd_->weights_desc(),
          phi::funcs::to_void_cast(scaleshift->data<float>()));
    } else {
      const unsigned int C = phi::vectorize(scale->dims())[0];

      auto scaleshift_memory =
          this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc());

      auto mem_data_handle =
          reinterpret_cast<float*>(scaleshift_memory->get_data_handle());
      std::copy(
          scale->data<float>(), scale->data<float>() + C, mem_data_handle);
      std::copy(
          shift->data<float>(), shift->data<float>() + C, mem_data_handle + C);
      return scaleshift_memory;
    }
  }

  std::shared_ptr<dnnl::memory> AcquireMeanMemory(phi::DenseTensor* mean) {
    T* mean_data = mean->mutable_data<T>(this->place_,
                                         this->fwd_pd_->mean_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->mean_desc(),
                                            mean_data);
  }

  std::shared_ptr<dnnl::memory> AcquireVarianceMemory(
      phi::DenseTensor* variance) {
    T* variance_data = variance->mutable_data<T>(
        this->place_, this->fwd_pd_->variance_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->variance_desc(),
                                            variance_data);
  }
};

template <typename T>
class LayerNormMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Y");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");

    const float epsilon = ctx.Attr<float>("epsilon");
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const bool is_test = ctx.Attr<bool>("is_test");

    auto& dev_ctx = ctx.template device_context<phi::OneDNNContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto src_tz = phi::vectorize(x->dims());
    PADDLE_ENFORCE_EQ(begin_norm_axis,
                      (src_tz.size() - 1),
                      platform::errors::InvalidArgument(
                          "MKL-DNN Layer Norm supports only last logical "
                          "axis:%d as begin_norm_axis.",
                          (src_tz.size() - 1)));

    const bool with_scaleshift = (scale && bias);
    dnnl::normalization_flags flags{};

    if (with_scaleshift) {
      flags |= dnnl::normalization_flags::use_scale_shift;
    }

    LayerNormOneDNNHandler<T> handler(
        src_tz, epsilon, flags, is_test, x, onednn_engine, ctx.GetPlace());

    auto src_memory = handler.AcquireSrcMemory(x);
    auto dst_memory = handler.AcquireDstMemory(out);

    auto layer_norm_p = handler.AcquireForwardPrimitive();

    auto& astream = phi::OneDNNContext::tls().get_stream();
    std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_SRC, *src_memory},
                                                  {DNNL_ARG_DST, *dst_memory}};

    if (!is_test) {
      auto* mean = ctx.Output<phi::DenseTensor>("Mean");
      auto* var = ctx.Output<phi::DenseTensor>("Variance");

      auto mean_memory = handler.AcquireMeanMemory(mean);
      auto variance_memory = handler.AcquireVarianceMemory(var);

      args.insert({DNNL_ARG_MEAN, *mean_memory});
      args.insert({DNNL_ARG_VARIANCE, *variance_memory});
    }

    if (with_scaleshift) {
      std::shared_ptr<dnnl::memory> scaleshift_memory =
          handler.AcquireScaleShiftMemory(scale, bias, ctx);
      args.insert({DNNL_ARG_SCALE_SHIFT, *scaleshift_memory});
    }

    layer_norm_p->execute(astream, args);
    astream.wait();

    out->set_mem_desc(dst_memory->get_desc());
  }
};

}  // namespace operators
}  // namespace paddle

// TODO(jczaja): Enable FP32 when performance is good
namespace ops = paddle::operators;
REGISTER_OP_KERNEL(layer_norm,
                   MKLDNN,
                   ::phi::CPUPlace,
                   ops::LayerNormMKLDNNOpKernel<float>,
                   ops::LayerNormMKLDNNOpKernel<paddle::platform::bfloat16>);
