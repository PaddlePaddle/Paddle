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
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace operators {

template <typename T>
class LayerNormMKLDNNHandler : public platform::MKLDNNHandlerNoCachingT<
                                   T, dnnl::layer_normalization_forward> {
 public:
  LayerNormMKLDNNHandler(const std::vector<int64_t>& dims, const float& epsilon,
                         const dnnl::normalization_flags& flags,
                         const bool& is_test, const Tensor* x,
                         const dnnl::engine engine, platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::layer_normalization_forward>(
            engine, cpu_place) {
    if (!is_test) {
      // TODO(grygielski) Delete forcing stats_md after DNNL 1.2 is introduced
      auto stats_md = dnnl::memory::desc(
          {begin(dims), end(dims) - 1}, platform::MKLDNNGetDataType<float>(),
          platform::GetPlainMKLDNNFormat(dims.size() - 1));
      this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                              x->mem_desc(), stats_md, epsilon,
                                              flags);
    } else {
      this->AcquireForwardPrimitiveDescriptor(
          dnnl::prop_kind::forward_inference, x->mem_desc(), epsilon, flags);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireScaleShiftMemory(const Tensor* scale,
                                                        const Tensor* shift) {
    // OneDNN requires a single piece of memory for scale and shift data
    const unsigned int C = phi::vectorize(scale->dims())[0];

    auto scaleshift_memory =
        this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc());

    auto mem_data_handle =
        reinterpret_cast<float*>(scaleshift_memory->get_data_handle());
    std::copy(scale->data<float>(), scale->data<float>() + C, mem_data_handle);
    std::copy(shift->data<float>(), shift->data<float>() + C,
              mem_data_handle + C);
    return scaleshift_memory;
  }

  std::shared_ptr<dnnl::memory> AcquireMeanMemory(framework::Tensor* mean) {
    T* mean_data = mean->mutable_data<T>(this->place_,
                                         this->fwd_pd_->mean_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->mean_desc(),
                                            mean_data);
  }

  std::shared_ptr<dnnl::memory> AcquireVarianceMemory(
      framework::Tensor* variance) {
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
    auto* x = ctx.Input<Tensor>("X");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* out = ctx.Output<Tensor>("Y");

    const float epsilon = ctx.Attr<float>("epsilon");
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const bool is_test = ctx.Attr<bool>("is_test");

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto src_tz = phi::vectorize(x->dims());
    PADDLE_ENFORCE_EQ(begin_norm_axis, (src_tz.size() - 1),
                      platform::errors::InvalidArgument(
                          "MKL-DNN Layer Norm supports only last logical "
                          "axis:%d as begin_norm_axis.",
                          (src_tz.size() - 1)));

    const bool with_scaleshift = (scale && bias);
    dnnl::normalization_flags flags{};

    if (with_scaleshift) {
      flags |= dnnl::normalization_flags::use_scale_shift;
    }

    LayerNormMKLDNNHandler<T> handler(src_tz, epsilon, flags, is_test, x,
                                      mkldnn_engine, ctx.GetPlace());

    auto src_memory = handler.AcquireSrcMemory(x);
    auto dst_memory = handler.AcquireDstMemory(out);

    auto layer_norm_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_SRC, *src_memory},
                                                  {DNNL_ARG_DST, *dst_memory}};

    if (!is_test) {
      auto* mean = ctx.Output<Tensor>("Mean");
      auto* var = ctx.Output<Tensor>("Variance");

      auto mean_memory = handler.AcquireMeanMemory(mean);
      auto variance_memory = handler.AcquireVarianceMemory(var);

      args.insert({DNNL_ARG_MEAN, *mean_memory});
      args.insert({DNNL_ARG_VARIANCE, *variance_memory});
    }

    if (with_scaleshift) {
      std::shared_ptr<dnnl::memory> scaleshift_memory =
          handler.AcquireScaleShiftMemory(scale, bias);
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
REGISTER_OP_KERNEL(layer_norm, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::LayerNormMKLDNNOpKernel<float>,
                   ops::LayerNormMKLDNNOpKernel<paddle::platform::bfloat16>);
