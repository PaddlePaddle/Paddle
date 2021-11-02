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

#include "paddle/fluid/operators/layer_norm_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

template <typename T>
class LayerNormMKLDNNHandler : public platform::MKLDNNHandlerNoCachingT<
                                   T, dnnl::layer_normalization_forward> {
 public:
  LayerNormMKLDNNHandler(const std::vector<int64_t>& dims, const float& epsilon,
                         const dnnl::normalization_flags& flags,
                         const bool& is_test, const MKLDNNMemoryFormat fmt,
                         const mkldnn::engine engine, platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::layer_normalization_forward>(
            engine, cpu_place) {
    auto md = dnnl::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);
    if (!is_test) {
      // TODO(grygielski) Delete forcing stats_md after DNNL 1.2 is introduced
      auto stats_md = dnnl::memory::desc(
          {begin(dims), end(dims) - 1}, platform::MKLDNNGetDataType<float>(),
          platform::MKLDNNFormatForSize(dims.size() - 1,
                                        MKLDNNMemoryFormat::nchw));
      this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                              md, stats_md, epsilon, flags);
    } else {
      this->AcquireForwardPrimitiveDescriptor(
          dnnl::prop_kind::forward_inference, md, epsilon, flags);
    }
  }

  std::shared_ptr<dnnl::memory> AcquireScaleShiftMemory(
      std::vector<float>& scaleshift_data) {
    // scaleshift_data comes from temporary buffer so we need to copy it into
    // created memory primitivie
    auto scaleshift_mem =
        this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc());
    auto data_ptr = scaleshift_mem->get_data_handle();
    std::size_t num_bytes = scaleshift_data.size() * sizeof(float);
    std::memcpy(data_ptr, scaleshift_data.data(), num_bytes);
    return scaleshift_mem;
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
    auto* y = ctx.Output<Tensor>("Y");

    const float epsilon = ctx.Attr<float>("epsilon");
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const bool is_test = ctx.Attr<bool>("is_test");

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto src_tz = paddle::framework::vectorize(x->dims());
    PADDLE_ENFORCE_EQ(begin_norm_axis, (src_tz.size() - 1),
                      platform::errors::InvalidArgument(
                          "MKL-DNN Layer Norm supports only last logical "
                          "axis:%d as begin_norm_axis.",
                          (src_tz.size() - 1)));

    y->mutable_data<T>(ctx.GetPlace());
    const bool with_scaleshift = (scale && bias);
    dnnl::normalization_flags flags{};

    if (with_scaleshift) {
      flags |= dnnl::normalization_flags::use_scale_shift;
    }

    LayerNormMKLDNNHandler<T> handler(src_tz, epsilon, flags, is_test,
                                      x->format(), mkldnn_engine,
                                      ctx.GetPlace());

    auto src_memory = handler.AcquireSrcMemory(x);
    auto dst_memory = handler.AcquireDstMemory(y);

    auto layer_norm_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    std::unordered_map<int, dnnl::memory> args;

    args.insert({DNNL_ARG_SRC, *src_memory});
    args.insert({DNNL_ARG_DST, *dst_memory});

    if (!is_test) {
      auto* mean = ctx.Output<Tensor>("Mean");
      auto* var = ctx.Output<Tensor>("Variance");
      mean->mutable_data<T>(ctx.GetPlace());
      var->mutable_data<T>(ctx.GetPlace());

      auto mean_memory = handler.AcquireMeanMemory(mean);
      auto variance_memory = handler.AcquireVarianceMemory(var);

      args.insert({DNNL_ARG_MEAN, *mean_memory});
      args.insert({DNNL_ARG_VARIANCE, *variance_memory});
    }

    std::shared_ptr<mkldnn::memory> scaleshift_memory;
    if (with_scaleshift) {
      auto scale_tz = paddle::framework::vectorize(scale->dims());
      const unsigned int C = scale_tz[0];

      // MKLDNN requires a single piece of memory for scale and shift/bias
      // data
      std::vector<float> scaleshift_data;
      scaleshift_data.reserve(2 * C);
      scaleshift_data.insert(scaleshift_data.begin(), scale->data<float>(),
                             scale->data<float>() + C);

      scaleshift_data.insert(scaleshift_data.end(), bias->data<float>(),
                             bias->data<float>() + C);

      scaleshift_memory = handler.AcquireScaleShiftMemory(scaleshift_data);
      args.insert({DNNL_ARG_SCALE_SHIFT, *scaleshift_memory});
    }

    layer_norm_p->execute(astream, args);
    astream.wait();

    y->set_layout(DataLayout::kMKLDNN);
    y->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

}  // namespace operators
}  // namespace paddle

// TODO(jczaja): Enable FP32 when performance is good
namespace ops = paddle::operators;
REGISTER_OP_KERNEL(layer_norm, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::LayerNormMKLDNNOpKernel<paddle::platform::bfloat16>);
