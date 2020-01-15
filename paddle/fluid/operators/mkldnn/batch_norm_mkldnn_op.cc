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

#include "mkldnn.hpp"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using paddle::platform::MKLDNNDeviceContext;
using platform::to_void_cast;

template <typename T>
class BatchNormMKLDNNHandler
    : public platform::MKLDNNHandlerT<T, mkldnn::batch_normalization_forward,
                                      mkldnn::batch_normalization_backward> {
 public:
  BatchNormMKLDNNHandler(const std::vector<int64_t> &dims, const float &epsilon,
                         const mkldnn::normalization_flags &flags,
                         const bool &global_stats, const MKLDNNMemoryFormat fmt,
                         const platform::MKLDNNDeviceContext &dev_ctx,
                         platform::Place cpu_place,
                         const std::string &uniq_name)
      : platform::MKLDNNHandlerT<T, mkldnn::batch_normalization_forward,
                                 mkldnn::batch_normalization_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, uniq_name)) {
    auto md = mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);

    this->AcquireForwardPrimitiveDescriptor(
        global_stats == true ? mkldnn::prop_kind::forward_scoring
                             : mkldnn::prop_kind::forward_training,
        md, epsilon, flags);
  }
  BatchNormMKLDNNHandler(const std::vector<int64_t> &dims, const float &epsilon,
                         const mkldnn::normalization_flags &flags,
                         const MKLDNNMemoryFormat diff_fmt,
                         const MKLDNNMemoryFormat src_fmt,
                         const platform::MKLDNNDeviceContext &dev_ctx,
                         platform::Place cpu_place,
                         const std::string &uniq_name)
      : platform::MKLDNNHandlerT<T, mkldnn::batch_normalization_forward,
                                 mkldnn::batch_normalization_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, uniq_name)) {
    auto diff_dst_md =
        mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), diff_fmt);
    auto src_md =
        mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), src_fmt);

    this->AcquireBackwardPrimitiveDescriptor(
        mkldnn::prop_kind::backward, diff_dst_md, src_md, epsilon, flags);
  }

  std::shared_ptr<mkldnn::memory> AcquireScaleShiftMemory(T *scaleshift_data) {
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(), scaleshift_data, "@scaleshift_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffScaleShiftMemory(
      T *diff_scaleshift_data) {
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->diff_weights_desc(),
                                            diff_scaleshift_data,
                                            "@diff_scaleshift_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireMeanMemory(
      const framework::Tensor *mean) {
    const T *mean_data = mean->data<T>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->mean_desc(), to_void_cast<T>(mean_data), "@mean_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireMeanMemory(framework::Tensor *mean) {
    T *mean_data = mean->mutable_data<T>(this->place_,
                                         this->fwd_pd_->mean_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->mean_desc(),
                                            mean_data, "@mean_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireVarianceMemory(
      const framework::Tensor *variance) {
    const T *variance_data = variance->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->variance_desc(),
                                            to_void_cast<T>(variance_data),
                                            "@variance_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireVarianceMemory(
      framework::Tensor *variance) {
    T *variance_data = variance->mutable_data<T>(
        this->place_, this->fwd_pd_->variance_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->variance_desc(),
                                            variance_data, "@variance_mem_p");
  }
};

template <typename T>
class BatchNormMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool fuse_with_relu = ctx.Attr<bool>("fuse_with_relu");

    bool global_stats = is_test || use_global_stats;

    auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

    const auto *x = ctx.Input<Tensor>("X");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *shift = ctx.Input<Tensor>("Bias");

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *batch_mean = ctx.Output<Tensor>("SavedMean");
    auto *batch_variance = ctx.Output<Tensor>("SavedVariance");

    PADDLE_ENFORCE_EQ(x->layout(), DataLayout::kMKLDNN,
                      "Wrong layout set for X tensor");
    PADDLE_ENFORCE_NE(x->format(), MKLDNNMemoryFormat::undef,
                      "Wrong format set for X tensor");

    auto src_tz = paddle::framework::vectorize<int64_t>(x->dims());
    auto scale_tz = paddle::framework::vectorize<int64_t>(scale->dims());
    PADDLE_ENFORCE(scale_tz.size() == 1, "Dims of scale tensor is NOT 1");
    const unsigned int C = scale_tz[0];

    // MKLDNN requires a single piece of memory for scale and shift/bias data

    std::vector<T> scaleshift_data(scale->data<T>(), scale->data<T>() + C);
    scaleshift_data.reserve(2 * C);
    scaleshift_data.insert(scaleshift_data.end(), shift->data<T>(),
                           shift->data<T>() + C);

    // Flags are added by bitwise OR operation
    auto flags = mkldnn::normalization_flags::use_scale_shift;  // 001
    if (global_stats)
      flags |= mkldnn::normalization_flags::use_global_stats;  // 010
    if (fuse_with_relu && is_test)
      flags |= mkldnn::normalization_flags::fuse_norm_relu;  // 100

    BatchNormMKLDNNHandler<T> handler(
        src_tz, epsilon, flags, global_stats,
        platform::MKLDNNFormatForSize(src_tz.size(), x->format()), dev_ctx,
        ctx.GetPlace(), ctx.OutputName("SavedMean"));

    auto src_memory = handler.AcquireSrcMemory(x);
    auto scaleshift_memory =
        handler.AcquireScaleShiftMemory(scaleshift_data.data());
    auto dst_memory = handler.AcquireDstMemory(y);

    auto batch_norm_p = handler.AcquireForwardPrimitive();

    std::shared_ptr<memory> mean_memory;
    std::shared_ptr<memory> variance_memory;

    if (global_stats) {
      // mean and variance are taken from input Tensor
      const auto *mean = ctx.Input<Tensor>("Mean");
      const auto *variance = ctx.Input<Tensor>("Variance");

      mean_memory = handler.AcquireMeanMemory(mean);
      variance_memory = handler.AcquireVarianceMemory(variance);
    } else {
      // mean and variance are calculated and saved in output Tensor
      mean_memory = handler.AcquireMeanMemory(batch_mean);
      variance_memory = handler.AcquireVarianceMemory(batch_variance);
    }

    y->set_layout(DataLayout::kMKLDNN);
    y->set_format(platform::GetMKLDNNFormat(*dst_memory));

    mkldnn::stream astream(dev_ctx.GetEngine());
    batch_norm_p->execute(astream,
                          {{MKLDNN_ARG_SRC, *src_memory},
                           {MKLDNN_ARG_SCALE_SHIFT, *scaleshift_memory},
                           {MKLDNN_ARG_MEAN, *mean_memory},
                           {MKLDNN_ARG_VARIANCE, *variance_memory},
                           {MKLDNN_ARG_DST, *dst_memory}});
    astream.wait();

    if (!global_stats) {
      // mkldnn only compute stats for current batch
      // so we need compute momentum stats via Eigen lib
      EigenVectorArrayMap<T> batch_mean_e(
          batch_mean->mutable_data<T>(ctx.GetPlace()), C);
      EigenVectorArrayMap<T> batch_variance_e(
          batch_variance->mutable_data<T>(ctx.GetPlace()), C);

      EigenVectorArrayMap<T> running_mean_e(
          mean_out->mutable_data<T>(ctx.GetPlace()), C);
      EigenVectorArrayMap<T> running_variance_e(
          variance_out->mutable_data<T>(ctx.GetPlace()), C);

      running_mean_e =
          running_mean_e * momentum + batch_mean_e * (1. - momentum);
      running_variance_e =
          running_variance_e * momentum + batch_variance_e * (1. - momentum);
    }
  }
};

template <typename T>
class BatchNormMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();

    const float epsilon = ctx.Attr<float>("epsilon");

    const auto *x = ctx.Input<Tensor>("X");
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *shift = ctx.Input<Tensor>("Bias");
    const auto *batch_mean = ctx.Input<Tensor>("SavedMean");
    const auto *batch_variance = ctx.Input<Tensor>("SavedVariance");
    const auto *diff_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto *diff_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *diff_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *diff_shift = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    PADDLE_ENFORCE_EQ(diff_y->layout(), DataLayout::kMKLDNN,
                      "Wrong layout set for Input diff_y tensor");
    PADDLE_ENFORCE_NE(diff_y->format(), MKLDNNMemoryFormat::undef,
                      "Wrong format set for Input diff_y tensor");

    auto src_tz = paddle::framework::vectorize<int64_t>(x->dims());
    auto scale_tz = paddle::framework::vectorize<int64_t>(scale->dims());
    PADDLE_ENFORCE(scale_tz.size() == 1, "Dims of scale tensor is NOT 1");

    const unsigned int C = scale_tz[0];

    MKLDNNMemoryFormat dst_format =
        platform::MKLDNNFormatForSize(src_tz.size(), diff_y->format());

    MKLDNNMemoryFormat input_format =
        platform::MKLDNNFormatForSize(src_tz.size(), x->format());

    BatchNormMKLDNNHandler<T> handler(
        src_tz, epsilon, mkldnn::normalization_flags::use_scale_shift,
        dst_format, input_format, dev_ctx, ctx.GetPlace(),
        ctx.InputName("SavedMean"));

    // MKLDNN requires a single piece of memory for scale and shift/bias data
    const size_t scaleshift_size = 2 * C;
    std::vector<T> scaleshift_data(scale->data<T>(), scale->data<T>() + C);
    scaleshift_data.reserve(scaleshift_size);
    scaleshift_data.insert(scaleshift_data.end(), shift->data<T>(),
                           shift->data<T>() + C);

    std::vector<T> diff_scaleshift_data;
    diff_scaleshift_data.reserve(scaleshift_size);

    auto src_memory = handler.AcquireSrcMemory(x);
    auto mean_memory = handler.AcquireMeanMemory(batch_mean);
    auto variance_memory = handler.AcquireVarianceMemory(batch_variance);
    auto diff_dst_memory = handler.AcquireDiffDstMemory(diff_y);
    auto scaleshift_memory =
        handler.AcquireScaleShiftMemory(scaleshift_data.data());
    auto diff_src_memory = handler.AcquireDiffSrcMemory(diff_x);
    auto diff_scaleshift_memory =
        handler.AcquireDiffScaleShiftMemory(diff_scaleshift_data.data());

    // finally create batch_norm backward primitive
    auto batch_norm_bwd_p = handler.AcquireBackwardPrimitive();

    mkldnn::stream astream(dev_ctx.GetEngine());
    batch_norm_bwd_p->execute(
        astream, {{MKLDNN_ARG_SRC, *src_memory},
                  {MKLDNN_ARG_MEAN, *mean_memory},
                  {MKLDNN_ARG_VARIANCE, *variance_memory},
                  {MKLDNN_ARG_DIFF_DST, *diff_dst_memory},
                  {MKLDNN_ARG_SCALE_SHIFT, *scaleshift_memory},
                  {MKLDNN_ARG_DIFF_SRC, *diff_src_memory},
                  {MKLDNN_ARG_DIFF_SCALE_SHIFT, *diff_scaleshift_memory}});
    astream.wait();

    T *diff_scale_data = diff_scale->mutable_data<T>(ctx.GetPlace());
    T *diff_shift_data = diff_shift->mutable_data<T>(ctx.GetPlace());

    // copy back diff sacle/shift to output tensors (diff scale/shift)
    diff_scaleshift_data.resize(scaleshift_size);
    auto it = std::begin(diff_scaleshift_data);
    std::copy(it, std::next(it, C), diff_scale_data);
    std::copy(std::next(it, C), std::end(diff_scaleshift_data),
              diff_shift_data);

    // set layout/format of output tensors
    diff_x->set_layout(DataLayout::kMKLDNN);
    diff_x->set_format(platform::GetMKLDNNFormat(*diff_src_memory));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(batch_norm, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(batch_norm_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNGradOpKernel<float>);
