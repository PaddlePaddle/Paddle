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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using mkldnn::memory;

template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

namespace {
template <typename T>
struct bn_type_traits {
  using op_type = T;
  using op_desc = typename op_type::desc;
  using op_prim = typename op_type::primitive_desc;
};

template <typename T, typename Container>
void copy_to_weights(T scale_begin, T scale_end, T shift_begin, T shift_end,
                     Container &c) {
  auto it = std::begin(c);

  std::copy(scale_begin, scale_end, std::inserter(c, it));
  std::copy(
      shift_begin, shift_end,
      std::inserter(c, std::next(it, std::distance(scale_begin, scale_end))));
}

template <typename Op, typename... Args>
void run_batch_norm_op(Args &&... args) {
  Op batch_norm_fwd_op{args...};

  std::vector<mkldnn::primitive> pipeline;
  pipeline.push_back(batch_norm_fwd_op);
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}
}  // namespace

template <typename T>
class BatchNormMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");

    const auto *x = ctx.Input<Tensor>("X");
    const auto *mean = ctx.Input<Tensor>("Mean");
    const auto *variance = ctx.Input<Tensor>("Variance");

    auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *batch_mean = ctx.Output<Tensor>("SavedMean");
    auto *batch_variance = ctx.Output<Tensor>("SavedVariance");

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *shift = ctx.Input<Tensor>("Bias");

    y->mutable_data<T>(ctx.GetPlace());
    mean_out->mutable_data<T>(ctx.GetPlace());
    variance_out->mutable_data<T>(ctx.GetPlace());

    if (!is_test) {
      batch_mean->mutable_data<T>(ctx.GetPlace());
      batch_variance->mutable_data<T>(ctx.GetPlace());
    }

    auto propagation = is_test == true ? mkldnn::prop_kind::forward_scoring
                                       : mkldnn::prop_kind::forward_training;

    auto x_dims = x->dims();
    const int in = x_dims[0];
    const int ic = x_dims[1];
    const int ih = x_dims[2];
    const int iw = x_dims[3];

    memory::dims tz = {in, ic, ih, iw};

    memory::desc src_md{{tz}, memory::data_type::f32, memory::format::nchw};
    memory::desc dst_md{{tz}, memory::data_type::f32, memory::format::nchw};

    mkldnn::memory::primitive_desc src_pd{src_md, mkldnn_engine};
    mkldnn::memory::primitive_desc dst_pd{dst_md, mkldnn_engine};

    mkldnn::memory src{src_pd,
                       static_cast<void *>(const_cast<T *>(x->data<T>()))};
    mkldnn::memory dst{dst_pd, y->data<T>()};

    unsigned flags = mkldnn::use_scale_shift;

    if (is_test) flags |= mkldnn::use_global_stats;

    using bn_fwd_types = bn_type_traits<mkldnn::batch_normalization_forward>;

    bn_fwd_types::op_desc batch_norm_fwd_desc{propagation, src_md, epsilon,
                                              flags};
    typename bn_fwd_types::op_prim batch_norm_fwd_pd{batch_norm_fwd_desc,
                                                     mkldnn_engine};

    // MKLDNN requires a single piece of memory for scale and shift/bias data
    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(2 * ic);

    copy_to_weights(scale->data<T>(), scale->data<T>() + ic, shift->data<T>(),
                    shift->data<T>() + ic, scaleshift_data);

    mkldnn::memory scaleshift_memory{batch_norm_fwd_pd.weights_primitive_desc(),
                                     scaleshift_data.data()};

    if (is_test) {
      mkldnn::memory mean_memory{
          batch_norm_fwd_pd.mean_primitive_desc(),
          static_cast<void *>(const_cast<T *>(mean->data<T>()))};

      mkldnn::memory variance_memory{
          batch_norm_fwd_pd.variance_primitive_desc(),
          static_cast<void *>(const_cast<T *>(variance->data<T>()))};

      run_batch_norm_op<typename bn_fwd_types::op_type>(
          batch_norm_fwd_pd, src, (const mkldnn::primitive::at &)mean_memory,
          (const mkldnn::primitive::at &)variance_memory, scaleshift_memory,
          dst);
    } else {
      mkldnn::memory mean_memory{
          batch_norm_fwd_pd.mean_primitive_desc(),
          static_cast<void *>(const_cast<T *>(batch_mean->data<T>()))};

      mkldnn::memory variance_memory{
          batch_norm_fwd_pd.variance_primitive_desc(),
          static_cast<void *>(const_cast<T *>(batch_variance->data<T>()))};

      run_batch_norm_op<bn_fwd_types::op_type>(batch_norm_fwd_pd, src,
                                               scaleshift_memory, dst,
                                               mean_memory, variance_memory);
    }

    if (!is_test) {
      const int sample_size = x->numel() / in / ic;

      // saved_xx is use just in this batch of data
      EigenVectorArrayMap<T> saved_mean_e(
          batch_mean->mutable_data<T>(ctx.GetPlace()), ic);
      EigenVectorArrayMap<T> saved_variance_e(
          batch_variance->mutable_data<T>(ctx.GetPlace()), ic);
      saved_mean_e.setZero();
      saved_variance_e.setZero();

      ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, in * ic);
      for (int nc = 0; nc < in * ic; ++nc) {
        saved_mean_e(nc % ic) += x_arr.col(nc).sum();
      }
      saved_mean_e /= in * sample_size;
      for (int nc = 0; nc < in * ic; ++nc) {
        saved_variance_e(nc % ic) +=
            (x_arr.col(nc) - saved_mean_e(nc % ic)).matrix().squaredNorm();
      }
      saved_variance_e /= in * sample_size;

      ConstEigenVectorArrayMap<T> mean_arr{mean->data<T>(), ic};
      ConstEigenVectorArrayMap<T> variance_arr{variance->data<T>(), ic};

      EigenVectorArrayMap<T> running_mean_arr(
          mean_out->mutable_data<T>(ctx.GetPlace()), ic);
      EigenVectorArrayMap<T> running_var_arr(
          variance_out->mutable_data<T>(ctx.GetPlace()), ic);
      running_mean_arr = mean_arr * momentum + saved_mean_e * (1. - momentum);
      running_var_arr =
          variance_arr * momentum + saved_variance_e * (1. - momentum);
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
    // SavedVariance have been reverted in forward operator
    const auto *batch_inv_variance = ctx.Input<Tensor>("SavedVariance");

    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_shift = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    d_scale->mutable_data<T>(ctx.GetPlace());
    d_shift->mutable_data<T>(ctx.GetPlace());

    auto x_dims = x->dims();
    const int in = x_dims[0];
    const int ic = x_dims[1];
    const int ih = x_dims[2];
    const int iw = x_dims[3];

    memory::dims tz = {in, ic, ih, iw};

    unsigned flags = mkldnn::use_scale_shift | !mkldnn::use_global_stats;

    memory::desc src_md{{tz}, memory::data_type::f32, memory::format::nchw};
    memory::desc dst_md{{tz}, memory::data_type::f32, memory::format::nchw};
    memory::desc diff_src_md{
        {tz}, memory::data_type::f32, memory::format::nchw};
    memory::desc diff_dst_md{
        {tz}, memory::data_type::f32, memory::format::nchw};

    using bn_bwd_types = bn_type_traits<mkldnn::batch_normalization_backward>;
    using bn_fwd_types = bn_type_traits<mkldnn::batch_normalization_forward>;

    typename bn_fwd_types::op_desc batch_norm_fwd_desc{
        mkldnn::prop_kind::forward_training, src_md, epsilon, flags};
    typename bn_fwd_types::op_prim batch_norm_fwd_pd{batch_norm_fwd_desc,
                                                     mkldnn_engine};

    typename bn_bwd_types::op_desc batch_norm_bwd_desc{
        mkldnn::prop_kind::backward, diff_dst_md, dst_md, epsilon, flags};
    typename bn_bwd_types::op_prim batch_norm_bwd_pd{
        batch_norm_bwd_desc, mkldnn_engine, batch_norm_fwd_pd};

    auto src =
        mkldnn::memory{{src_md, mkldnn_engine},
                       static_cast<void *>(const_cast<T *>(x->data<T>()))};

    auto mean = mkldnn::memory{
        batch_norm_bwd_pd.mean_primitive_desc(),
        static_cast<void *>(const_cast<T *>(batch_mean->data<T>()))};

    auto variance = mkldnn::memory{
        batch_norm_bwd_pd.variance_primitive_desc(),
        static_cast<void *>(const_cast<T *>(batch_inv_variance->data<T>()))};

    auto diff_dst =
        mkldnn::memory{{diff_dst_md, mkldnn_engine},
                       static_cast<void *>(const_cast<T *>(d_y->data<T>()))};

    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(2 * ic);
    copy_to_weights(scale->data<T>(), scale->data<T>() + ic, shift->data<T>(),
                    shift->data<T>() + ic, scaleshift_data);

    mkldnn::memory scaleshift_memory{batch_norm_bwd_pd.weights_primitive_desc(),
                                     scaleshift_data.data()};

    std::vector<T> diff_scaleshift_data;
    diff_scaleshift_data.reserve(2 * ic);
    copy_to_weights(d_scale->data<T>(), d_scale->data<T>() + ic,
                    d_shift->data<T>(), d_shift->data<T>() + ic,
                    diff_scaleshift_data);

    mkldnn::memory diff_scaleshift_memory{
        batch_norm_bwd_pd.diff_weights_primitive_desc(),
        diff_scaleshift_data.data()};

    auto diff_src = mkldnn::memory{{diff_src_md, mkldnn_engine},
                                   static_cast<void *>(d_x->data<T>())};

    run_batch_norm_op<bn_bwd_types::op_type>(
        batch_norm_bwd_pd, src, mean, variance, diff_dst, scaleshift_memory,
        diff_src, diff_scaleshift_memory);

    auto it = std::begin(diff_scaleshift_data);

    std::copy(it, std::next(it, ic), d_scale->data<T>());
    std::copy(std::next(it, ic), std::end(diff_scaleshift_data),
              d_shift->data<T>());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(batch_norm, MKLDNN, paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(batch_norm_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNGradOpKernel<float>);
