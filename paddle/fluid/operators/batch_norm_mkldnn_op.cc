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
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;
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
                     Container *c) {
  auto it = std::begin(*c);

  std::copy(scale_begin, scale_end, std::inserter(*c, it));
  std::copy(
      shift_begin, shift_end,
      std::inserter(*c, std::next(it, std::distance(scale_begin, scale_end))));
}

template <typename Op, typename... Args>
void run_batch_norm_op(Args &&... args) {
  Op batch_norm_op{args...};

  std::vector<mkldnn::primitive> pipeline;
  pipeline.push_back(batch_norm_op);
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}

template <typename T>
inline void *cast_const_to_void(const T *t) {
  return static_cast<void *>(const_cast<T *>(t));
}
}  // namespace

template <typename T>
class BatchNormMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto data_layout_str = ctx.Attr<std::string>("data_layout");
    auto data_layout = framework::StringToDataLayout(data_layout_str);
    PADDLE_ENFORCE(data_layout == framework::DataLayout::kNCHW,
                   "MKLDNN batch normalization handles only NCHW data layout");

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

    auto dims = paddle::framework::vectorize2int(x->dims());

    auto src_md =
        MKLDNNMemDesc(dims, memory::data_type::f32, memory::format::nchw);
    auto dst_md =
        MKLDNNMemDesc(dims, memory::data_type::f32, memory::format::nchw);

    auto src_pd = mkldnn::memory::primitive_desc{src_md, mkldnn_engine};
    auto dst_pd = mkldnn::memory::primitive_desc{dst_md, mkldnn_engine};

    auto src = mkldnn::memory{src_pd, cast_const_to_void(x->data<T>())};
    auto dst = mkldnn::memory{dst_pd, y->data<T>()};

    unsigned flags = mkldnn::use_scale_shift;
    if (is_test) flags |= mkldnn::use_global_stats;

    using bn_fwd_types = bn_type_traits<mkldnn::batch_normalization_forward>;
    auto batch_norm_fwd_desc =
        bn_fwd_types::op_desc{propagation, src_md, epsilon, flags};
    auto batch_norm_fwd_pd =
        bn_fwd_types::op_prim{batch_norm_fwd_desc, mkldnn_engine};

    const unsigned int ic = dims[1];

    // MKLDNN requires a single piece of memory for scale and shift/bias data
    const size_t scaleshift_size = 2 * ic;
    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(scaleshift_size);

    copy_to_weights(scale->data<T>(), scale->data<T>() + ic, shift->data<T>(),
                    shift->data<T>() + ic, &scaleshift_data);

    auto scaleshift_memory = mkldnn::memory{
        batch_norm_fwd_pd.weights_primitive_desc(), scaleshift_data.data()};

    if (is_test) {
      auto mean_memory = mkldnn::memory{batch_norm_fwd_pd.mean_primitive_desc(),
                                        cast_const_to_void(mean->data<T>())};

      auto variance_memory =
          mkldnn::memory{batch_norm_fwd_pd.variance_primitive_desc(),
                         cast_const_to_void(variance->data<T>())};

      run_batch_norm_op<typename bn_fwd_types::op_type>(
          batch_norm_fwd_pd, src, (const mkldnn::primitive::at &)mean_memory,
          (const mkldnn::primitive::at &)variance_memory, scaleshift_memory,
          dst);
    } else {
      auto mean_memory =
          mkldnn::memory{batch_norm_fwd_pd.mean_primitive_desc(),
                         cast_const_to_void(batch_mean->data<T>())};

      auto variance_memory =
          mkldnn::memory{batch_norm_fwd_pd.variance_primitive_desc(),
                         cast_const_to_void(batch_variance->data<T>())};

      run_batch_norm_op<bn_fwd_types::op_type>(batch_norm_fwd_pd, src,
                                               scaleshift_memory, dst,
                                               mean_memory, variance_memory);
    }

    if (!is_test) {
      const unsigned int in = dims[0];
      const unsigned int sample_size = x->numel() / in / ic;

      // saved_xx is use just in this batch of data
      EigenVectorArrayMap<T> saved_mean_e(
          batch_mean->mutable_data<T>(ctx.GetPlace()), ic);
      EigenVectorArrayMap<T> saved_variance_e(
          batch_variance->mutable_data<T>(ctx.GetPlace()), ic);
      saved_mean_e.setZero();
      saved_variance_e.setZero();

      const unsigned int x_arr_size = in * ic;
      ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, x_arr_size);
      for (unsigned int nc = 0; nc < x_arr_size; ++nc) {
        saved_mean_e(nc % ic) += x_arr.col(nc).sum();
      }
      saved_mean_e /= in * sample_size;
      for (unsigned int nc = 0; nc < x_arr_size; ++nc) {
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

      auto one_minus_momentum = 1. - momentum;
      running_mean_arr =
          mean_arr * momentum + saved_mean_e * one_minus_momentum;
      running_var_arr =
          variance_arr * momentum + saved_variance_e * one_minus_momentum;
    }
  }
};

template <typename T>
class BatchNormMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto data_layout_str = ctx.Attr<std::string>("data_layout");
    auto data_layout = framework::StringToDataLayout(data_layout_str);
    PADDLE_ENFORCE(data_layout == framework::DataLayout::kNCHW,
                   "MKLDNN batch normalization handles only NCHW data layout");

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

    diff_x->mutable_data<T>(ctx.GetPlace());
    diff_scale->mutable_data<T>(ctx.GetPlace());
    diff_shift->mutable_data<T>(ctx.GetPlace());

    auto dims = paddle::framework::vectorize2int(x->dims());
    unsigned flags = mkldnn::use_scale_shift | !mkldnn::use_global_stats;

    auto src_md =
        MKLDNNMemDesc(dims, memory::data_type::f32, memory::format::nchw);
    auto dst_md =
        MKLDNNMemDesc(dims, memory::data_type::f32, memory::format::nchw);
    auto diff_src_md =
        MKLDNNMemDesc(dims, memory::data_type::f32, memory::format::nchw);
    auto diff_dst_md =
        MKLDNNMemDesc(dims, memory::data_type::f32, memory::format::nchw);

    using bn_bwd_types = bn_type_traits<mkldnn::batch_normalization_backward>;
    using bn_fwd_types = bn_type_traits<mkldnn::batch_normalization_forward>;

    auto batch_norm_fwd_desc = bn_fwd_types::op_desc{
        mkldnn::prop_kind::forward_training, src_md, epsilon, flags};
    auto batch_norm_fwd_pd =
        bn_fwd_types::op_prim{batch_norm_fwd_desc, mkldnn_engine};

    auto batch_norm_bwd_desc = bn_bwd_types::op_desc{
        mkldnn::prop_kind::backward, diff_dst_md, dst_md, epsilon, flags};
    auto batch_norm_bwd_pd = bn_bwd_types::op_prim{
        batch_norm_bwd_desc, mkldnn_engine, batch_norm_fwd_pd};

    auto src = mkldnn::memory{{src_md, mkldnn_engine},
                              cast_const_to_void(x->data<T>())};

    auto mean = mkldnn::memory{batch_norm_bwd_pd.mean_primitive_desc(),
                               cast_const_to_void(batch_mean->data<T>())};

    auto variance =
        mkldnn::memory{batch_norm_bwd_pd.variance_primitive_desc(),
                       cast_const_to_void(batch_variance->data<T>())};

    auto diff_dst = mkldnn::memory{{diff_dst_md, mkldnn_engine},
                                   cast_const_to_void(diff_y->data<T>())};

    const unsigned int ic = dims[1];

    const size_t scaleshift_size = 2 * ic;

    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(scaleshift_size);
    copy_to_weights(scale->data<T>(), scale->data<T>() + ic, shift->data<T>(),
                    shift->data<T>() + ic, &scaleshift_data);

    auto scaleshift_memory = mkldnn::memory{
        batch_norm_bwd_pd.weights_primitive_desc(), scaleshift_data.data()};

    std::vector<T> diff_scaleshift_data;
    diff_scaleshift_data.reserve(scaleshift_size);
    copy_to_weights(diff_scale->data<T>(), diff_scale->data<T>() + ic,
                    diff_shift->data<T>(), diff_shift->data<T>() + ic,
                    &diff_scaleshift_data);

    auto diff_scaleshift_memory =
        mkldnn::memory{batch_norm_bwd_pd.diff_weights_primitive_desc(),
                       diff_scaleshift_data.data()};

    auto diff_src = mkldnn::memory{{diff_src_md, mkldnn_engine},
                                   static_cast<void *>(diff_x->data<T>())};

    run_batch_norm_op<bn_bwd_types::op_type>(
        batch_norm_bwd_pd, src, mean, variance, diff_dst, scaleshift_memory,
        diff_src, diff_scaleshift_memory);

    auto it = std::begin(diff_scaleshift_data);
    std::copy(it, std::next(it, ic), diff_scale->data<T>());
    std::copy(std::next(it, ic), std::end(diff_scaleshift_data),
              diff_shift->data<T>());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(batch_norm, MKLDNN, paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(batch_norm_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNGradOpKernel<float>);
