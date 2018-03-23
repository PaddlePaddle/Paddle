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
    batch_mean->mutable_data<T>(ctx.GetPlace());
    batch_variance->mutable_data<T>(ctx.GetPlace());

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

    unsigned flags = mkldnn::use_scale_shift | !mkldnn::use_global_stats;

    using bnfwd = mkldnn::batch_normalization_forward;
    using bnfwd_desc = bnfwd::desc;
    using bnfwd_prim = bnfwd::primitive_desc;

    bnfwd_desc batch_norm_fwd_desc{propagation, src_md, epsilon, flags};
    bnfwd_prim batch_norm_fwd_pd{batch_norm_fwd_desc, mkldnn_engine};

    // MKLDNN requires a single piece of memory for scale and shift/bias data
    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(2 * ic);

    typename std::vector<T>::iterator it = std::begin(scaleshift_data);

    std::copy(scale->data<T>(), scale->data<T>() + ic,
              std::inserter(scaleshift_data, it));
    std::advance(it, ic);
    std::copy(shift->data<T>(), shift->data<T>() + ic,
              std::inserter(scaleshift_data, it));

    mkldnn::memory scaleshift_memory{batch_norm_fwd_pd.weights_primitive_desc(),
                                     scaleshift_data.data()};

    mkldnn::memory mean_memory{
        batch_norm_fwd_pd.mean_primitive_desc(),
        static_cast<void *>(const_cast<T *>(batch_mean->data<T>()))};

    mkldnn::memory variance_memory{
        batch_norm_fwd_pd.variance_primitive_desc(),
        static_cast<void *>(const_cast<T *>(batch_variance->data<T>()))};

    std::vector<T> mean_copy;
    mean_copy.reserve(ic);
    std::copy(mean->data<T>(), mean->data<T>() + ic,
              std::inserter(mean_copy, std::begin(mean_copy)));

    bnfwd batch_norm_fwd_op{batch_norm_fwd_pd, src,
                            scaleshift_memory, dst,
                            mean_memory,       variance_memory};

    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(batch_norm_fwd_op);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

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

      EigenVectorArrayMap<T> saved_inv_std(
          ctx.Output<Tensor>("SavedVariance")->data<T>(), ic);
      // inverse SavedVariance first, gradient will use it too.
      saved_inv_std = (saved_inv_std + epsilon).inverse().sqrt();
    }
  }
};
}
}

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(batch_norm, MKLDNN, paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNOpKernel<float>);
