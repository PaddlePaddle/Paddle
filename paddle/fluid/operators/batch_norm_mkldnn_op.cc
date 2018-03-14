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
class BatchNormMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    //    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");

    const auto *x = ctx.Input<Tensor>("X");
    //    const auto *mean = ctx.Input<Tensor>("Mean");
    //    const auto *variance = ctx.Input<Tensor>("Variance");

    auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto mkldnn_engine = dev_ctx.GetEngine();

    auto *y = ctx.Output<Tensor>("Y");
    //    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    //    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *batch_mean = ctx.Output<Tensor>("SavedMean");
    auto *batch_variance = ctx.Output<Tensor>("SavedVariance");

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *shift = ctx.Input<Tensor>("Bias");

    y->mutable_data<T>(ctx.GetPlace());
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

    unsigned flag = 0;  // mkldnn::use_scale_shift;

    using bnfwd = mkldnn::batch_normalization_forward;
    using bnfwd_desc = bnfwd::desc;
    using bnfwd_prim = bnfwd::primitive_desc;

    bnfwd_desc batch_norm_fwd_desc{propagation, src_md, epsilon, flag};
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

    //    mkldnn::memory scaleshift_memory
    //    {batch_norm_fwd_pd.weights_primitive_desc(), scaleshift_data.data()};

    mkldnn::memory mean_memory{batch_norm_fwd_pd.mean_primitive_desc(),
                               batch_mean->data<T>()};

    mkldnn::memory variance_memory{batch_norm_fwd_pd.variance_primitive_desc(),
                                   batch_variance->data<T>()};

    bnfwd batch_norm_fwd_op{batch_norm_fwd_pd, src,
                            (const mkldnn::primitive::at)mean_memory,
                            (const mkldnn::primitive::at)variance_memory, dst};

    std::vector<mkldnn::primitive> pipeline;
    pipeline.push_back(batch_norm_fwd_op);
    mkldnn::stream(mkldnn::stream::kind::lazy).submit(pipeline).wait();
  }
};
}
}

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(batch_norm, MKLDNN, paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNOpKernel<float>);
