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

using batch_norm_bwd = mkldnn::batch_normalization_backward;
using batch_norm_fwd = mkldnn::batch_normalization_forward;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;
using platform::to_void_cast;

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

    PADDLE_ENFORCE(x->layout() == DataLayout::kMKLDNN &&
                       x->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input x tensor");

    const T *x_data = x->data<T>();
    const T *mean_data = mean->data<T>();
    const T *variance_data = variance->data<T>();
    T *y_data = y->mutable_data<T>(ctx.GetPlace());
    T *mean_out_data = mean_out->mutable_data<T>(ctx.GetPlace());
    T *variance_out_data = variance_out->mutable_data<T>(ctx.GetPlace());
    T *batch_mean_data = nullptr;
    T *batch_variance_data = nullptr;

    if (!is_test) {
      batch_mean_data = batch_mean->mutable_data<T>(ctx.GetPlace());
      batch_variance_data = batch_variance->mutable_data<T>(ctx.GetPlace());
    }

    auto propagation = is_test == true ? mkldnn::prop_kind::forward_scoring
                                       : mkldnn::prop_kind::forward_training;

    auto src_tz = paddle::framework::vectorize2int(x->dims());
    auto scale_tz = paddle::framework::vectorize2int(scale->dims());
    PADDLE_ENFORCE(scale_tz.size() == 1, "Dims of scale tensor is NOT 1");
    const unsigned int ic = scale_tz[0];

    unsigned flags = mkldnn::use_scale_shift;
    if (is_test) flags |= mkldnn::use_global_stats;

    // create mkldnn memory from input x tensor
    auto src_memory =
        memory({{{src_tz}, memory::data_type::f32, x->format()}, mkldnn_engine},
               to_void_cast(x_data));

    // create primitive descriptor for batch norm forward
    using bn_fwd_types = bn_type_traits<mkldnn::batch_normalization_forward>;
    auto batch_norm_fwd_desc = bn_fwd_types::op_desc{
        propagation, src_memory.get_primitive_desc().desc(), epsilon, flags};
    std::shared_ptr<batch_norm_fwd::primitive_desc> batch_norm_fwd_pd =
        std::shared_ptr<batch_norm_fwd::primitive_desc>(
            new batch_norm_fwd::primitive_desc(batch_norm_fwd_desc,
                                               mkldnn_engine));

    // Save the pd to be used in backward pass
    const std::string key = ctx.op().Output("SavedMean");
    const std::string key_batch_norm_fwd_pd = key + "@bn_fwd_pd";
    dev_ctx.SetBlob(key_batch_norm_fwd_pd, batch_norm_fwd_pd);

    // MKLDNN requires a single piece of memory for scale and shift/bias data
    const size_t scaleshift_size = 2 * ic;
    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(scaleshift_size);

    copy_to_weights(scale->data<T>(), scale->data<T>() + ic, shift->data<T>(),
                    shift->data<T>() + ic, &scaleshift_data);

    // crate mkldnn memory for weights(scale/shift)
    auto scaleshift_memory = memory(batch_norm_fwd_pd->weights_primitive_desc(),
                                    scaleshift_data.data());

    // create mkldnn memory for output y tensor
    auto dst_memory = memory(batch_norm_fwd_pd->dst_primitive_desc(), y_data);

    if (is_test) {
      // create mkldnn memory for stats (as input)
      auto mean_memory = memory(batch_norm_fwd_pd->mean_primitive_desc(),
                                to_void_cast(mean_data));
      auto variance_memory =
          memory(batch_norm_fwd_pd->variance_primitive_desc(),
                 to_void_cast(variance_data));

      run_batch_norm_op<typename bn_fwd_types::op_type>(
          *batch_norm_fwd_pd, src_memory,
          (const mkldnn::primitive::at &)mean_memory,
          (const mkldnn::primitive::at &)variance_memory, scaleshift_memory,
          dst_memory);
    } else {
      // create mkldnn memory for stats (as output)
      auto mean_memory =
          memory(batch_norm_fwd_pd->mean_primitive_desc(), batch_mean_data);
      auto variance_memory = memory(
          batch_norm_fwd_pd->variance_primitive_desc(), batch_variance_data);

      run_batch_norm_op<bn_fwd_types::op_type>(*batch_norm_fwd_pd, src_memory,
                                               scaleshift_memory, dst_memory,
                                               mean_memory, variance_memory);
    }

    if (!is_test) {
      // mkldnn only compute stats for current batch
      // so we need compute momentum stats via Eigen lib
      EigenVectorArrayMap<T> batch_mean_e(batch_mean_data, ic);
      EigenVectorArrayMap<T> batch_variance_e(batch_variance_data, ic);
      ConstEigenVectorArrayMap<T> mean_e(mean_data, ic);
      ConstEigenVectorArrayMap<T> variance_e{variance_data, ic};

      EigenVectorArrayMap<T> running_mean_e(mean_out_data, ic);
      EigenVectorArrayMap<T> running_variance_e(variance_out_data, ic);

      auto one_minus_momentum = 1. - momentum;
      running_mean_e = mean_e * momentum + batch_mean_e * one_minus_momentum;
      running_variance_e =
          variance_e * momentum + batch_variance_e * one_minus_momentum;
    }

    y->set_layout(DataLayout::kMKLDNN);
    y->set_format(
        (memory::format)dst_memory.get_primitive_desc().desc().data.format);
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

    PADDLE_ENFORCE(diff_y->layout() == DataLayout::kMKLDNN &&
                       diff_y->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input diff_y tensor");

    const T *x_data = x->data<T>();
    const T *diff_y_data = diff_y->data<T>();
    const T *batch_mean_data = batch_mean->data<T>();
    const T *batch_variance_data = batch_variance->data<T>();
    const T *scale_data = scale->data<T>();
    const T *shift_data = shift->data<T>();
    T *diff_x_data = diff_x->mutable_data<T>(ctx.GetPlace());
    T *diff_scale_data = diff_scale->mutable_data<T>(ctx.GetPlace());
    T *diff_shift_data = diff_shift->mutable_data<T>(ctx.GetPlace());

    auto src_tz = paddle::framework::vectorize2int(x->dims());
    auto diff_src_tz = src_tz;
    auto dst_tz = src_tz;
    auto diff_dst_tz = dst_tz;
    auto scale_tz = paddle::framework::vectorize2int(scale->dims());
    PADDLE_ENFORCE(scale_tz.size() == 1, "Dims of scale tensor is NOT 1");

    const unsigned int ic = scale_tz[0];

    // Retrieve bn_fwd_pd from device context
    const std::string key = ctx.op().Input("SavedMean");
    const std::string key_batch_norm_fwd_pd = key + "@bn_fwd_pd";
    auto batch_norm_fwd_pd =
        std::static_pointer_cast<batch_norm_fwd::primitive_desc>(
            dev_ctx.GetBlob(key_batch_norm_fwd_pd));
    PADDLE_ENFORCE(batch_norm_fwd_pd != nullptr,
                   "Fail to find batch_norm_fwd_pd in device context");

    using bn_bwd_types = bn_type_traits<mkldnn::batch_normalization_backward>;

    // create mkldnn memory from input diff_y tensor
    auto user_diff_dst_memory =
        memory({{{diff_dst_tz}, memory::data_type::f32, diff_y->format()},
                mkldnn_engine},
               to_void_cast(diff_y_data));

    // create mkldnn memory from input x tensor
    auto src_memory =
        memory({{{src_tz}, memory::data_type::f32, x->format()}, mkldnn_engine},
               to_void_cast(x_data));

    // for diff_dst, try to use same format as dst in forward pass
    auto diff_dst_pd = batch_norm_fwd_pd.get()->dst_primitive_desc();
    auto diff_dst_md = diff_dst_pd.desc();

    // create primitive descriptor for batch norm backward
    unsigned flags = mkldnn::use_scale_shift;
    auto batch_norm_bwd_desc = bn_bwd_types::op_desc{
        mkldnn::prop_kind::backward, diff_dst_md,
        src_memory.get_primitive_desc().desc(), epsilon, flags};
    auto batch_norm_bwd_pd = bn_bwd_types::op_prim{
        batch_norm_bwd_desc, mkldnn_engine, *batch_norm_fwd_pd};

    // reorder user_diff_dst if it's not in preferred format
    auto diff_dst_memory = user_diff_dst_memory;
    primitive reorder_diff_dst;
    bool is_diff_dst_reordered = false;
    if (diff_dst_pd != user_diff_dst_memory.get_primitive_desc()) {
      diff_dst_memory = memory(diff_dst_pd);
      reorder_diff_dst = reorder(user_diff_dst_memory, diff_dst_memory);
      is_diff_dst_reordered = true;
    }

    // create mkldnn memory for input tensors (src/mean/variance)
    auto mean_memory = memory(batch_norm_bwd_pd.mean_primitive_desc(),
                              to_void_cast(batch_mean_data));
    auto variance_memory = memory(batch_norm_bwd_pd.variance_primitive_desc(),
                                  to_void_cast(batch_variance_data));

    // MKLDNN requires a single piece of memory for scale and shift/bias data
    const size_t scaleshift_size = 2 * ic;

    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(scaleshift_size);
    copy_to_weights(scale_data, scale_data + ic, shift_data, shift_data + ic,
                    &scaleshift_data);

    // create mkldnn memory for input tensors (scale/shift)
    auto scaleshift_memory = memory(batch_norm_bwd_pd.weights_primitive_desc(),
                                    scaleshift_data.data());

    // create mkldnn memory for output diff weights (combined scale/shift)
    std::vector<T> diff_scaleshift_data;
    diff_scaleshift_data.reserve(scaleshift_size);
    auto diff_scaleshift_memory =
        memory(batch_norm_bwd_pd.diff_weights_primitive_desc(),
               diff_scaleshift_data.data());

    // here assume diff_src is in the same format of src
    auto diff_src_memory = memory(src_memory.get_primitive_desc(), diff_x_data);

    // finally create batch_norm backward primitive
    auto batch_norm_bwd_prim =
        batch_norm_bwd(batch_norm_bwd_pd, src_memory, mean_memory,
                       variance_memory, diff_dst_memory, scaleshift_memory,
                       diff_src_memory, diff_scaleshift_memory);

    // execute optional reorder and batch_norm backward primitive
    std::vector<primitive> pipeline;
    if (is_diff_dst_reordered) pipeline.push_back(reorder_diff_dst);
    pipeline.push_back(batch_norm_bwd_prim);
    stream(stream::kind::eager).submit(pipeline).wait();

    // copy back diff sacle/shift to output tensors (diff scale/shift)
    diff_scaleshift_data.resize(scaleshift_size);
    auto it = std::begin(diff_scaleshift_data);
    std::copy(it, std::next(it, ic), diff_scale_data);
    std::copy(std::next(it, ic), std::end(diff_scaleshift_data),
              diff_shift_data);

    // set layout/format of output tensors
    diff_x->set_layout(DataLayout::kMKLDNN);
    diff_x->set_format((memory::format)diff_src_memory.get_primitive_desc()
                           .desc()
                           .data.format);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(batch_norm, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(batch_norm_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNGradOpKernel<float>);
