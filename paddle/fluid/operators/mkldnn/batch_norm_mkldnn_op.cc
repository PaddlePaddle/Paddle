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

class BatchNormMKLDNNHandler : public platform::MKLDNNHandler {
 public:
  BatchNormMKLDNNHandler(const platform::MKLDNNDeviceContext &dev_ctx,
                         mkldnn::engine engine, const std::string &base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key) {}

  std::shared_ptr<memory> AcquireScaleshiftMemoryFromPrimitive(void *ptr) {
    return this->AcquireMemory(batch_norm_pd_->weights_desc(), ptr, "@scaleshift_mem_p");
  }

  std::shared_ptr<memory> AcquireMeanMemoryFromPrimitive(void *ptr) {
    return this->AcquireMemory(batch_norm_pd_->mean_desc(), ptr, "@mean_mem_p");
  }

  std::shared_ptr<memory> AcquireVarianceMemoryFromPrimitive(void *ptr) {
    return this->AcquireMemory(batch_norm_pd_->variance_desc(), ptr, "@variance_mem_p");
  }

  std::shared_ptr<batch_norm_fwd::primitive_desc>
  AcquireBatchNormPrimitiveDescriptor(const batch_norm_fwd::desc &bn_fwd_desc,
                                      const mkldnn::engine &engine) {
    // BatchNorm PD has to be passed to Grad op that
    // may be executed by diffrent thread, hence
    // for that one we use key that does not contain TID
    const std::string key_batch_norm_fwd_pd = key_common_ + "@bn_fwd_pd";
    batch_norm_pd_ = std::static_pointer_cast<batch_norm_fwd::primitive_desc>(
        dev_ctx_.GetBlob(key_batch_norm_fwd_pd));

    if (batch_norm_pd_ == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);
      batch_norm_pd_ = std::static_pointer_cast<batch_norm_fwd::primitive_desc>(
          dev_ctx_.GetBlob(key_batch_norm_fwd_pd));
      if (batch_norm_pd_ == nullptr) {
        batch_norm_pd_.reset(
            new batch_norm_fwd::primitive_desc(bn_fwd_desc, engine));
        dev_ctx_.SetBlob(key_batch_norm_fwd_pd, batch_norm_pd_);
      }
    }
    return batch_norm_pd_;
  }

  std::shared_ptr<batch_norm_fwd> AcquireTestTrainingBatchNormFwd(bool is_test) {
    auto prim_key = key_ + "@batch_norm_p";
    auto batch_norm_p =
        std::static_pointer_cast<batch_norm_fwd>(dev_ctx_.GetBlob(prim_key));

    if (batch_norm_p == nullptr) {
      if (is_test) {
        batch_norm_p = std::make_shared<batch_norm_fwd>(*batch_norm_pd_);
      } else {
        batch_norm_p = std::make_shared<batch_norm_fwd>(*batch_norm_pd_);
      }

      dev_ctx_.SetBlob(prim_key, batch_norm_p);
    }

    return batch_norm_p;
  }

  static std::string GetHash(const memory::dims &input_dims, float epsilon,
                             mkldnn::normalization_flags flag, bool is_test,
                             memory::format_tag format,
                             const std::string &suffix = "") {
    auto dims2str = [](const memory::dims &operand_dims) {
      std::string dstr = "";
      for (size_t i = 0; i < operand_dims.size(); ++i) {
        dstr += std::to_string(operand_dims[i]) + "-";
      }
      return dstr;
    };
    return dims2str(input_dims) + std::to_string(epsilon) +
           std::to_string(static_cast<int>(flag)) + std::to_string(is_test) +
           std::to_string(static_cast<int>(format)) + suffix;
  }

 private:
  std::shared_ptr<batch_norm_fwd::primitive_desc> batch_norm_pd_;
};

std::shared_ptr<memory> UpdateMemoryData(
    const platform::MKLDNNDeviceContext &dev_ctx, const std::string &key,
    void *new_ptr) {
  auto mem = std::static_pointer_cast<memory>(dev_ctx.GetBlob(key));
  PADDLE_ENFORCE(
      mem != nullptr,
      (std::string("Fail to find memory in device context [key: ") + key + "]")
          .c_str());
  mem->set_data_handle(new_ptr);
  return mem;
}

template <typename T, typename Container>
void copy_to_weights(T scale_begin, T scale_end, T shift_begin, T shift_end,
                     Container *c) {
  auto it = std::begin(*c);

  std::copy(scale_begin, scale_end, std::inserter(*c, it));
  std::copy(
      shift_begin, shift_end,
      std::inserter(*c, std::next(it, std::distance(scale_begin, scale_end))));
}

}  // namespace

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
                       x->format() != memory::format_tag::undef,
                   "Wrong layout/format set for Input x tensor");

    const T *x_data = x->data<T>();
    const T *mean_data = mean->data<T>();
    const T *variance_data = variance->data<T>();
    T *y_data = y->mutable_data<T>(ctx.GetPlace());
    T *mean_out_data = mean_out->mutable_data<T>(ctx.GetPlace());
    T *variance_out_data = variance_out->mutable_data<T>(ctx.GetPlace());
    T *batch_mean_data = nullptr;
    T *batch_variance_data = nullptr;

    if (!global_stats) {
      batch_mean_data = batch_mean->mutable_data<T>(ctx.GetPlace());
      batch_variance_data = batch_variance->mutable_data<T>(ctx.GetPlace());
    }

    auto propagation = global_stats == true
                           ? mkldnn::prop_kind::forward_scoring
                           : mkldnn::prop_kind::forward_training;

    auto src_tz = paddle::framework::vectorize(x->dims());
    auto scale_tz = paddle::framework::vectorize(scale->dims());
    PADDLE_ENFORCE(scale_tz.size() == 1, "Dims of scale tensor is NOT 1");
    const unsigned int ic = scale_tz[0];

    // MKLDNN requires a single piece of memory for scale and shift/bias data
    const size_t scaleshift_size = 2 * ic;
    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(scaleshift_size);

    copy_to_weights(scale->data<T>(), scale->data<T>() + ic, shift->data<T>(),
                    shift->data<T>() + ic, &scaleshift_data);

    auto flags = mkldnn::normalization_flags::use_scale_shift;
    if (global_stats) flags |= mkldnn::normalization_flags::use_global_stats;
    if (fuse_with_relu) flags |= mkldnn::normalization_flags::fuse_norm_relu;

    // create mkldnn memory from input x tensor
    mkldnn::memory::format_tag input_format =
        platform::MKLDNNFormatForSize(src_tz.size(), x->format());

    // keys for backward pass
    const std::string key = BatchNormMKLDNNHandler::GetHash(
        src_tz, epsilon, flags, global_stats, input_format,
        ctx.op().Output("SavedMean"));
    BatchNormMKLDNNHandler handler(dev_ctx, mkldnn_engine, key);

    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), input_format);

    // create primitive descriptor for batch norm forward
    using bn_fwd_types = bn_type_traits<mkldnn::batch_normalization_forward>;
    auto batch_norm_fwd_desc =
        bn_fwd_types::op_desc{propagation, user_src_md, epsilon, flags};

    auto batch_norm_fwd_pd = handler.AcquireBatchNormPrimitiveDescriptor(
        batch_norm_fwd_desc, mkldnn_engine);

    auto src_memory =
        handler.AcquireSrcMemory(user_src_md, to_void_cast(x_data));

    // crate mkldnn memory for weights(scale/shift)
    auto scaleshift_memory =
        handler.AcquireScaleshiftMemoryFromPrimitive(scaleshift_data.data());

    // create mkldnn memory for output y tensor
    auto dst_memory =
        handler.AcquireDstMemory(batch_norm_fwd_pd->dst_desc(), y_data);

    std::shared_ptr<memory> mean_memory;
    std::shared_ptr<memory> variance_memory;

    std::shared_ptr<batch_norm_fwd> batch_norm_p;
    if (global_stats) {
      // create mkldnn memory for stats (as input)
      mean_memory =
          handler.AcquireMeanMemoryFromPrimitive(to_void_cast(mean_data));
      variance_memory = handler.AcquireVarianceMemoryFromPrimitive(
          to_void_cast(variance_data));

      batch_norm_p = handler.AcquireTestTrainingBatchNormFwd(true);
    } else {
      // create mkldnn memory for stats (as output)
      mean_memory = handler.AcquireMeanMemoryFromPrimitive(batch_mean_data);
      variance_memory =
          handler.AcquireVarianceMemoryFromPrimitive(batch_variance_data);

      batch_norm_p = handler.AcquireTestTrainingBatchNormFwd(false);
    }

    y->set_layout(DataLayout::kMKLDNN);
    y->set_format(platform::GetMKLDNNFormat(*dst_memory));

    mkldnn::stream astream(mkldnn_engine);
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
                       diff_y->format() != memory::format_tag::undef,
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

    auto src_tz = paddle::framework::vectorize(x->dims());
    auto diff_src_tz = src_tz;
    auto dst_tz = src_tz;
    auto diff_dst_tz = dst_tz;
    auto scale_tz = paddle::framework::vectorize(scale->dims());
    PADDLE_ENFORCE(scale_tz.size() == 1, "Dims of scale tensor is NOT 1");

    const int64_t ic = scale_tz[0];

    using bn_bwd_types = bn_type_traits<mkldnn::batch_normalization_backward>;

    mkldnn::memory::format_tag dst_format =
        platform::MKLDNNFormatForSize(src_tz.size(), diff_y->format());

    mkldnn::memory::format_tag input_format =
        platform::MKLDNNFormatForSize(src_tz.size(), x->format());

    auto flags = mkldnn::normalization_flags::use_scale_shift;

    // keys from forward pass
    const std::string key = BatchNormMKLDNNHandler::GetHash(
        src_tz, epsilon, flags, false, input_format,
        ctx.op().Input("SavedMean"));
    const std::string key_batch_norm_fwd_pd = key + "@bn_fwd_pd";

    // keys for primitives reuse
    const std::string key_with_hash =
        key + BatchNormMKLDNNHandler::GetHash(src_tz, epsilon, flags, false,
                                              input_format);
    const std::string key_batch_norm_bwd_p =
        key_with_hash + "@batch_norm_bwd_p";
    const std::string key_batch_norm_src_mem_p =
        key_with_hash + "@batch_norm_bwd_src_mem_p";
    const std::string key_batch_norm_mean_mem_p =
        key_with_hash + "@batch_norm_bwd_mean_mem_p";
    const std::string key_batch_norm_variance_mem_p =
        key_with_hash + "@batch_norm_bwd_variance_mem_p";
    const std::string key_batch_norm_scaleshift_mem_p =
        key_with_hash + "@batch_norm_bwd_scaleshift_mem_p";
    const std::string key_batch_norm_diff_scaleshift_mem_p =
        key_with_hash + "@batch_norm_bwd_diff_scaleshift_mem_p";
    const std::string key_batch_norm_diff_src_mem_p =
        key_with_hash + "@batch_norm_bwd_diff_src_mem_p";
    const std::string key_batch_norm_diff_dst_mem_p =
        key_with_hash + "@batch_norm_bwd_diff_dst_mem_p";

    std::shared_ptr<mkldnn::reorder> reorder_diff_dst;
    bool is_diff_dst_reordered = false;
    auto user_diff_dst_memory =
        memory({{diff_dst_tz}, memory::data_type::f32, dst_format},
               mkldnn_engine, to_void_cast(diff_y_data));

    // MKLDNN requires a single piece of memory for scale and shift/bias data
    const size_t scaleshift_size = 2 * ic;

    std::vector<T> scaleshift_data;
    scaleshift_data.reserve(scaleshift_size);
    copy_to_weights(scale_data, scale_data + ic, shift_data, shift_data + ic,
                    &scaleshift_data);

    std::vector<T> diff_scaleshift_data;
    diff_scaleshift_data.reserve(scaleshift_size);

    auto batch_norm_fwd_pd =
        std::static_pointer_cast<batch_norm_fwd::primitive_desc>(
            dev_ctx.GetBlob(key_batch_norm_fwd_pd));
    PADDLE_ENFORCE(batch_norm_fwd_pd != nullptr,
                   "Fail to find batch_norm_fwd_pd in device context");

    auto batch_norm_bwd_p = std::static_pointer_cast<batch_norm_bwd>(
        dev_ctx.GetBlob(key_batch_norm_bwd_p));

    // create mkldnn memory for input tensors (src/mean/variance)

    // for diff_dst, try to use same format as dst in forward pass
    auto diff_dst_md = batch_norm_fwd_pd.get()->dst_desc();

    auto src_memory = std::shared_ptr<memory>(
        new memory({{src_tz}, memory::data_type::f32, input_format},
                   mkldnn_engine, to_void_cast(x_data)));

    // create primitive descriptor for batch norm backward
    auto batch_norm_bwd_desc =
        bn_bwd_types::op_desc{mkldnn::prop_kind::backward, diff_dst_md,
                              src_memory->get_desc(), epsilon, flags};
    auto batch_norm_bwd_pd = bn_bwd_types::op_prim{
        batch_norm_bwd_desc, mkldnn_engine, *batch_norm_fwd_pd};

    auto mean_memory =
        std::make_shared<memory>(batch_norm_bwd_pd.mean_desc(), mkldnn_engine,
                                 to_void_cast(batch_mean_data));
    auto variance_memory = std::make_shared<memory>(
        batch_norm_bwd_pd.variance_desc(), mkldnn_engine,
        to_void_cast(batch_variance_data));

    // create mkldnn memory for input tensors (scale/shift)
    auto scaleshift_memory =
        std::make_shared<memory>(batch_norm_bwd_pd.weights_desc(),
                                 mkldnn_engine, scaleshift_data.data());

    // create mkldnn memory for output diff weights (combined scale/shift)
    auto diff_scaleshift_memory =
        std::make_shared<memory>(batch_norm_bwd_pd.diff_weights_desc(),
                                 mkldnn_engine, diff_scaleshift_data.data());

    std::shared_ptr<memory> diff_src_memory;
    std::shared_ptr<memory> diff_dst_memory;

    if (batch_norm_bwd_p == nullptr) {
      // reorder user_diff_dst if it's not in preferred format
      diff_dst_memory = std::make_shared<memory>(user_diff_dst_memory);
      if (diff_dst_md != user_diff_dst_memory.get_desc()) {
        diff_dst_memory = std::make_shared<memory>(diff_dst_md, mkldnn_engine);
        reorder_diff_dst = std::make_shared<mkldnn::reorder>(
            user_diff_dst_memory, *diff_dst_memory);
        is_diff_dst_reordered = true;
      }

      // here assume diff_src is in the same format of src
      diff_src_memory = std::make_shared<memory>(src_memory->get_desc(),
                                                 mkldnn_engine, diff_x_data);

      // finally create batch_norm backward primitive
      batch_norm_bwd_p = std::make_shared<batch_norm_bwd>(batch_norm_bwd_pd);

      dev_ctx.SetBlob(key_batch_norm_bwd_p, batch_norm_bwd_p);
      dev_ctx.SetBlob(key_batch_norm_src_mem_p, src_memory);
      dev_ctx.SetBlob(key_batch_norm_mean_mem_p, mean_memory);
      dev_ctx.SetBlob(key_batch_norm_variance_mem_p, variance_memory);
      dev_ctx.SetBlob(key_batch_norm_scaleshift_mem_p, scaleshift_memory);
      dev_ctx.SetBlob(key_batch_norm_diff_scaleshift_mem_p,
                      diff_scaleshift_memory);
      dev_ctx.SetBlob(key_batch_norm_diff_src_mem_p, diff_src_memory);
      dev_ctx.SetBlob(key_batch_norm_diff_dst_mem_p, diff_dst_memory);

      // set layout/format of output tensors
      diff_x->set_layout(DataLayout::kMKLDNN);
      diff_x->set_format(paddle::platform::GetMKLDNNFormat(*diff_src_memory));
    } else {
      // primitives already exist
      UpdateMemoryData(dev_ctx, key_batch_norm_src_mem_p, to_void_cast(x_data));
      UpdateMemoryData(dev_ctx, key_batch_norm_mean_mem_p,
                       to_void_cast(batch_mean_data));
      UpdateMemoryData(dev_ctx, key_batch_norm_variance_mem_p,
                       to_void_cast(batch_variance_data));
      UpdateMemoryData(dev_ctx, key_batch_norm_scaleshift_mem_p,
                       scaleshift_data.data());
      UpdateMemoryData(dev_ctx, key_batch_norm_diff_scaleshift_mem_p,
                       diff_scaleshift_data.data());
      diff_src_memory = UpdateMemoryData(dev_ctx, key_batch_norm_diff_src_mem_p,
                                         to_void_cast(diff_x_data));
      diff_dst_memory = UpdateMemoryData(dev_ctx, key_batch_norm_diff_dst_mem_p,
                                         to_void_cast(diff_y_data));

      // reorder user_diff_dst if it's not in preferred format
      if (diff_dst_memory->get_desc() != user_diff_dst_memory.get_desc()) {
        reorder_diff_dst = std::make_shared<mkldnn::reorder>(
            user_diff_dst_memory, *diff_dst_memory);
        is_diff_dst_reordered = true;
      }

      // set layout/format of output tensors
      diff_x->set_layout(DataLayout::kMKLDNN);
      diff_x->set_format(paddle::platform::GetMKLDNNFormat(*diff_src_memory));
    }

    // execute optional reorder and batch_norm backward primitive
    mkldnn::stream astream(mkldnn_engine);
    if (is_diff_dst_reordered) {
      reorder_diff_dst->execute(astream, user_diff_dst_memory,
                                *diff_dst_memory);
      astream.wait();
    }
    batch_norm_bwd_p->execute(
        astream, {{MKLDNN_ARG_SRC, *src_memory},
                  {MKLDNN_ARG_MEAN, *mean_memory},
                  {MKLDNN_ARG_VARIANCE, *variance_memory},
                  {MKLDNN_ARG_DIFF_DST, *diff_dst_memory},
                  {MKLDNN_ARG_SCALE_SHIFT, *scaleshift_memory},
                  {MKLDNN_ARG_DIFF_SRC, *diff_src_memory},
                  {MKLDNN_ARG_DIFF_SCALE_SHIFT, *diff_scaleshift_memory}});
    astream.wait();

    // copy back diff sacle/shift to output tensors (diff scale/shift)
    diff_scaleshift_data.resize(scaleshift_size);
    auto it = std::begin(diff_scaleshift_data);
    std::copy(it, std::next(it, ic), diff_scale_data);
    std::copy(std::next(it, ic), std::end(diff_scaleshift_data),
              diff_shift_data);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(batch_norm, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(batch_norm_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::BatchNormMKLDNNGradOpKernel<float>);
