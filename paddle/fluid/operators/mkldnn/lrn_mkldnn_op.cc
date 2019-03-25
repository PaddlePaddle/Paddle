/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/lrn_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;

namespace {
template <typename T, typename... Args>
std::shared_ptr<T> insert_to_context(const std::string& key,
                                     const MKLDNNDeviceContext& dev_ctx,
                                     Args&&... args) {
  auto p = std::static_pointer_cast<T, void>(dev_ctx.GetBlob(key));

  if (!p) {
    p = std::make_shared<T>(args...);
    dev_ctx.SetBlob(key, std::static_pointer_cast<void, T>(p));
  }

  return p;
}

template <typename... Args>
void run_primitive(Args&&... args) {
  auto forward_op = mkldnn::lrn_forward{args...};

  std::vector<mkldnn::primitive> pipeline = {forward_op};
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}
}  // namespace

template <typename T>
class LRNMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const bool is_float_type = std::is_same<T, float>::value;
    PADDLE_ENFORCE(is_float_type, "MKLDNN LRN must use float data.");
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "MKLDNN LRN must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto x = ctx.Input<Tensor>("X");
    auto out = ctx.Output<Tensor>("Out");
    auto mid = ctx.Output<Tensor>("MidOut");

    auto input_data = x->data<T>();
    auto output_data = out->mutable_data<T>(ctx.GetPlace());
    mid->mutable_data<T>(ctx.GetPlace());

    const int n = ctx.Attr<int>("n");
    // MKL-DNN implements LRN in a caffe way:
    // http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
    // Where sum of squares is divided by size of normalization window
    // this is not the case for PaddlePaddle LRN.
    // Hence we need to compensate for this diffrence by
    // multipliing alpha by size of window(n)
    const float alpha = ctx.Attr<float>("alpha") * static_cast<float>(n);
    const float beta = ctx.Attr<float>("beta");
    const float k = ctx.Attr<float>("k");
    const bool is_test = ctx.Attr<bool>("is_test");

    auto e_mid = framework::EigenTensor<T, 4>::From(*mid);
    e_mid = e_mid.constant(k);

    auto src_md = x->get_mkldnn_prim_desc().desc();

    auto forward_desc = mkldnn::lrn_forward::desc{mkldnn::prop_kind::forward,
                                                  mkldnn::lrn_across_channels,
                                                  src_md,
                                                  n,
                                                  alpha,
                                                  beta,
                                                  k};

    auto src_memory_pd = x->get_mkldnn_prim_desc();

    if (!is_test) {
      const std::string key = ctx.op().Output("Out");
      const std::string key_src_memory = key + "@lrn_src_memory";
      const std::string key_pd = key + "@lrn_pd";
      const std::string key_workspace_memory = key + "@lrn_workspace_memory";

      auto forward_pd = insert_to_context<mkldnn::lrn_forward::primitive_desc>(
          key_pd, dev_ctx, forward_desc, mkldnn_engine);

      auto src_memory = insert_to_context<mkldnn::memory>(
          key_src_memory, dev_ctx, src_memory_pd);

      src_memory->set_data_handle(
          static_cast<void*>(const_cast<T*>(input_data)));

      auto dst_memory_pd = forward_pd->dst_primitive_desc();
      auto dst_memory =
          mkldnn::memory(dst_memory_pd, static_cast<void*>(output_data));
      auto workspace_memory = insert_to_context<mkldnn::memory>(
          key_workspace_memory, dev_ctx,
          forward_pd->workspace_primitive_desc());

      run_primitive(*forward_pd, *src_memory, *workspace_memory, dst_memory);
      out->set_mkldnn_prim_desc(dst_memory_pd);
    } else {
      auto forward_pd =
          mkldnn::lrn_forward::primitive_desc{forward_desc, mkldnn_engine};
      auto src_memory = mkldnn::memory{
          src_memory_pd, static_cast<void*>(const_cast<T*>(input_data))};
      auto workspace_memory =
          mkldnn::memory{forward_pd.workspace_primitive_desc()};
      auto dst_memory_pd = forward_pd.dst_primitive_desc();
      auto dst_memory = mkldnn::memory(forward_pd.dst_primitive_desc(),
                                       static_cast<void*>(output_data));

      run_primitive(forward_pd, src_memory, workspace_memory, dst_memory);
      out->set_mkldnn_prim_desc(dst_memory_pd);
    }
  }
};

template <typename T>
class LRNMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const bool is_float_type = std::is_same<T, float>::value;
    PADDLE_ENFORCE(is_float_type, "MKLDNN LRN must use float data.");
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "MKLDNN LRN must use CPUPlace.");
    PADDLE_ENFORCE(
        !ctx.Attr<bool>("is_test"),
        "is_test attribute should be set to False in training phase.");

    auto x = ctx.Input<Tensor>("X");

    auto out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    const std::string key = ctx.op().Input("Out");
    const std::string key_src_memory = key + "@lrn_src_memory";
    const std::string key_pd = key + "@lrn_pd";
    const std::string key_workspace_memory = key + "@lrn_workspace_memory";

    const int n = ctx.Attr<int>("n");
    const float alpha = ctx.Attr<float>("alpha") * static_cast<float>(n);
    const float beta = ctx.Attr<float>("beta");
    const float k = ctx.Attr<float>("k");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto x_grad_data = x_grad->mutable_data<T>(ctx.GetPlace());
    auto out_grad_data = out_grad->data<T>();

    auto dims = paddle::framework::vectorize2int(x->dims());

    auto src_md = paddle::platform::MKLDNNMemDesc(
        dims, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);

    auto diff_src_md = paddle::platform::MKLDNNMemDesc(
        dims, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);

    auto diff_dst_md = paddle::platform::MKLDNNMemDesc(
        dims, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw);

    auto diff_dst_memory =
        mkldnn::memory{{diff_dst_md, mkldnn_engine},
                       static_cast<void*>(const_cast<float*>(out_grad_data))};

    auto diff_src_memory = mkldnn::memory{{diff_src_md, mkldnn_engine},
                                          static_cast<void*>(x_grad_data)};

    auto backward_desc = mkldnn::lrn_backward::desc{
        mkldnn::lrn_across_channels, src_md, diff_src_md, n, alpha, beta, k};

    auto forward_pd = dev_ctx.GetBlob(key_pd);

    auto backward_pd = mkldnn::lrn_backward::primitive_desc{
        backward_desc, mkldnn_engine,
        *static_cast<mkldnn::lrn_forward::primitive_desc*>(forward_pd.get())};

    std::shared_ptr<void> workspace_memory =
        dev_ctx.GetBlob(key_workspace_memory);

    auto src_memory = dev_ctx.GetBlob(key_src_memory);
    auto backward_op = mkldnn::lrn_backward{
        backward_pd, *static_cast<mkldnn::memory*>(src_memory.get()),
        diff_dst_memory, *static_cast<mkldnn::memory*>(workspace_memory.get()),
        diff_src_memory};

    std::vector<mkldnn::primitive> pipeline = {backward_op};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(lrn, MKLDNN, paddle::platform::CPUPlace,
                   ops::LRNMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(lrn_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::LRNMKLDNNGradOpKernel<float>);
