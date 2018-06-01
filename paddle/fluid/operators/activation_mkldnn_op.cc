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

#include "mkldnn.hpp"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/mkldnn_activation_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;

namespace {
std::string gethash(const mkldnn::memory::dims &operand_dims,
                    const mkldnn::algorithm algorithm) {
  auto dim2str = [](const mkldnn::memory::dims &operand_dims) {
    std::string dstr = "";
    for (size_t i = 0; i < operand_dims.size(); ++i) {
      dstr += std::to_string(operand_dims[i]) + "-";
    }
    return dstr;
  };
  return dim2str(operand_dims) + std::to_string(algorithm);
}

template <typename T, typename ExecContext>
void eltwise_forward(const ExecContext &ctx, mkldnn::algorithm algorithm,
                     const T alpha = 0, const T beta = 0) {
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                 "It must use CPUPlace.");

  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  const auto &mkldnn_engine = dev_ctx.GetEngine();

  // get buffers
  const auto *src = ctx.template Input<Tensor>("X");
  const auto *src_data = src->template data<T>();

  auto *dst = ctx.template Output<Tensor>("Out");
  T *dst_data = dst->template mutable_data<T>(ctx.GetPlace());

  // get memory dim
  PADDLE_ENFORCE(src->dims().size() == 2 || src->dims().size() == 4,
                 "Input dim must be with 2 or 4");
  std::vector<int> src_tz = framework::vectorize2int(src->dims());

  const std::string key = gethash(src_tz, algorithm);
  const std::string key_src_data =
      key + ctx.op().Output("Out") + "@eltwise_fwd_src_data";
  const std::string key_src_mem = key + "@eltwise_fwd_src_mem";
  const std::string key_dst_mem = key + "@eltwise_fwd_dst_mem";
  const std::string key_fwd = key + "@eltwise_fwd";

  auto p_fwd = std::static_pointer_cast<mkldnn::eltwise_forward>(
      dev_ctx.GetBlob(key_fwd));

  // save input data to be referred in backward path
  auto p_src_data = std::make_shared<const T *>(src_data);
  dev_ctx.SetBlob(key_src_data, p_src_data);

  if (p_fwd == nullptr) {
    // create memory description
    auto data_md = src_tz.size() == 2
                       ? platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                                 mkldnn::memory::format::nc)
                       : platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                                 mkldnn::memory::format::nchw);

    // create memory primitives
    auto p_src_mem = std::make_shared<mkldnn::memory>(mkldnn::memory(
        {data_md, mkldnn_engine}, platform::to_void_cast(src_data)));
    dev_ctx.SetBlob(key_src_mem, p_src_mem);

    auto p_dst_mem = std::make_shared<mkldnn::memory>(mkldnn::memory(
        {data_md, mkldnn_engine}, platform::to_void_cast(dst_data)));
    dev_ctx.SetBlob(key_dst_mem, p_dst_mem);

    auto fwd_desc = mkldnn::eltwise_forward::desc(
        mkldnn::prop_kind::forward_training, algorithm, data_md, alpha, beta);
    auto p_fwd_pd = std::make_shared<mkldnn::eltwise_forward::primitive_desc>(
        fwd_desc, mkldnn_engine);
    const std::string key_fwd_pd = key + "eltwise_fwd_pd";
    dev_ctx.SetBlob(key_fwd_pd, p_fwd_pd);
    p_fwd = std::make_shared<mkldnn::eltwise_forward>(
        *p_fwd_pd, *(p_src_mem.get()), *(p_dst_mem.get()));
    dev_ctx.SetBlob(key_fwd, p_fwd);
  } else {
    // primitives already exist
    auto p_src_mem =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(key_src_mem));
    PADDLE_ENFORCE(p_src_mem != nullptr,
                   "Fail to find eltwise p_src_mem in device context.");
    auto p_dst_mem =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(key_dst_mem));
    PADDLE_ENFORCE(p_dst_mem != nullptr,
                   "Fail to find eltwise p_src_mem in device context.");

    p_src_mem->set_data_handle(platform::to_void_reinterpret_cast(src_data));
    p_dst_mem->set_data_handle(dst_data);
  }

  // push primitive to stream and wait until it's executed
  std::vector<mkldnn::primitive> pipeline = {*(p_fwd.get())};
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}

template <typename T, typename ExecContext>
void eltwise_grad(const ExecContext &ctx, mkldnn::algorithm algorithm,
                  const T alpha = 0, const T beta = 0) {
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  const auto &mkldnn_engine = dev_ctx.GetEngine();

  // get buffers
  const auto *out = ctx.template Input<Tensor>("Out");

  auto *dout = ctx.template Input<Tensor>(framework::GradVarName("Out"));
  const auto *diff_dst = dout->template data<T>();

  auto *dx =
      ctx.template Output<framework::Tensor>(framework::GradVarName("X"));
  const T *diff_src = dx->template mutable_data<T>(ctx.GetPlace());

  // get memory dim
  std::vector<int> src_tz = framework::vectorize2int(out->dims());

  const std::string key = gethash(src_tz, algorithm);
  const std::string key_diff_src_mem = key + "@eltwise_diff_src_mem";
  const std::string key_diff_dst_mem = key + "@eltwise_diff_dst_mem";
  const std::string key_grad = key + "@eltwise_grad";

  const std::string key_src_data =
      key + ctx.op().Input("Out") + "@eltwise_fwd_src_data";
  const auto p_src_data =
      std::static_pointer_cast<T *>(dev_ctx.GetBlob(key_src_data));

  const std::string key_src_mem = key + "@eltwise_fwd_src_mem";
  auto p_src_mem =
      std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(key_src_mem));
  p_src_mem->set_data_handle(*p_src_data.get());

  auto p_grad = std::static_pointer_cast<mkldnn::eltwise_forward::primitive>(
      dev_ctx.GetBlob(key_grad));

  if (p_grad == nullptr) {
    // create memory description
    auto data_md = src_tz.size() == 2
                       ? platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                                 mkldnn::memory::format::nc)
                       : platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                                 mkldnn::memory::format::nchw);

    // create memory primitives
    std::shared_ptr<void> p_diff_src_mem =
        std::make_shared<mkldnn::memory>(mkldnn::memory(
            {data_md, mkldnn_engine}, platform::to_void_cast(diff_src)));
    dev_ctx.SetBlob(key_diff_src_mem, p_diff_src_mem);
    std::shared_ptr<void> p_diff_dst_mem =
        std::make_shared<mkldnn::memory>(mkldnn::memory(
            {data_md, mkldnn_engine}, platform::to_void_cast(diff_dst)));
    dev_ctx.SetBlob(key_diff_dst_mem, p_diff_dst_mem);

    auto bwd_desc = mkldnn::eltwise_backward::desc(algorithm, data_md, data_md,
                                                   alpha, beta);

    const std::string key_fwd_pd = key + "eltwise_fwd_pd";
    auto *p_fwd_pd = static_cast<mkldnn::eltwise_forward::primitive_desc *>(
        dev_ctx.GetBlob(key_fwd_pd).get());

    auto eltwise_bwd_prim_desc = mkldnn::eltwise_backward::primitive_desc(
        bwd_desc, mkldnn_engine, *p_fwd_pd);

    p_grad = std::make_shared<mkldnn::eltwise_backward>(
        eltwise_bwd_prim_desc, *static_cast<mkldnn::memory *>(p_src_mem.get()),
        *(static_cast<mkldnn::memory *>(p_diff_dst_mem.get())),
        *(static_cast<mkldnn::memory *>(p_diff_src_mem.get())));
  } else {
    // primitives already exist
    auto p_diff_src_mem = std::static_pointer_cast<mkldnn::memory>(
        dev_ctx.GetBlob(key_diff_src_mem));
    auto p_diff_dst_mem = std::static_pointer_cast<mkldnn::memory>(
        dev_ctx.GetBlob(key_diff_dst_mem));

    p_diff_src_mem->set_data_handle(
        platform::to_void_reinterpret_cast(diff_src));
    p_diff_dst_mem->set_data_handle(
        platform::to_void_reinterpret_cast(diff_dst));
  }

  // push primitive to stream and wait until it's executed
  std::vector<mkldnn::primitive> pipeline = {*(p_grad.get())};
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}
}  // anonymous namespace

template <typename T, mkldnn::algorithm algorithm>
struct MKLDNNActivationFunc : public BaseActivationFunctor<T> {
  template <typename ExecContext>
  void operator()(const ExecContext &ctx) const {
    eltwise_forward<T>(ctx, algorithm);
  }
};

template <typename T, mkldnn::algorithm algorithm>
struct MKLDNNActivationGradFunc : public BaseActivationFunctor<T> {
  template <typename ExecContext>
  void operator()(const ExecContext &ctx) const {
    eltwise_grad<T>(ctx, algorithm);
  }
};

template <typename T>
using ReluMkldnnFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_relu>;

template <typename T>
using TanhMkldnnFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_tanh>;

template <typename T>
using SqrtMkldnnFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_sqrt>;

template <typename T>
using AbsMkldnnFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_abs>;

template <typename T>
using ReluMkldnnGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_relu>;

template <typename T>
using TanhMkldnnGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_tanh>;

template <typename T>
using SqrtMkldnnGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_sqrt>;

template <typename T>
using AbsMkldnnGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_abs>;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_ACTIVATION_MKLDNN_KERNEL(act_type, functor, grad_functor) \
  REGISTER_OP_KERNEL(act_type, MKLDNN, ::paddle::platform::CPUPlace,       \
                     ops::MKLDNNActivationKernel<ops::functor<float>>);    \
  REGISTER_OP_KERNEL(                                                      \
      act_type##_grad, MKLDNN, ::paddle::platform::CPUPlace,               \
      ops::MKLDNNActivationGradKernel<ops::grad_functor<float>>);

#define FOR_EACH_MKLDNN_KERNEL_FUNCTOR(__macro)            \
  __macro(relu, ReluMkldnnFunctor, ReluMkldnnGradFunctor); \
  __macro(tanh, TanhMkldnnFunctor, TanhMkldnnGradFunctor); \
  __macro(sqrt, SqrtMkldnnFunctor, SqrtMkldnnGradFunctor); \
  __macro(abs, AbsMkldnnFunctor, AbsMkldnnGradFunctor);

FOR_EACH_MKLDNN_KERNEL_FUNCTOR(REGISTER_ACTIVATION_MKLDNN_KERNEL);
