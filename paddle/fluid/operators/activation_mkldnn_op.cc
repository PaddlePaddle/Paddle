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

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;

namespace {
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
  const T *dst_data = dst->template mutable_data<T>(ctx.GetPlace());

  // get memory dim
  PADDLE_ENFORCE(src->dims().size() == 2 || src->dims().size() == 4,
                 "Input dim must be with 2 or 4");
  std::vector<int> src_tz = framework::vectorize2int(src->dims());

  // create memory description
  auto data_md = src_tz.size() == 2
                     ? platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                               mkldnn::memory::format::nc)
                     : platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                               mkldnn::memory::format::nchw);

  // create memory primitives
  auto src_memory = std::make_shared<mkldnn::memory>(
      mkldnn::memory({data_md, mkldnn_engine},
                     static_cast<void *>(const_cast<float *>(src_data))));
  // save source memory to device context to be referred in backward path
  dev_ctx.SetBlob("InputX@eltwise_pd", src_memory);
  auto dst_memory =
      mkldnn::memory({data_md, mkldnn_engine},
                     static_cast<void *>(const_cast<float *>(dst_data)));

  auto forward_desc = mkldnn::eltwise_forward::desc(
      mkldnn::prop_kind::forward_training, algorithm, data_md, alpha, beta);

  // save prim desc into global device context to be referred in backward path
  const std::string key = ctx.op().Output("Out");
  const std::string key_eltwise_pd = key + "@eltwise_pd";
  auto forward_pd = std::make_shared<mkldnn::eltwise_forward::primitive_desc>(
      forward_desc, mkldnn_engine);
  dev_ctx.SetBlob(key_eltwise_pd, forward_pd);

  auto eltwise = mkldnn::eltwise_forward(*forward_pd, *src_memory, dst_memory);

  // push primitive to stream and wait until it's executed
  std::vector<mkldnn::primitive> pipeline = {eltwise};
  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
}

template <typename T, typename ExecContext>
void eltwise_grad(const ExecContext &ctx, mkldnn::algorithm algorithm,
                  const T alpha = 0, const T beta = 0) {
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  const auto &mkldnn_engine = dev_ctx.GetEngine();

  // get buffers
  const auto *x = ctx.template Input<Tensor>("Out");

  auto *dout = ctx.template Input<Tensor>(framework::GradVarName("Out"));
  const auto *diff_dst = dout->template data<T>();

  auto *dx =
      ctx.template Output<framework::Tensor>(framework::GradVarName("X"));
  const T *diff_src = dx->template mutable_data<T>(ctx.GetPlace());

  // get memory dim
  std::vector<int> src_tz = framework::vectorize2int(x->dims());

  // create memory description
  auto data_md = src_tz.size() == 2
                     ? platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                               mkldnn::memory::format::nc)
                     : platform::MKLDNNMemDesc(src_tz, mkldnn::memory::f32,
                                               mkldnn::memory::format::nchw);

  // retrieve source memory from device context
  const std::shared_ptr<void> src_memory = dev_ctx.GetBlob("InputX@eltwise_pd");
  auto *p_src_memory = static_cast<mkldnn::memory *>(src_memory.get());

  // create memory primitives
  auto diff_src_memory =
      mkldnn::memory({data_md, mkldnn_engine},
                     static_cast<void *>(const_cast<float *>(diff_src)));
  auto diff_dst_memory =
      mkldnn::memory({data_md, mkldnn_engine},
                     static_cast<void *>(const_cast<float *>(diff_dst)));

  auto backward_desc =
      mkldnn::eltwise_backward::desc(algorithm, data_md, data_md, alpha, beta);

  // retrieve eltwise primitive desc from device context
  const std::string key = ctx.op().Input("Out");
  const std::string key_eltwise_pd = key + "@eltwise_pd";
  const std::shared_ptr<void> forward_pd = dev_ctx.GetBlob(key_eltwise_pd);
  PADDLE_ENFORCE(forward_pd != nullptr,
                 "Fail to find eltwise_pd in device context");
  auto *p_forward_pd =
      static_cast<mkldnn::eltwise_forward::primitive_desc *>(forward_pd.get());

  auto eltwise_bwd_prim_desc = mkldnn::eltwise_backward::primitive_desc(
      backward_desc, mkldnn_engine, *p_forward_pd);

  auto eltwise_bwd = mkldnn::eltwise_backward(
      eltwise_bwd_prim_desc, *p_src_memory, diff_dst_memory, diff_src_memory);

  // push primitive to stream and wait until it's executed
  std::vector<mkldnn::primitive> pipeline = {eltwise_bwd};
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
