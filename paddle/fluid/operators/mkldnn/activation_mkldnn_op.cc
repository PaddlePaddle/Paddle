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

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::stream;
using platform::GetMKLDNNFormat;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;

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
}  // namespace

template <typename Functor>
class MKLDNNActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    PADDLE_ENFORCE(x->layout() == DataLayout::kMKLDNN &&
                       x->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input x tensor");

    Functor functor;
    functor(ctx);
  }
};

template <typename Functor>
class MKLDNNActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *diff_y = ctx.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE(diff_y->layout() == DataLayout::kMKLDNN &&
                       diff_y->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input OutGrad tensor");

    PADDLE_ENFORCE(
        !ctx.Attr<bool>("is_test"),
        "is_test attribute should be set to False in training phase.");

    Functor functor;
    functor(ctx);
  }
};

template <typename T>
void eltwise_forward(const framework::ExecutionContext &ctx,
                     mkldnn::algorithm algorithm, const T alpha = 0,
                     const T beta = 0) {
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                 "It must use CPUPlace.");
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  const auto &mkldnn_engine = dev_ctx.GetEngine();

  const auto *x = ctx.Input<Tensor>("X");
  auto *y = ctx.Output<Tensor>("Out");

  const T *x_data = x->data<T>();
  T *y_data = y->mutable_data<T>(ctx.GetPlace());

  PADDLE_ENFORCE(
      x->dims().size() == 2 || x->dims().size() == 3 || x->dims().size() == 4,
      "Input dim must be with 2, 3 or 4");

  std::vector<int> src_tz = framework::vectorize2int(x->dims());

  auto src_format =
      src_tz.size() == 2 ? mkldnn::memory::format::nc : x->format();

  bool is_test = ctx.Attr<bool>("is_test");

  // TODO(jczaja): When adding leaky-relu , swish , elu make sure to extend key
  // with alpha, beta
  std::string key = platform::MKLDNNHandler::GetHash(
      src_tz, std::to_string(algorithm) + ctx.op().Output("Out"));

  // TODO(jczaja): Make it Thread safe
  // save input data and layout to be referred in backward path
  const std::string key_src_data = key + "@eltwise_fwd_src_data";
  const std::string key_src_layout = key + "@eltwise_fwd_src_layout";
  // Just in case some int8 models are run interchangebly
  // with float models then format maybe diffrent
  key += std::to_string(src_format);
  const std::string key_src_mem = key + "@eltwise_fwd_src_mem";
  auto p_src_data = std::make_shared<const T *>(x_data);
  auto p_src_layout = std::make_shared<memory::format>(src_format);
  if (!is_test) {
    dev_ctx.SetBlob(key_src_data, p_src_data);
    dev_ctx.SetBlob(key_src_layout, p_src_layout);
  }

  platform::ActivationMKLDNNHandler handler(dev_ctx, mkldnn_engine, key);

  auto md = platform::MKLDNNMemDesc(src_tz, platform::MKLDNNGetDataType<T>(),
                                    src_format);

  auto activation_pd = handler.AcquireActivationPrimitiveDescriptor(
      is_test ? mkldnn::prop_kind::forward_inference
              : mkldnn::prop_kind::forward_training,
      algorithm, md, alpha, beta);

  auto src_memory_p = handler.AcquireSrcMemory(md, to_void_cast<T>(x_data));
  // jczaja: Workaround, src_memory_p is needed in BWD so it has
  // to be accessible under key not dependant on TID
  if (!is_test) {
    dev_ctx.SetBlob(key_src_mem, src_memory_p);
  }

  auto dst_memory_p =
      handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(y_data));
  auto activation_p = handler.AcquireActivation(dst_memory_p, src_memory_p);

  // push primitive to stream and wait until it's executed
  std::vector<primitive> pipeline;
  pipeline.push_back(*activation_p);
  stream(stream::kind::eager).submit(pipeline).wait();

  y->set_layout(DataLayout::kMKLDNN);
  y->set_format(GetMKLDNNFormat(*dst_memory_p));
}

template <typename T>
void eltwise_grad(const framework::ExecutionContext &ctx,
                  mkldnn::algorithm algorithm, const T alpha = 0,
                  const T beta = 0) {
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  const auto &mkldnn_engine = dev_ctx.GetEngine();

  const auto *diff_y = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto *diff_x = ctx.Output<Tensor>(framework::GradVarName("X"));

  const T *diff_y_data = diff_y->data<T>();
  T *diff_x_data = diff_x->mutable_data<T>(ctx.GetPlace());

  std::vector<int> diff_dst_tz = framework::vectorize2int(diff_y->dims());

  auto diff_y_format =
      diff_dst_tz.size() == 2 ? mkldnn::memory::format::nc : diff_y->format();

  auto diff_dst_md = platform::MKLDNNMemDesc(
      diff_dst_tz, platform::MKLDNNGetDataType<T>(), diff_y_format);

  std::string key = platform::MKLDNNHandler::GetHash(
      diff_dst_tz, std::to_string(algorithm) + ctx.op().Input("Out"));

  const std::string key_src_data = key + "@eltwise_fwd_src_data";
  const std::string key_src_layout = key + "@eltwise_fwd_src_layout";

  // Get Data from FWD op
  const auto p_src_layout =
      std::static_pointer_cast<memory::format>(dev_ctx.GetBlob(key_src_layout));
  const auto p_src_data =
      std::static_pointer_cast<T *>(dev_ctx.GetBlob(key_src_data));
  key += std::to_string(*p_src_layout);
  const std::string key_src_mem = key + "@eltwise_fwd_src_mem";
  auto src_memory =
      std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(key_src_mem));
  PADDLE_ENFORCE(src_memory != nullptr,
                 "Fail to find src_memory in device context");
  src_memory->set_data_handle(*p_src_data);

  platform::ActivationMKLDNNHandler handler(dev_ctx, mkldnn_engine, key);

  auto diff_dst_memory_p =
      handler.AcquireDiffDstMemory(diff_dst_md, to_void_cast<T>(diff_y_data));

  auto activation_backward_pd =
      handler.AcquireActivationBackwardPrimitiveDescriptor(
          algorithm, diff_dst_md, src_memory->get_primitive_desc().desc(),
          alpha, beta);

  auto diff_src_memory_p =
      handler.AcquireDiffSrcMemoryFromPrimitive(diff_x_data);

  auto activation_backward_p = handler.AcquireActivationBackward(
      diff_src_memory_p, diff_dst_memory_p, src_memory);

  // push primitive to stream and wait until it's executed
  std::vector<primitive> pipeline;
  pipeline.push_back(*activation_backward_p);
  stream(stream::kind::eager).submit(pipeline).wait();

  diff_x->set_layout(DataLayout::kMKLDNN);
  diff_x->set_format(GetMKLDNNFormat(*diff_src_memory_p));
}

template <typename T, mkldnn::algorithm algorithm>
struct MKLDNNActivationFunc : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    eltwise_forward<T>(ctx, algorithm);
  }
};

template <typename T, mkldnn::algorithm algorithm>
struct MKLDNNActivationGradFunc : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    eltwise_grad<T>(ctx, algorithm);
  }
};

template <typename T>
using ReluMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_relu>;

template <typename T>
using TanhMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_tanh>;

template <typename T>
using SqrtMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_sqrt>;

template <typename T>
using AbsMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_abs>;

template <typename T>
using ReluMKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_relu>;

template <typename T>
using TanhMKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_tanh>;

template <typename T>
using SqrtMKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_sqrt>;

template <typename T>
using AbsMKLDNNGradFunctor =
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
  __macro(relu, ReluMKLDNNFunctor, ReluMKLDNNGradFunctor); \
  __macro(tanh, TanhMKLDNNFunctor, TanhMKLDNNGradFunctor); \
  __macro(sqrt, SqrtMKLDNNFunctor, SqrtMKLDNNGradFunctor); \
  __macro(abs, AbsMKLDNNFunctor, AbsMKLDNNGradFunctor);

FOR_EACH_MKLDNN_KERNEL_FUNCTOR(REGISTER_ACTIVATION_MKLDNN_KERNEL);
