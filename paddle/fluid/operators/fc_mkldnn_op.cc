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

#include "paddle/fluid/operators/fc_mkldnn_op.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;

void FCOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "X(Input) of Fully Connected should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "Out(Output) of Fully Connected should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("W"),
                 "W(Input) of Fully Connected should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  auto w_dims = ctx->GetInputDim("W");
  std::vector<int64_t> output_shape({in_dims[0], w_dims[1]});

  PADDLE_ENFORCE(in_dims.size() == 4,
                 "Fully Connected input should be 4-D tensor.");

  PADDLE_ENFORCE(w_dims.size() == 2,
                 "Fully Connected input should be 2-D tensor.");

  ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  ctx->ShareLoD("Input", "Out");
}

framework::OpKernelType FCOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library{framework::LibraryType::kMKLDNN};

  std::string data_format = ctx.Attr<std::string>("data_format");
  framework::DataLayout layout = framework::StringToDataLayout(data_format);

  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<Tensor>("Input")->type()), ctx.GetPlace(),
      layout, library);
}

void FCOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  auto in_dims = ctx->GetInputDim("Input");
  auto w_dims = ctx->GetInputDim("W");

  if (ctx->HasOutput(framework::GradVarName("Input"))) {
    ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
  }
  if (ctx->HasOutput(framework::GradVarName("W"))) {
    ctx->SetOutputDim(framework::GradVarName("W"), w_dims);
  }
}

framework::OpKernelType FCOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library{framework::LibraryType::kMKLDNN};

  std::string data_format = ctx.Attr<std::string>("data_format");
  framework::DataLayout layout = framework::StringToDataLayout(data_format);

  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<Tensor>("Input")->type()), ctx.GetPlace(),
      layout, library);
}

class FCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FCOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "Input",
        "(Tensor) The input tensor of fully connected operator. "
        "The format of input tensor is NCHW, where N is batch size, C is the "
        "number of channels, H is the height of the feature, "
        "and W is the width of the feature.");
    AddInput("W", "(Tensor), The second input tensor of fc op.");
    AddOutput("Out",
              "(Tensor) The output tensor of pooling operator. "
              "The format of output tensor is also NCHW, "
              "where N is batch size, C is the number of channels, "
              "H is the height of the feature, "
              "and W is the width of the feature.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<bool>("with_bias",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("AnyLayout");
    AddComment(R"DOC(
)DOC");
  }
};

struct MKLDNNMatrixSize final {
  explicit MKLDNNMatrixSize(const std::vector<int>& in,
                            const std::vector<int>& w)
      : mb{in[0]}, ic{in[1]}, oc{w[1]}, h{in[2]}, w{in[3]} {}

  bool is_spatial() const { return h > 1 && w > 1; }

  const int mb;
  const int ic;
  const int oc;
  const int h, w;
};

template <typename T>
class MKLDNNMD {
 public:
  explicit MKLDNNMD(const T* in, const T* w, bool bias)
      : sz_(std::unique_ptr<MKLDNNMatrixSize>(new MKLDNNMatrixSize(
            paddle::framework::vectorize2int(in->dims()),
            paddle::framework::vectorize2int(w->dims())))) {
    with_bias_ = bias;
  }

  mkldnn::memory::desc dst() const {
    return platform::MKLDNNMemDesc({sz_->mb, sz_->oc},
                                   mkldnn::memory::data_type::f32,
                                   mkldnn::memory::format::nc);
  }

  mkldnn::memory::desc src() const {
    return sz_->is_spatial()
               ? platform::MKLDNNMemDesc({sz_->mb, sz_->ic, sz_->h, sz_->w},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::nchw)
               : platform::MKLDNNMemDesc({sz_->mb, sz_->ic},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::nc);
  }

  mkldnn::memory::desc weights() const {
    return sz_->is_spatial()
               ? platform::MKLDNNMemDesc({sz_->oc, sz_->ic, sz_->h, sz_->w},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::oihw)
               : platform::MKLDNNMemDesc({sz_->oc, sz_->ic},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::oi);
  }

  mkldnn::memory::desc bias() const {
    return with_bias_
               ? platform::MKLDNNMemDesc({sz_->oc},
                                         mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::format_undef)
               : platform::MKLDNNMemDesc({}, mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::format_undef);
  }

 private:
  std::unique_ptr<MKLDNNMatrixSize> sz_;
  bool with_bias_;
};

class MKLDNNMemory {
 public:
  MKLDNNMemory(MKLDNNMD<Tensor>* t, const mkldnn::engine& e)
      : md_{t}, engine_{e} {}
  virtual ~MKLDNNMemory() = default;

  template <typename Output>
  mkldnn::memory dst(const Output* out) {
    return mkldnn::memory({md_->dst(), engine_},
                          static_cast<void*>(const_cast<float*>(out)));
  }

  template <typename Output>
  mkldnn::memory dst(Output* out) {
    return mkldnn::memory({md_->dst(), engine_}, out);
  }

  template <typename Input>
  mkldnn::memory src(const Input* in) {
    return mkldnn::memory({md_->src(), engine_},
                          static_cast<void*>(const_cast<float*>(in)));
  }

  template <typename Weight>
  mkldnn::memory weights(const Weight* w) {
    return mkldnn::memory({md_->weights(), engine_},
                          static_cast<void*>(const_cast<float*>(w)));
  }

  mkldnn::memory bias() {
    return mkldnn::memory(mkldnn::memory::primitive_desc(md_->bias(), engine_));
  }

 private:
  MKLDNNMD<Tensor>* md_;
  const mkldnn::engine& engine_;
};

template <typename T>
class FCMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto input = ctx.Input<Tensor>("Input");
    auto w = ctx.Input<Tensor>("W");

    PADDLE_ENFORCE(input->dims().size() == 4,
                   "Input must be with 4 dimensions, i.e. NCHW");
    PADDLE_ENFORCE(w->dims().size() == 2,
                   "Weights must be with 2 dimensions, i.e. NC");

    bool with_bias = ctx.Attr<bool>("with_bias");
    MKLDNNMD<Tensor> md(input, w, with_bias);

    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> pd =
        FcFwdPrimitiveDesc(md.src(), md.weights(), md.dst(), md.bias(),
                           with_bias, mkldnn_engine);

    const std::string key = ctx.op().Output("Out");
    const std::string key_fc_pd = key + "@fc_pd";

    dev_ctx.SetBlob(key_fc_pd, pd);

    MKLDNNMemory mem(&md, mkldnn_engine);

    const T* input_data = input->data<T>();
    const T* w_data = w->data<T>();

    auto output = ctx.Output<Tensor>("Out");
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    auto dst_memory = mem.dst(output_data);
    auto src_memory = mem.src(input_data);
    auto weights_memory = mem.weights(w_data);
    auto bias_memory = mem.bias();

    auto forward = with_bias ? mkldnn::inner_product_forward(
                                   *pd, src_memory, weights_memory, bias_memory,
                                   dst_memory)
                             : mkldnn::inner_product_forward(
                                   *pd, src_memory, weights_memory, dst_memory);

    std::vector<mkldnn::primitive> pipeline = {forward};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
  }

 private:
  std::unique_ptr<mkldnn::inner_product_forward::primitive_desc>
  FcFwdPrimitiveDesc(const mkldnn::memory::desc& src,
                     const mkldnn::memory::desc& weights,
                     const mkldnn::memory::desc& dst,
                     const mkldnn::memory::desc& bias, const bool with_bias,
                     const mkldnn::engine& engine) const {
    auto desc = with_bias
                    ? mkldnn::inner_product_forward::desc(
                          mkldnn::prop_kind::forward, src, weights, bias, dst)
                    : mkldnn::inner_product_forward::desc(
                          mkldnn::prop_kind::forward, src, weights, dst);

    auto pd = new mkldnn::inner_product_forward::primitive_desc(desc, engine);
    return std::unique_ptr<mkldnn::inner_product_forward::primitive_desc>(pd);
  }
};

template <typename T>
class FCMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    T* input_grad_data = nullptr;
    T* w_grad_data = nullptr;

    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* w_grad = ctx.Output<Tensor>(framework::GradVarName("W"));

    if (input_grad) {
      input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (w_grad) {
      w_grad_data = w_grad->mutable_data<T>(ctx.GetPlace());
    }

    const Tensor* input = ctx.Input<Tensor>("Input");
    const T* input_data = input->data<T>();

    const Tensor* w = ctx.Input<Tensor>("W");
    const T* w_data = w->data<T>();

    const Tensor* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    const T* out_grad_data = out_grad->data<T>();

    bool with_bias = ctx.Attr<bool>("with_bias");

    MKLDNNMD<Tensor> md(input, w, with_bias);
    MKLDNNMemory mem(&md, mkldnn_engine);

    auto dst_memory = mem.dst(out_grad_data);
    auto src_memory = mem.src(input_data);
    auto weights_memory = mem.weights(w_data);
    auto bias_memory = mem.bias();

    const std::string key = ctx.op().Input("Out");
    const std::string key_fc_pd = key + "@fc_pd";

    auto pd =
        std::static_pointer_cast<mkldnn::inner_product_forward::primitive_desc>(
            dev_ctx.GetBlob(key_fc_pd));

    PADDLE_ENFORCE(pd != nullptr, "Fail to find key_fc_pd in device context");

    if (w_grad) {
      auto weights_grad_memory = mem.weights(w_grad_data);

      mkldnn::inner_product_backward_weights::primitive_desc bwd_weight_pd =
          FcBwdWeightsPrimitiveDesc(md.src(), md.weights(), md.dst(), md.bias(),
                                    with_bias, *pd, mkldnn_engine);

      auto bwd_weights_prim = mkldnn::inner_product_backward_weights(
          bwd_weight_pd, src_memory, dst_memory, weights_grad_memory,
          bias_memory);

      std::vector<mkldnn::primitive> pipeline{bwd_weights_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }

    if (input_grad) {
      auto src_grad_memory = mem.src(input_grad_data);

      mkldnn::inner_product_backward_data::primitive_desc bwd_data_pd =
          FcBwdDataPrimitiveDesc(md.src(), md.weights(), md.dst(), *pd,
                                 mkldnn_engine);

      auto bwd_data_prim = mkldnn::inner_product_backward_data(
          bwd_data_pd, dst_memory, weights_memory, src_grad_memory);

      std::vector<mkldnn::primitive> pipeline{bwd_data_prim};
      mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();
    }
  }

 private:
  mkldnn::inner_product_backward_weights::primitive_desc
  FcBwdWeightsPrimitiveDesc(
      const mkldnn::memory::desc& src, const mkldnn::memory::desc& diff_weights,
      const mkldnn::memory::desc& diff_dst, const mkldnn::memory::desc& bias,
      const bool with_bias,
      const mkldnn::inner_product_forward::primitive_desc& pd,
      const mkldnn::engine& engine) const {
    auto bwd_weight_desc = with_bias
                               ? mkldnn::inner_product_backward_weights::desc(
                                     src, diff_weights, bias, diff_dst)
                               : mkldnn::inner_product_backward_weights::desc(
                                     src, diff_weights, bias, diff_dst);

    return mkldnn::inner_product_backward_weights::primitive_desc(
        bwd_weight_desc, engine, pd);
  }

  mkldnn::inner_product_backward_data::primitive_desc FcBwdDataPrimitiveDesc(
      const mkldnn::memory::desc& diff_src, const mkldnn::memory::desc& weights,
      const mkldnn::memory::desc& diff_dst,
      const mkldnn::inner_product_forward::primitive_desc& pd,
      const mkldnn::engine& engine) const {
    auto bwd_data_desc =
        mkldnn::inner_product_backward_data::desc(diff_src, weights, diff_dst);
    return mkldnn::inner_product_backward_data::primitive_desc(bwd_data_desc,
                                                               engine, pd);
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP(fc, paddle::operators::FCOp, paddle::operators::FCOpMaker, fc_grad,
            paddle::operators::FCOpGrad);

REGISTER_OP_KERNEL(fc, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::FCMKLDNNOpKernel<float>);

REGISTER_OP_KERNEL(fc_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::FCMKLDNNGradOpKernel<float>);
