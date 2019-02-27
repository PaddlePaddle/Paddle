/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fake_quantize_op.h"
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/clip_op.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
struct Compare {
 public:
  bool operator()(const T a, const T b) { return (std::abs(a) < std::abs(b)); }
};

template <typename T>
struct FindAbsMaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const T* in,
                  const int num, T* out) {
    *out = std::abs(*(std::max_element(in + 0, in + num, Compare<T>())));
  }
};

template struct FindAbsMaxFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct ClipAndFakeQuantFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, framework::Tensor* out) {
    T s = scale.data<T>()[0];
    platform::Transform<platform::CPUDeviceContext> trans;
    trans(ctx, in.data<T>(), in.data<T>() + in.numel(),
          out->mutable_data<T>(ctx.GetPlace()), ClipFunctor<T>(-s, s));
    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(*ctx.eigen_device()) = (bin_cnt / s * out_e).round();
  }
};

template struct ClipAndFakeQuantFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct FindRangeAbsMaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& cur_scale,
                  const framework::Tensor& last_scale,
                  const framework::Tensor& iter, const int window_size,
                  framework::Tensor* scales_arr, framework::Tensor* out_scale) {
    T* scale_arr = scales_arr->mutable_data<T>(ctx.GetPlace());
    int64_t it = iter.data<int64_t>()[0];
    int idx = it % window_size;
    T removed = scale_arr[idx];
    T cur = cur_scale.data<T>()[0];
    scale_arr[idx] = cur;

    T max = last_scale.data<T>()[0];
    if (max < cur) {
      max = cur;
    } else if (fabs(removed - max) < 1e-6) {
      int size = (it > window_size) ? window_size : it;
      FindAbsMaxFunctor<platform::CPUDeviceContext, T>()(ctx, scale_arr, size,
                                                         &max);
    }
    out_scale->mutable_data<T>(ctx.GetPlace())[0] = max;
  }
};

template struct FindRangeAbsMaxFunctor<platform::CPUDeviceContext, float>;

class FakeQuantizeAbsMaxOp : public framework::OperatorWithKernel {
 public:
  FakeQuantizeAbsMaxOp(const std::string& type,
                       const framework::VariableNameMap& inputs,
                       const framework::VariableNameMap& outputs,
                       const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of FakeQuantizeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FakeQuantizeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("OutScale"),
                   "Output(Scale) of FakeQuantizeOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::LoDTensor>("X")->type(),
                                   ctx.device_context());
  }
};

class FakeQuantizeAbsMaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current scale");
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE(bit_length >= 1 && bit_length <= 16,
                         "'bit_length' should be between 1 and 16.");
        });
    AddComment(R"DOC(
FakeQuantize operator

$$scale = max(abs(X))$$
$$range = 2^{bit_length - 1} - 1$$
$$Out = round(X/scale * range)$$

)DOC");
  }
};

class FakeQuantizeRangeAbsMaxOp : public framework::OperatorWithKernel {
 public:
  FakeQuantizeRangeAbsMaxOp(const std::string& type,
                            const framework::VariableNameMap& inputs,
                            const framework::VariableNameMap& outputs,
                            const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of FakeQuantizeRangeAbsMaxOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out) of FakeQuantizeRangeAbsMaxOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("OutScale"),
        "Output(OutScale) of FakeQuantizeRangeAbsMaxOp should not be null");
    if (ctx->HasOutput("OutScales")) {
      int window_size = ctx->Attrs().Get<int>("window_size");
      ctx->SetOutputDim("OutScales", {window_size});
    }
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::LoDTensor>("X")->type(),
                                   ctx.device_context());
  }
};

class FakeQuantizeRangeAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("InScale", "Last scale.");
    AddInput("Iter", "Global step iteration.").AsDispensable();
    AddOutput("Out", "(Tensor) Output of quantized low level tensor.");
    AddOutput("OutScale", " Current scale");
    AddOutput("OutScales", "(Tensor) scale buffer.").AsDispensable();
    AddAttr<int>("window_size", "(int, default 10000) window range size.")
        .SetDefault(10000);
    AddAttr<int>("bit_length", "(int, default 8), quantization bit number.")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE(bit_length >= 1 && bit_length <= 16,
                         "'bit_length' should be between 1 and 16.");
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
FakeQuantize operator is used in static quantization.

$$scale = max(max(abs(x)), history_abs_max)$$
$$range = 2^{bit_length - 1} - 1$$
$$Out = round(X/scale * range)$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(fake_quantize_abs_max, ops::FakeQuantizeAbsMaxOp,
                  ops::FakeQuantizeAbsMaxOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(fake_quantize_abs_max,
                       ops::FakeQuantizeAbsMaxKernel<CPU, float>);

REGISTER_OPERATOR(fake_quantize_range_abs_max, ops::FakeQuantizeRangeAbsMaxOp,
                  ops::FakeQuantizeRangeAbsMaxOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(fake_quantize_range_abs_max,
                       ops::FakeQuantizeRangeAbsMaxKernel<CPU, float>);
