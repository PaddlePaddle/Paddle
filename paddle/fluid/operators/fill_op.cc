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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

struct FillOpVisitor {
  FillOpVisitor(framework::LoDTensor *tensor, const std::vector<float> &value)
      : tensor_(tensor), value_(value) {}

  template <typename T>
  void apply() const {
    platform::CPUPlace cpu;
    auto *data = tensor_->mutable_data<T>(cpu);
    std::transform(value_.data(), value_.data() + tensor_->numel(), data,
                   [](float dat) { return static_cast<T>(dat); });
  }

  framework::LoDTensor *tensor_;
  const std::vector<float> &value_;
};

class FillOp : public framework::OperatorBase {
 public:
  FillOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &out =
        detail::Ref(detail::Ref(scope.FindVar(Output("Out")),
                                "Cannot find variable %s", Output("Out"))
                        .GetMutable<framework::LoDTensor>());
    out.Resize(framework::make_ddim(Attr<std::vector<int>>("shape")));
    auto dtype =
        static_cast<framework::proto::VarType::Type>(Attr<int>("dtype"));
    platform::CPUPlace cpu;
    auto force_cpu = Attr<bool>("force_cpu");
    out.mutable_data(force_cpu ? cpu : place, dtype);

    framework::LoDTensor tensor;

    if (force_cpu || platform::is_cpu_place(place)) {
      tensor.ShareDataWith(out);
    } else {
      // Always make tensor in CPU memory.
      tensor.Resize(out.dims());
      tensor.mutable_data(cpu, dtype);
    }

    framework::VisitDataType(
        dtype, FillOpVisitor(&tensor, Attr<std::vector<float>>("value")));

    if (!force_cpu && platform::is_gpu_place(place)) {
      // Copy tensor to out
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(place);
      framework::TensorCopy(tensor, place, dev_ctx, &out);
    }
  }
};

class FillOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Fill operator

Fill an tensor with `value` and `shape`. The type of the tensor is specify by
`dtype`.
)DOC");
    AddOutput("Out", "(LoDTensor) The output tensor.");
    AddAttr<std::vector<float>>(
        "value", "The float values of tensor, which are flatten in row major");
    AddAttr<std::vector<int>>("shape", "The shape of output tensor");
    AddAttr<int>("dtype", "The data type of output tensor, Default is float")
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<bool>("force_cpu",
                  "Whether the output tensor must be at CPU memory or not. "
                  "Default is false.")
        .SetDefault(false);
  }
};

class FillOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    context->SetOutputDim(
        "Out",
        framework::make_ddim(context->Attrs().Get<std::vector<int>>("shape")));
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OPERATOR(fill, ops::FillOp, ops::FillOpInferShape, ops::FillOpMaker);
