// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
class SplitPrimOp : public framework::OperatorBase {
 public:
  SplitPrimOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator split_p should not be excuted directly"));
  }
};

class SplitPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of split_p op.");
    AddOutput("YS", "(Tensor), The output tensors of split_p op.")
        .AsDuplicable();
    AddAttr<int64_t>("axis", "(int64_t), The axis along which to split.");
    AddAttr<std::vector<int64_t>>(
        "num_or_sections",
        "(std::vector<int64_t>) If num_or_sections has only one element, then "
        "num_or_sections indicates the number of equal sized sub-Tensors that "
        "the input will be divided into. If num_or_sections has more then one "
        "element, the length of it indicates the number of sub-Tensors and the "
        "elements in it indicate the sizes of sub-Tensors’ dimension orderly. "
        "The length of the vector must not be larger than the input's size of "
        "specified axis.");
    AddComment(R"DOC(
Autograd primitive split_p operator.
)DOC");
  }
};

class SplitPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    auto y_var_ptrs = ctx->GetOutputVarPtrs("YS");
    framework::VarDesc *x_var = BOOST_GET(framework::VarDesc *, x_var_ptr);
    auto x_shape = x_var->GetShape();
    auto axis = ctx->Attrs().Get<int64_t>("axis");
    auto num_or_sections =
        ctx->Attrs().Get<std::vector<int64_t>>("num_or_sections");
    std::vector<int64_t> y_shape(x_shape);
    if (num_or_sections.size() == 1) {
      PADDLE_ENFORCE_EQ(x_shape[axis] % num_or_sections[0], 0,
                        platform::errors::InvalidArgument(
                            "The input tensor can't be devided equally into %d "
                            "parts equally along axis %d",
                            num_or_sections[0], axis));
      y_shape[axis] = x_shape[axis] / num_or_sections[0];
      for (size_t i = 0; i < size_t(num_or_sections[0]); ++i) {
        BOOST_GET(framework::VarDesc *, y_var_ptrs[i])->SetShape(y_shape);
      }
    } else {
      int64_t cnt_along_axis = 0;
      for (size_t i = 0; i < num_or_sections.size(); ++i) {
        y_shape[axis] = num_or_sections[i];
        cnt_along_axis += num_or_sections[i];
        BOOST_GET(framework::VarDesc *, y_var_ptrs[i])->SetShape(y_shape);
      }
      PADDLE_ENFORCE_EQ(
          x_shape[axis], cnt_along_axis,
          platform::errors::InvalidArgument(
              "The input tensor has %d elements along axis %d, thus can't be "
              "devided into %d tensor with %d elements totally.",
              x_shape[axis], axis, num_or_sections.size(), cnt_along_axis));
    }
  }
};

class SplitPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = Input(ctx, "X")[0];
    auto y_names = Output(ctx, "YS");
    for (auto y_name : y_names) {
      SetType(ctx, y_name, GetType(ctx, x_name));
      SetDataType(ctx, y_name, GetDataType(ctx, x_name));
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(split_p, paddle::operators::SplitPrimOp,
                  paddle::operators::SplitPrimOpMaker,
                  paddle::operators::SplitPrimOpShapeInference,
                  paddle::operators::SplitPrimOpVarTypeInference);
