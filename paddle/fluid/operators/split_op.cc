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

#include "paddle/fluid/operators/split_op.h"

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
using LoDTensor = phi::DenseTensor;

using framework::Variable;

class SplitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(X) of SplitOp should not be null."));
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(),
                      1UL,
                      platform::errors::InvalidArgument(
                          "Outputs(Out) of SplitOp should not be empty."));
    int axis = static_cast<int>(ctx->Attrs().Get<int>("axis"));
    int num = static_cast<int>(ctx->Attrs().Get<int>("num"));
    std::vector<int> sections = static_cast<std::vector<int>>(
        ctx->Attrs().Get<std::vector<int>>("sections"));
    // Construct MetaTensor for InferMeta Func
    using CompatMetaTensor = framework::CompatMetaTensor;
    CompatMetaTensor x(ctx->GetInputVarPtrs("X")[0], ctx->IsRuntime());
    std::vector<CompatMetaTensor> out;
    size_t out_size = ctx->GetOutputVarPtrs("Out").size();
    out.reserve(out_size);
    for (size_t i = 0; i < out_size; i++) {
      out.emplace_back(
          CompatMetaTensor(ctx->GetOutputVarPtrs("Out")[i], ctx->IsRuntime()));
    }
    std::vector<phi::MetaTensor *> out_ptr(out_size);
    for (size_t i = 0; i < out_size; i++) {
      out_ptr[i] = &out[i];
    }
    phi::Scalar axis_final;
    phi::IntArray sections_final;
    // Construct axis_final
    if (ctx->IsRuntime() && ctx->HasInput("AxisTensor")) {
      Variable *var =
          PADDLE_GET_CONST(Variable *, ctx->GetInputVarPtrs("AxisTensor")[0]);
      axis_final = std::move(experimental::MakePhiScalarFromVar(*var));
    } else if (!ctx->IsRuntime() && ctx->HasInput("AxisTensor")) {
      axis_final = std::move(phi::Scalar(-1));
      axis_final.SetFromTensor(true);
    } else {
      axis_final = std::move(phi::Scalar(axis));
    }

    // Construct sections_final
    if (ctx->IsRuntime() && ctx->HasInputs("SectionsTensorList")) {
      int sections_tensor_list_size =
          ctx->GetInputVarPtrs("SectionsTensorList").size();
      const paddle::small_vector<framework::InferShapeVarPtr,
                                 phi::kInputSmallVectorSize>
          &sections_varptr_list = ctx->GetInputVarPtrs("SectionsTensorList");
      std::vector<LoDTensor> sections_from_tensor;
      sections_from_tensor.reserve(sections_tensor_list_size);
      for (const auto &section_varptr : sections_varptr_list) {
        Variable *var = PADDLE_GET_CONST(Variable *, section_varptr);
        sections_from_tensor.emplace_back(var->Get<LoDTensor>());
      }
      sections_final = std::move(phi::IntArray(sections_from_tensor));
    } else if (!ctx->IsRuntime() && ctx->HasInputs("SectionsTensorList")) {
      sections_final = std::move(phi::IntArray(std::vector<int>(
          ctx->GetInputVarPtrs("SectionsTensorList").size(), -1)));
      sections_final.SetFromTensor(true);
    } else {
      sections_final = std::move(phi::IntArray(sections));
    }
    if (sections.size() > 0) {
      if (ctx->IsRuntime()) {
        phi::SplitInferMeta(
            x, sections_final, axis_final, out_ptr, {true, false});
      } else {
        phi::SplitInferMeta(
            x, sections_final, axis_final, out_ptr, {false, false});
      }
    } else {
      if (ctx->IsRuntime()) {
        phi::SplitWithNumInferMeta(x, num, axis_final, out_ptr, {true, false});
      } else {
        phi::SplitWithNumInferMeta(x, num, axis_final, out_ptr, {false, false});
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      // OneDNN uses blocking format, which cannot be always supported with
      // reorders, because if blocked dimension is not divisible by 8 or
      // 16(depending on which blocking format is used) submemory cannot be
      // created, so in that scenario a fallback is needed
      const auto x_md = ctx.Input<phi::DenseTensor>("X")->mem_desc();
      if (x_md.data.format_desc.blocking.inner_nblks == 0)
        return framework::OpKernelType(input_data_type,
                                       ctx.GetPlace(),
                                       phi::DataLayout::ONEDNN,
                                       framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "AxisTensor" || var_name == "SectionsTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(
        expected_kernel_type.data_type_, tensor.place(), tensor.layout());
  }
};

class SplitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of the split operator.");
    AddInput("AxisTensor",
             "(Tensor) The axis which the input will be split on. "
             "It has higher priority than Attr(axis). "
             "The shape of AxisTensor must be [1]")
        .AsDispensable();
    AddInput("SectionsTensorList",
             "(vector<Tensor<int>>, optional). "
             "The length of each output along the specified axis. "
             "It has a higher priority than Attr(sections)."
             "The shape of the element in vector must be [1].")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out", "(Tensor) Output tensors of the split operator.")
        .AsDuplicable();
    AddComment(R"DOC(
Split operator

This operator splits the input tensor into multiple sub-tensors.

Example:
  Input = [[1,2],
           [3,4],
           [5,6]]
  sections = [2,1]
  axis = 0
  Output[0] = [[1,2],
               [3,4]]
  Output[1] = [[5,6]]

    )DOC");
    AddAttr<std::vector<int>>("sections",
                              "(vector<int>) "
                              "the length of each output along the "
                              "specified axis.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("num",
                 "(int, default 0)"
                 "Number of sub-tensors. This must evenly divide "
                 "Input.dims()[axis]")
        .SetDefault(0);
    AddAttr<int>("axis",
                 "(int, default 0) "
                 "The axis which the input will be split on.")
        .SetDefault(0);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "bfloat16", "int8", "uint8"});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(split,
                  ops::SplitOp,
                  ops::SplitOpMaker,
                  ops::SplitGradMaker<paddle::framework::OpDesc>,
                  ops::SplitGradMaker<paddle::imperative::OpBase>);
