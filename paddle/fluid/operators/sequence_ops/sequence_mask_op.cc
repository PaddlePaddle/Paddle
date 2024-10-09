// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class SequenceMaskOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    if (var_name == "depth_tensor") {
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

class SequenceMaskOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of sequence_mask op.");
    AddOutput("Y", "The output mask of sequence_mask op.");
    AddInput("MaxLenTensor",
             "Max length tensor"
             "have higher priority than maxlen attribute")
        .AsDispensable();
    AddAttr<int>("maxlen",
                 "The maximum length of the sequence. If maxlen < 0, maxlen "
                 "= max(Input(X)).")
        .SetDefault(-1)
        .AddCustomChecker([](const int& v) {
          PADDLE_ENFORCE_EQ(
              v < 0 || v >= 1,
              true,
              common::errors::InvalidArgument(
                  "Attr(maxlen) must be less than 0 or larger than 1"));
        });
    AddAttr<int>("out_dtype", "Output data type");
    AddComment(R"DOC(
SequenceMask Operator

This operator outputs a Mask according to Input(X) and Attr(maxlen).
Supposing Input(X) is a phi::DenseTensor with shape [d_1, d_2, ..., d_n], the
Output(Y) is a mask with shape [d_1, d_2, ..., d_n, maxlen], where:

Y(i_1, i_2, ..., i_n, j) = (j < X(i_1, i_2, ..., i_n))

If maxlen < 0, maxlen = max(X)
    )DOC");
  }
};
}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(sequence_mask,
                            SequenceMaskInferShapeFunctor,
                            PD_INFER_META(phi::SequenceMaskInferMeta));

REGISTER_OPERATOR(
    sequence_mask,
    paddle::operators::SequenceMaskOp,
    paddle::operators::SequenceMaskOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    SequenceMaskInferShapeFunctor);
