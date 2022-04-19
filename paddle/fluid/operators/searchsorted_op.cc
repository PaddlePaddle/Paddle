// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class SearchSortedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "SortedSequence");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class SearchSortedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("SortedSequence",
             "(Tensor), N-D or 1-D tensor, The value of the tensor"
             "monotonically increases in the innermost dimension.");
    AddInput("Values", "(Tensor), N-D tensor given values.");
    AddOutput("Out", "(Tensor), The output tensor of searchsorted op.");
    AddAttr<bool>("out_int32",
                  "the output tensor is int64 type if False and on the"
                  "contrary for int32")
        .SetDefault(false);
    AddAttr<bool>(
        "right",
        "corresponding to lower bound if False and upper bound if True")
        .SetDefault(false);

    AddComment(R"DOC(
  Searchsorted Operator.

  This OP is used to find the index of the corresponding sorted_sequence in the innermost dimension based on the given values.
 
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(searchsorted, SearchsortedInferShapeFunctor,
                            PD_INFER_META(phi::SearchsortedInferMeta));
REGISTER_OPERATOR(searchsorted, ops::SearchSortedOp, ops::SearchSortedOpMaker,
                  SearchsortedInferShapeFunctor);
