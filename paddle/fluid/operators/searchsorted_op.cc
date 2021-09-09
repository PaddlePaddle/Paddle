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

#include "paddle/fluid/operators/searchsorted_op.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

class SearchSortedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  static bool SearchsortedDimsMatchedBeforeLastDim(
      const framework::DDim& sequences_dims,
      const framework::DDim& values_dims) {
    if (sequences_dims.size() != values_dims.size()) {
      return false;
    }
    const auto& sequences_dims_size = sequences_dims.size();
    for (int64_t dim = 0; dim < sequences_dims_size - 1; ++dim) {
      if (sequences_dims[dim] != values_dims[dim]) {
        return false;
      }
    }
    return true;
  }

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("SortedSequence"), "Input", "SortedSequence",
                   "searchsorted");
    OP_INOUT_CHECK(ctx->HasInput("Values"), "Input", "Values", "searchsorted");

    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "searchsorted");

    auto sequences_dims = ctx->GetInputDim("SortedSequence");
    auto values_dims = ctx->GetInputDim("Values");
    auto out_int32 = ctx->Attrs().Get<bool>("out_int32");

    if (sequences_dims.size() != 1) {
      PADDLE_ENFORCE_EQ(
          SearchsortedDimsMatchedBeforeLastDim(sequences_dims, values_dims),
          true,
          platform::errors::Unavailable(
              "The dimensions of sorted_sequence tensor ( %s ) and values "
              "tensor ( %s ) can not match. Because the input sorted_sequence "
              "tensor must be 1 dimension or the first N-1 dimensions of "
              "sorted_sequence tensor and input values tensor must match. "
              "Please input appropriate sorted_sequence and values again! ",
              sequences_dims, values_dims));
    }

    if (out_int32) {
      PADDLE_ENFORCE_LT(
          sequences_dims[sequences_dims.size() - 1],
          std::numeric_limits<int>::max(),
          platform::errors::Unavailable(
              "The size of sorted_sequence %d exceed the maximum limit d%. "
              "Because the size of sorted_sequence should be less than the "
              "output maximum value for int32 bit. Please set appropriate "
              "sorted_sequence to meet this requirement! ",
              sequences_dims[sequences_dims.size() - 1],
              std::numeric_limits<int>::max()));
    }

    ctx->SetOutputDim("Out", values_dims);
  }

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

REGISTER_OPERATOR(searchsorted, ops::SearchSortedOp, ops::SearchSortedOpMaker);

REGISTER_OP_CPU_KERNEL(
    searchsorted,
    ops::SearchSortedKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SearchSortedKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SearchSortedKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SearchSortedKernel<paddle::platform::CPUDeviceContext, int64_t>);
