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

#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class CheckMemoryContinueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class CheckMemoryContinueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(vector<LoDTensor>) The input tensors.").AsDuplicable();
    AddOutput("Out", "(LoDTensor) The output tensor.").AsDuplicable();
    AddOutput(
        "XOut",
        "(vector<LoDTensor>) The output tensors which are the same as x. It is "
        "used to build the graph dependency");
    AddComment(R"DOC(
CheckMemoryContinue Operator.

Check if the address of input tensor are continuous.

Used for converting fused_all_reduce_op_handle in Graph to Program.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(check_memory_continue,
                            CheckMemoryContinueInferShapeFunctor,
                            PD_INFER_META(phi::CheckMemoryContinueInferMeta));

REGISTER_OPERATOR(check_memory_continue,
                  paddle::operators::CheckMemoryContinueOp,
                  paddle::operators::CheckMemoryContinueOpMaker,
                  CheckMemoryContinueInferShapeFunctor);
