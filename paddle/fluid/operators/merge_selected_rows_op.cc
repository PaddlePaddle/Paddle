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

#include <unordered_map>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class MergeSelectedRowsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class MergeSelectedRowsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input type is SelectedRows, and the selected rows may be "
             "duplicated.");
    AddOutput("Out",
              "The output type is SelectedRows, and the selected rows are not "
              "duplicated.");
    AddComment(
        R"DOC(
MergeSelectedRows Operator.

MergeSelectedRows is used to merge the duplicated rows of the input. The
output's row has no duplicated, and it's order is incremental.

Example:
  Input:
    X.rows is [0, 5, 5, 4, 19]
    X.height is 20
    X.value is:
        [[1, 1]
         [2, 2]
         [3, 3]
         [4, 4]
         [6, 6]]

   Output:
    Out.row is [0, 4, 5, 19]
    Out.height is 20
    Out.value is:
        [[1, 1]
         [4, 4]
         [5, 5]
         [6, 6]]
)DOC");
  }
};

class MergeSelectedRowsOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(merge_selected_rows,
                            MergeSelectedRowsInferMetaFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(merge_selected_rows,
                  ops::MergeSelectedRowsOp,
                  ops::MergeSelectedRowsOpMaker,
                  ops::MergeSelectedRowsOpInferVarType,
                  MergeSelectedRowsInferMetaFunctor);
