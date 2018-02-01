//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/framework/op_registry.h"
#include "paddle/framework/reader.h"

namespace paddle {
namespace operators {

// general infershape
class CreateReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of CreateReaderOp should not be null.");
  }
};

template <typename T>
class CreateRandomReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;
  void Run(const framework::Scope& scope,
           const platform::Place& dev_place) const override {
    const auto& shape_concat = Attr<std::vector<int>>("shape_concat");
    const auto& ranks = Attr<std::vector<int>>("ranks");
    PADDLE_ENFORCE_EQ(std::accumulate(ranks.begin(), ranks.end(), 0),
                      int(shape_concat.size()),
                      "The accumulate of all ranks should be equal to the "
                      "shape concat's length.");
    std::vector<framework::DDim> shapes;
    int offset = 0;
    for (int len : ranks) {
      auto start_it = shape_concat.begin() + offset;
      auto end_it = start_it + len;
      shapes.push_back(
          framework::make_ddim(std::vector<int>(start_it, end_it)));
      offset += len;
    }
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::RandomReader<T>>();
    out->Initialize(shapes, Attr<float>("min"), Attr<float>("max"));
  }
};

class CreateRandomReaderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CreateRandomReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(op_proto, op_checker) {
    AddOutput("Out", "(RandomReader) The created random reader.");
    AddAttr<std::vector<int>>("shape_concat",
                              "The concat of all data's shapes.");
    AddAttr<std::vector<int>>(
        "ranks",
        "The ranks of each data."
        "e.g."
        "shape_concat = [2,3,4,5,6]"
        "ranks = [3,2]"
        "It means the reader will generate two data each time,"
        "whose shapes are [2,3,4] and [5,6] respectively.");
    AddAttr<float>("min", "The lower bound of reader's uniform distribution.");
    AddAttr<float>("max", "The upper bound of reader's uniform distribution.");
    AddComment(R"DOC(
      CreateRandomReader Operator

      This Op creates a random reader. 
      The reader generates random data instead of really reading from files.
      Generated data follow an uniform distribution between 'min' and 'max'.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(create_random_reader, ops::CreateRandomReaderOp<float>,
                  ops::CreateReaderInferShape, ops::CreateRandomReaderOpMaker,
                  paddle::framework::EmptyGradOpMaker);