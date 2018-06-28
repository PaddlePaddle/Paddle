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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

class ReadInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Reader"),
                   "The ReadOp must take a reader as input.");
    PADDLE_ENFORCE(ctx->HasOutputs("Out"),
                   "The ReadOp should be assigned with output.");
    std::vector<framework::DDim> reader_dims = ctx->GetReaderDims("Reader");
    std::vector<std::string> out_names = ctx->Outputs("Out");
    PADDLE_ENFORCE_EQ(
        reader_dims.size(), out_names.size(),
        "The reader's dim number doesn't match the output number.");
    ctx->SetOutputsDim("Out", reader_dims);
  }
};

class ReadInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    std::string reader_name = op_desc.Input("Reader")[0];
    std::vector<std::string> out_names = op_desc.Output("Out");
    framework::VarDesc* reader = block->FindVarRecursive(reader_name);
    auto dtypes = reader->GetDataTypes();
    PADDLE_ENFORCE_EQ(dtypes.size(), out_names.size());
    for (size_t i = 0; i < dtypes.size(); ++i) {
      framework::VarDesc& out = block->FindRecursiveOrCreateVar(out_names[i]);
      out.SetType(framework::proto::VarType::LOD_TENSOR);
      out.SetDataType(dtypes[i]);
    }
  }
};

class ReadOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    framework::ReaderHolder* reader =
        detail::Ref(scope.FindVar(Input("Reader")),
                    "Cannot find reader variable %s", Input("Reader"))
            .GetMutable<framework::ReaderHolder>();
    std::vector<std::string> out_arg_names = Outputs("Out");
    std::vector<framework::LoDTensor> ins;
    reader->ReadNext(&ins);
    PADDLE_ENFORCE(!ins.empty(), "There is no next data.");
    PADDLE_ENFORCE_EQ(ins.size(), out_arg_names.size());
    for (size_t i = 0; i < ins.size(); ++i) {
      auto* out =
          scope.FindVar(out_arg_names[i])->GetMutable<framework::LoDTensor>();
      out->ShareDataWith(ins[i]);
      out->set_lod(ins[i].lod());
    }
  }
};

class ReadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Reader", "(ReaderHolder) The executed reader.");
    AddOutput("Out", "(LoDTensor) The output data.").AsDuplicable();
    AddComment(R"DOC(
      Read Operator

      Execute a given reader once and output data.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(read, ops::ReadOp, ops::ReadInferShape, ops::ReadOpMaker,
                  paddle::framework::EmptyGradOpMaker, ops::ReadInferVarType);
