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

namespace paddle {
namespace operators {

static std::vector<framework::DDim> RestoreShapes(
    const std::vector<int>& shape_concat, const std::vector<int>& ranks) {
  std::vector<framework::DDim> res;
  int offset = 0;
  for (int len : ranks) {
    auto start_it = shape_concat.begin() + offset;
    auto end_it = start_it + len;
    res.push_back(framework::make_ddim(std::vector<int>(start_it, end_it)));
    offset += len;
  }
  return res;
}

// general infershape for file readers
class CreateFileReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "The output file reader should not be null.");
    const auto shape_concat =
        ctx->Attrs().Get<std::vector<int>>("shape_concat");
    const auto ranks = ctx->Attrs().Get<std::vector<int>>("ranks");
    std::vector<framework::DDim> shapes = RestoreShapes(shape_concat, ranks);
    ctx->SetReaderDims("Out", shapes);

    if (ctx->IsRuntime()) {
      const auto lod_levels = ctx->Attrs().Get<std::vector<int>>("lod_levels");
      PADDLE_ENFORCE_EQ(
          lod_levels.size(), shapes.size(),
          "The number of 'lod_levels'(%d) doesn't match the number "
          "of 'shapes'(%d).",
          lod_levels.size(), shapes.size());
      framework::VarDesc* reader =
          boost::get<framework::VarDesc*>(ctx->GetOutputVarPtrs("Out")[0]);
      reader->SetLoDLevels(lod_levels);
    }
  }
};

// general infershape for decorated readers
class CreateDecoratedReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("UnderlyingReader"),
                   "Input(UnderlyingReader) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "The output decorated reader should not be null.");
    ctx->SetReaderDims("Out", ctx->GetReaderDims("UnderlyingReader"));

    if (ctx->IsRuntime()) {
      framework::VarDesc* in_reader = boost::get<framework::VarDesc*>(
          ctx->GetInputVarPtrs("UnderlyingReader")[0]);
      framework::VarDesc* out_reader =
          boost::get<framework::VarDesc*>(ctx->GetOutputVarPtrs("Out")[0]);
      out_reader->SetLoDLevels(in_reader->GetLoDLevels());
    }
  }
};

// general var type inference for file readers
class CreateFileReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    std::string reader_name = op_desc.Output("Out")[0];
    framework::VarDesc* reader = block->FindVarRecursive(reader_name);
    reader->SetType(framework::proto::VarType::READER);
  }
};

// general var type inference for decorated readers
class CreateDecoratedReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    std::string in_reader_name = op_desc.Input("UnderlyingReader")[0];
    framework::VarDesc* in_reader = block->FindVarRecursive(in_reader_name);
    std::string out_reader_name = op_desc.Output("Out")[0];
    framework::VarDesc* out_reader = block->FindVarRecursive(out_reader_name);
    out_reader->SetType(framework::proto::VarType::READER);
    out_reader->SetDataTypes(in_reader->GetDataTypes());
  }
};

template <typename T>
class CreateRandomDataGeneratorOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& shape_concat = Attr<std::vector<int>>("shape_concat");
    const auto& ranks = Attr<std::vector<int>>("ranks");
    PADDLE_ENFORCE(!shape_concat.empty() && !ranks.empty());
    PADDLE_ENFORCE_EQ(std::accumulate(ranks.begin(), ranks.end(), 0),
                      int(shape_concat.size()),
                      "The accumulate of all ranks should be equal to the "
                      "shape concat's length.");
    std::vector<framework::DDim> shapes = RestoreShapes(shape_concat, ranks);
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new framework::RandomDataGenerator<T>(shapes, Attr<float>("min"),
                                                     Attr<float>("max")));
  }
};

class CreateRandomDataGeneratorOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  CreateRandomDataGeneratorOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(op_proto, op_checker) {
    AddOutput("Out", "(ReaderHolder) The created random reader.");
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
    AddAttr<std::vector<int>>("lod_levels", "The LoD levels of each data.");
    AddAttr<float>("min", "The lower bound of reader's uniform distribution.");
    AddAttr<float>("max", "The upper bound of reader's uniform distribution.");
    AddComment(R"DOC(
      CreateRandomDataGenerator Operator

      This Op creates a random reader.
      The reader generates random data instead of really reading from files.
      Generated data follow an uniform distribution between 'min' and 'max'.
    )DOC");
  }
};

class CreateShuffleReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new framework::ShuffleReader(underlying_reader.Get(),
                                            Attr<int>("buffer_size")));
  }
};

class CreateShuffleReaderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CreateShuffleReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(op_proto, op_checker) {
    AddInput(
        "UnderlyingReader",
        "(ReaderHolder) The underlying reader for creating a shuffle reader.");
    AddOutput("Out", "(ReaderHolder) The created shuffle reader.");
    AddAttr<int>("buffer_size", "The shuffle buffer size.").GreaterThan(0);
    AddComment(R"DOC(
      CreateShuffleReader Operator

      A shuffle reader takes another reader as its 'underlying reader'
      and yields the underlying reader's outputs in a shuffled order.
    )DOC");
  }
};

class CreateBatchReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new framework::BatchReader(underlying_reader.Get(),
                                          Attr<int>("batch_size")));
  }
};

class CreateBatchReaderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CreateBatchReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(op_proto, op_checker) {
    AddInput(
        "UnderlyingReader",
        "(ReaderHolder) The underlying reader for creating a batch reader.");
    AddOutput("Out", "(ReaderHolder) The created batch reader.");
    AddAttr<int>("batch_size",
                 "How many instances the batch reader yields each time.")
        .GreaterThan(0);
    AddComment(R"DOC(
      CreateBatchReader Operator

      A batch reader takes another reader as its 'underlying reader',
      gathers the underlying reader's outputs and then yields them in batches.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(create_random_data_generator,
                  ops::CreateRandomDataGeneratorOp<float>,
                  ops::CreateFileReaderInferShape,
                  ops::CreateRandomDataGeneratorOpMaker,
                  paddle::framework::EmptyGradOpMaker,
                  ops::CreateFileReaderInferVarType);
REGISTER_OPERATOR(create_shuffle_reader, ops::CreateShuffleReaderOp,
                  ops::CreateDecoratedReaderInferShape,
                  ops::CreateShuffleReaderOpMaker,
                  paddle::framework::EmptyGradOpMaker,
                  ops::CreateDecoratedReaderInferVarType);
REGISTER_OPERATOR(create_batch_reader, ops::CreateBatchReaderOp,
                  ops::CreateDecoratedReaderInferShape,
                  ops::CreateBatchReaderOpMaker,
                  paddle::framework::EmptyGradOpMaker,
                  ops::CreateDecoratedReaderInferVarType);
