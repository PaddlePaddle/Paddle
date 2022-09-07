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

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class CustomReader : public framework::DecoratedReader {
 public:
  CustomReader(const std::shared_ptr<ReaderBase>& reader,
               const framework::BlockDesc& sub_block,
               const std::vector<std::string>& source_var_names,
               const std::vector<std::string>& sink_var_names)
      : DecoratedReader(reader),
        program_(*sub_block.Program()),
        sub_block_id_(sub_block.ID()),
        exe_(framework::Executor(platform::CPUPlace())),
        source_var_names_(source_var_names),
        sink_var_names_(sink_var_names) {}

  void ReadNextImpl(std::vector<framework::LoDTensor>* out) override;

 private:
  const framework::ProgramDesc program_;
  int sub_block_id_;
  framework::Executor exe_;
  framework::Scope scope_;

  std::vector<std::string> source_var_names_;
  std::vector<std::string> sink_var_names_;
};

class CreateCustomReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    auto* sub_block = Attr<framework::BlockDesc*>("sub_block");
    if (out->Get() != nullptr) {
      return;
    }
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    out->Reset(framework::MakeDecoratedReader<CustomReader>(
        underlying_reader,
        *sub_block,
        Attr<std::vector<std::string>>("source_var_names"),
        Attr<std::vector<std::string>>("sink_var_names")));
  }
};

class CreateCustomReaderOpMaker : public DecoratedReaderMakerBase {
 protected:
  void Apply() override {
    AddAttr<framework::BlockDesc*>(
        "sub_block", "The block to hold all preprocessing operators.");
    AddAttr<std::vector<std::string>>(
        "source_var_names",
        "Source variables are starting points of data preprocessing. They hold "
        "preprocessing's input tensors. Each source variable corresponds to "
        "one of underlying reader's output datas.");
    AddAttr<std::vector<std::string>>(
        "sink_var_names",
        "Sink variables are ending points of data preprocessing. They hold "
        "preprocessing's output tensors. Each sink variable corresponds to "
        "one of custom reader's output datas.");
    AddComment(R"DOC(
      CreateCustomReader Operator

      A custom reader can be used for input data preprocessing.
      A custom reader holds its own sub-block, which will be executed in CPU
      in its 'ReadNext()' function. Users can configurate their own
      preprocessing pipelines by inserting operators into custom reader's
      sub-block.
    )DOC");
  }
};

class CustomReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_NE(
        ctx->IsRuntime(),
        true,
        platform::errors::PreconditionNotMet(
            "'CustomReaderInferShape' should only be invoked during "
            "compile time."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"),
                      true,
                      platform::errors::NotFound(
                          "The output decorated reader should not be null."));
    const auto* sub_block =
        ctx->Attrs().Get<framework::BlockDesc*>("sub_block");
    const auto sink_var_names =
        ctx->Attrs().Get<std::vector<std::string>>("sink_var_names");
    std::vector<std::vector<int64_t>> res_dims;
    std::vector<int32_t> res_lod_levels;
    for (const std::string& var_name : sink_var_names) {
      auto* sink_var = sub_block->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          sink_var,
          platform::errors::NotFound(
              "The sink variable is not found in CustomReader."));
      res_dims.emplace_back(sink_var->GetShape());
      res_lod_levels.push_back(sink_var->GetLoDLevel());
    }
    auto* out_reader =
        PADDLE_GET(framework::VarDesc*, ctx->GetOutputVarPtrs("Out")[0]);
    out_reader->SetShapes(res_dims);
    out_reader->SetLoDLevels(res_lod_levels);
  }
};

class CustomReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto& out_var_name = ctx->Output("Out")[0];
    PADDLE_ENFORCE_EQ(ctx->HasVar(out_var_name),
                      true,
                      platform::errors::NotFound(
                          "The output reader variable should not be null."));
    ctx->SetType(out_var_name, framework::proto::VarType::READER);

    auto sink_var_names = PADDLE_GET_CONST(std::vector<std::string>,
                                           ctx->GetAttr("sink_var_names"));
    const auto* sub_block =
        PADDLE_GET_CONST(framework::BlockDesc*, ctx->GetAttr("sub_block"));
    std::vector<framework::proto::VarType::Type> res_data_types;
    for (const std::string& var_name : sink_var_names) {
      framework::VarDesc* var = sub_block->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          platform::errors::NotFound(
              "The sink variable is not found in CustomReader."));
      res_data_types.emplace_back(var->GetDataType());
    }
    ctx->SetDataTypes(out_var_name, res_data_types);
  }
};

void CustomReader::ReadNextImpl(paddle::framework::LoDTensorArray* out) {
  out->clear();
  paddle::framework::LoDTensorArray underlying_outs;
  reader_->ReadNext(&underlying_outs);
  if (underlying_outs.empty()) {
    // There is not next data.
    return;
  }
  PADDLE_ENFORCE_EQ(
      source_var_names_.size(),
      underlying_outs.size(),
      platform::errors::InvalidArgument(
          "The size of source_var_names(%d) and the size of "
          "underlying_outs(%d) are not consistent. Each feeding element "
          "must have its own source variable.",
          source_var_names_.size(),
          underlying_outs.size()));
  // The scope for CustomReader's sub-block should be independent and shouldn't
  // be any other computation scope's child. Otherwise, data preprocessing and
  // compution cannot be concurrent.
  framework::Scope* exe_scope = &scope_.NewScope();
  // 1. Copy LoDTensors from underlying reader's output to source variables.
  for (size_t i = 0; i < source_var_names_.size(); ++i) {
    framework::Variable* var = exe_scope->Var(source_var_names_[i]);
    framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();
    tensor->ShareDataWith(underlying_outs[i]);
    tensor->set_lod(underlying_outs[i].lod());
  }
  // 2. Run the sub-block.
  exe_.Run(program_, exe_scope, sub_block_id_, false, true, {}, true);
  // 3. Copy LoDTensors from sink variables to out.
  out->resize(sink_var_names_.size());
  for (size_t i = 0; i < sink_var_names_.size(); ++i) {
    auto* var = exe_scope->FindVar(sink_var_names_[i]);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::NotFound("The variable %s is not in current scope.",
                                   sink_var_names_[i]));
    const auto& tensor = var->Get<framework::LoDTensor>();
    framework::TensorCopySync(tensor, platform::CPUPlace(), &(*out)[i]);
  }
  scope_.DeleteScope(exe_scope);
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_OPERATOR(
    create_custom_reader,
    ops::CreateCustomReaderOp,
    ops::CreateCustomReaderOpMaker,
    ops::CustomReaderInferShape,
    ops::CustomReaderInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)
