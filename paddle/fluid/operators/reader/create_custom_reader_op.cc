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
  CustomReader(ReaderBase* reader, const framework::BlockDesc* sub_block,
               const framework::Scope* scope, const platform::Place& dev_place,
               const std::vector<std::string>& source_var_names,
               const std::vector<std::string>& sink_var_names)
      : DecoratedReader(reader),
        sub_block_(sub_block),
        scope_(scope),
        dev_place_(dev_place),
        source_var_names_(source_var_names),
        sink_var_names_(sink_var_names) {}

  void ReadNext(std::vector<framework::LoDTensor>* out) override;

  void UpdateBlockAndScope(const framework::BlockDesc* sub_block,
                           const framework::Scope* scope) {
    sub_block_ = sub_block;
    scope_ = scope;
  }

 private:
  const framework::BlockDesc* sub_block_;
  const framework::Scope* scope_;
  platform::Place dev_place_;

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
      auto* custom_reader = reinterpret_cast<CustomReader*>(out->Get());
      custom_reader->UpdateBlockAndScope(sub_block, &scope);
      return;
    }
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    out->Reset(
        new CustomReader(underlying_reader.Get(), sub_block, &scope, dev_place,
                         Attr<std::vector<std::string>>("source_var_names"),
                         Attr<std::vector<std::string>>("sink_var_names")));
  }
};

class CreateCustomReaderOpMaker : public DecoratedReaderMakerBase {
 protected:
  void Apply() override {
    AddAttr<framework::BlockDesc*>("sub_block", "");
    AddAttr<std::vector<std::string>>("source_var_names", "");
    AddAttr<std::vector<std::string>>("sink_var_names", "");
    AddComment(R"DOC(
      CreateCustomReader Operator

    )DOC");
  }
};

class CustomReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(!ctx->IsRuntime(),
                   "'CustomReaderInferShape' should only be invoked during "
                   "compile time.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "The output decorated reader should not be null.");
    const auto* sub_block =
        ctx->Attrs().Get<framework::BlockDesc*>("sub_block");
    const auto sink_var_names =
        ctx->Attrs().Get<std::vector<std::string>>("sink_var_names");
    std::vector<std::vector<int64_t>> res_dims;
    std::vector<int32_t> res_lod_levels;
    for (const std::string& var_name : sink_var_names) {
      auto* sink_var = sub_block->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(sink_var);
      res_dims.emplace_back(sink_var->GetShape());
      res_lod_levels.push_back(sink_var->GetLoDLevel());
    }
    auto* out_reader =
        boost::get<framework::VarDesc*>(ctx->GetOutputVarPtrs("Out")[0]);
    out_reader->SetShapes(res_dims);
    out_reader->SetLoDLevels(res_lod_levels);
  }
};

class CustomReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    framework::VarDesc* out_reader = block->FindVar(op_desc.Output("Out")[0]);
    PADDLE_ENFORCE_NOT_NULL(out_reader);
    out_reader->SetType(framework::proto::VarType::READER);

    auto sink_var_names =
        boost::get<std::vector<std::string>>(op_desc.GetAttr("sink_var_names"));
    const auto* sub_block =
        boost::get<framework::BlockDesc*>(op_desc.GetAttr("sub_block"));
    std::vector<framework::proto::VarType::Type> res_data_types;
    for (const std::string& var_name : sink_var_names) {
      framework::VarDesc* var = sub_block->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(var);
      res_data_types.emplace_back(var->GetDataType());
    }
    out_reader->SetDataTypes(res_data_types);
  }
};

void CustomReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  out->clear();
  std::vector<framework::LoDTensor> underlying_outs;
  reader_->ReadNext(&underlying_outs);
  if (underlying_outs.empty()) {
    // There is not next data.
    return;
  }
  PADDLE_ENFORCE(
      source_var_names_.size() == underlying_outs.size() &&
          sink_var_names_.size() == underlying_outs.size(),
      "The size of source_var_names(%d), the size of sink_var_names(%d) and "
      "the size of underlying_outs(%d) are not consistent. Each feeding "
      "element must have its own source and sink variable.",
      source_var_names_.size(), sink_var_names_.size(), underlying_outs.size());

  framework::Scope* exe_scope = &scope_->NewScope();
  // 1. Copy LoDTensors from underlying reader's output to source variables.
  for (size_t i = 0; i < source_var_names_.size(); ++i) {
    framework::Variable* var = exe_scope->Var(source_var_names_[i]);
    framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();
    tensor->ShareDataWith(underlying_outs[i]);
    tensor->set_lod(underlying_outs[i].lod());
  }
  // 2. Run the sub-block.
  framework::Executor executor(dev_place_);
  framework::ProgramDesc* program = sub_block_->Program();
  executor.Run(*program, exe_scope, sub_block_->ID(), false, true);
  // 3. Copy LoDTensors from sink variables to out.
  out->resize(sink_var_names_.size());
  for (size_t i = 0; i < sink_var_names_.size(); ++i) {
    framework::Variable* var = exe_scope->FindVar(sink_var_names_[i]);
    PADDLE_ENFORCE_NOT_NULL(var);
    const framework::LoDTensor& tensor = var->Get<framework::LoDTensor>();
    framework::TensorCopySync(tensor, platform::CPUPlace(), &(*out)[i]);
  }
  scope_->DeleteScope(exe_scope);
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_OPERATOR(create_custom_reader, ops::CreateCustomReaderOp,
                  ops::CreateCustomReaderOpMaker, ops::CustomReaderInferShape,
                  ops::CustomReaderInferVarType,
                  paddle::framework::EmptyGradOpMaker)
