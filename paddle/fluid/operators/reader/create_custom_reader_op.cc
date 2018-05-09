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

#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class CustomReader : public framework::DecoratedReader {
 public:
  CustomReader(ReaderBase* reader, const framework::BlockDesc& sub_block,
               const framework::Scope& scope, const platform::Place& dev_place,
               const std::vector<std::string>& source_var_names,
               const std::vector<std::string>& sink_var_names)
      : DecoratedReader(reader),
        sub_block_(sub_block),
        scope_(scope),
        dev_place_(dev_place),
        source_var_names_(source_var_names),
        sink_var_names_(sink_var_names) {}

  void ReadNext(std::vector<framework::LoDTensor>* out) override;

 private:
  const framework::BlockDesc& sub_block_;
  const framework::Scope& scope_;
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
    if (out->Get() != nullptr) {
      return;
    }
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    out->Reset(new CustomReader(
        underlying_reader.Get(), *Attr<framework::BlockDesc*>("sub_block"),
        scope, dev_place, Attr<std::vector<std::string>>("source_var_names"),
        Attr<std::vector<std::string>>("sink_var_names")));
  }
};

class CreateCustomReaderOpMaker : public DecoratedReaderMakerBase {
 public:
  CreateCustomReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
    AddAttr<framework::BlockDesc*>("sub_block", "");
    AddAttr<std::vector<std::string>>("source_var_names", "");
    AddAttr<std::vector<std::string>>("sink_var_names", "");
    AddComment(R"DOC(
      CreateCustomReader Operator

    )DOC");
  }
};

void CustomReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  PADDLE_ENFORCE_EQ(
      source_var_names_.size(), out->size(),
      "The size of source_var_names(%d) not equals to the size of 'out'(%d). "
      "Each element of 'out' must have its own source var in the CustomReader.",
      source_var_names_.size(), out->size());
  PADDLE_ENFORCE_EQ(
      sink_var_names_.size(), out->size(),
      "The size of sink_var_names(%d) not equals to the size of 'out'(%d). "
      "Each element of 'out' must have its own sink var in the CustomReader.",
      sink_var_names_.size(), out->size());

  for (size_t i = 0; i < source_var_names_.size(); ++i) {
    const std::string& var_name = source_var_names_[i];
    framework::Variable* var = scope_.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var, "CustomReader's source variable '%s' doesn't exist.");
    framework::LoDTensor* tensor = var->GetMutable<framework::loDtensor>();
  }
  // TODO(fengjiayi): 将vector中的数据拷贝到sorce_var和sink_var中
  framework::Executor executor(dev_place_);
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle
