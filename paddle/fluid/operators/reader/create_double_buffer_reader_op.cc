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
#include "paddle/phi/core/operators/reader/buffered_reader.h"

namespace paddle::operators::reader {
class CreateDoubleBufferReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const phi::Place& dev_place) const override {
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();

    if (out->Get() != nullptr) {
      auto* decorated_reader =
          dynamic_cast<framework::DecoratedReader*>(out->Get().get());
      PADDLE_ENFORCE_NOT_NULL(
          decorated_reader,
          common::errors::NotFound("The inited reader should be a "
                                   "DecoratedReader when running "
                                   "create_double_buffer_reader op."));
      if (decorated_reader->UnderlyingReader() == underlying_reader.Get()) {
        return;
      }
    }

    auto place_str = Attr<std::string>("place");
    phi::Place place;
    if (place_str == "AUTO") {
      place = dev_place;
    } else if (place_str == "PLACE(CPU)") {
      place = phi::CPUPlace();
    } else {
      place_str = place_str.substr(0, place_str.length() - 1);
      std::istringstream sin(place_str);
      sin.seekg(std::string("PLACE(GPU:").size(), std::ios::beg);  // NOLINT
      size_t num = 0;
      sin >> num;
      place = phi::GPUPlace(static_cast<int>(num));
    }

    VLOG(10) << "Create new double buffer reader on " << place;

    out->Clear();
    out->Reset(framework::MakeDecoratedReader<BufferedReader>(
        underlying_reader, place, 2));
  }
};

class CreateDoubleBufferReaderOpMaker : public DecoratedReaderMakerBase {
 protected:
  void Apply() override {
    AddComment(R"DOC(
      CreateDoubleBufferReader Operator

      A double buffer reader takes another reader as its 'underlying reader'.
      It launches another thread to execute the 'underlying reader' asynchronously,
      which prevents reading process from blocking subsequent training.
    )DOC");
    std::unordered_set<std::string> enum_range;
    constexpr size_t kMaxCUDADevs = 128;
    for (size_t i = 0; i < kMaxCUDADevs; ++i) {
      enum_range.insert(string::Sprintf("PLACE(GPU:%d)", i));
    }
    enum_range.insert("CPUPLACE");
    enum_range.insert("AUTO");
    AddAttr<std::string>("place", "The double buffer place")
        .SetDefault("AUTO")
        .InEnum({enum_range});
  }
};

}  // namespace paddle::operators::reader

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_double_buffer_reader,
                                   ops::CreateDoubleBufferReaderOp,
                                   ops::CreateDoubleBufferReaderOpMaker);
