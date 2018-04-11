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

#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class ThreadedReader : public framework::DecoratedReader {
 public:
  ThreadedReader(ReaderBase* reader, bool safe_mode)
      : DecoratedReader(reader), safe_mode_(safe_mode) {}

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    std::lock_guard<std::mutex> lock(mutex_);
    reader_->ReadNext(out);
  }

  void ReInit() override {
    if (safe_mode_) {
      PADDLE_THROW(
          "ThreadedReader::ReInit() is disabled when 'safe_mode' is true.");
    }
    VLOG(5) << "ThreadedReader::ReInit() is invoked! It might be buggy in "
               "multi-thread environment.";
    reader_->ReInit();
  }

 private:
  bool safe_mode_;
  std::mutex mutex_;
};

class CreateThreadedReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    auto* out = detail::Ref(scope.FindVar(Output("Out")))
                    .GetMutable<framework::ReaderHolder>();
    if (out->Get() != nullptr) {
      return;
    }
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    bool safe_mode = Attr<bool>("safe_mode");
    out->Reset(new ThreadedReader(underlying_reader.Get(), safe_mode));
  }
};

class CreateThreadedReaderOpMaker : public DecoratedReaderMakerBase {
 public:
  CreateThreadedReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
    AddAttr<bool>("safe_mode",
                  "When 'safe_mode' is true, 'ReInit()' is disabled to avoid "
                  "unexpected bugs in multi-thread environment.")
        .SetDefault(true);
    AddComment(R"DOC(
      CreateThreadedReader Operator

      This operator creates a threaded reader. A threaded reader's 
      'ReadNext()' can be invoked by several threads at the same 
      time. 
      When the attribute 'safe_mode' is true, the threaded reader's 
      'ReInit()' is disabled to avoid unexpected bugs in multi-thread 
      environment.
    )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_threaded_reader,
                                   reader::CreateThreadedReaderOp,
                                   reader::CreateThreadedReaderOpMaker);
