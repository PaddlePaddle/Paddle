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
  ThreadedReader(ReaderBase* reader, bool unsafe_mode)
      : DecoratedReader(reader), unsafe_mode_(unsafe_mode) {}

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!unsafe_mode_) {
      if (!reader_->HasNext()) {
        PADDLE_THROW("There is no next data!");
      }
      reader_->ReadNext(out);
    } else {
      auto& thread_buffer = thread_buffers_[std::this_thread::get_id()];
      if (thread_buffer.empty()) {
        PADDLE_THROW(
            "thread_buffer is empty! HasNext() must be invoked before "
            "ReadNext() in the same thread.");
      }
      *out = thread_buffer;
      thread_buffer.clear();
    }
  }

  bool HasNext() const override {
    if (!unsafe_mode_) {
      PADDLE_THROW(
          "ThreadedReader::HasNext() is disabled when 'unsafe_mode' is false.");
    }
    std::thread::id thread_id = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(mutex_);
    auto& thread_buffer = thread_buffers_[thread_id];
    if (thread_buffer.empty() && reader_->HasNext()) {
      reader_->ReadNext(&thread_buffer);
    }
    return !thread_buffer.empty();
  }

  void ReInit() override {
    if (!unsafe_mode_) {
      PADDLE_THROW(
          "ThreadedReader::ReInit() is disabled when 'unsafe_mode' is false.");
    }
    VLOG(5) << "ThreadedReader::ReInit() is invoked! It might be buggy in "
               "multi-thread environment.";
    reader_->ReInit();
  }

  ~ThreadedReader() {
    for (auto& p : thread_buffers_) {
      if (!p.second.empty()) {
        PADDLE_THROW(
            "Find an unused data batch in ThreadedReader! Maybe one thread "
            "invokes 'HasNext()' without subsequent 'ReadNext()'.");
      }
    }
  }

 private:
  bool unsafe_mode_;
  mutable std::mutex mutex_;
  mutable std::unordered_map<std::thread::id, std::vector<framework::LoDTensor>>
      thread_buffers_;
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
    bool unsafe_mode = Attr<bool>("unsafe_mode");
    out->Reset(new ThreadedReader(underlying_reader.Get(), unsafe_mode));
  }
};

class CreateThreadedReaderOpMaker : public DecoratedReaderMakerBase {
 public:
  CreateThreadedReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
    AddAttr<bool>("unsafe_mode",
                  "When 'unsafe_mode' is false, invoking 'HasNext()' or "
                  "'ReInit()' is not allowed to avoid unexpected bugs in "
                  "multi-thread environment.")
        .SetDefault(false);
    AddComment(R"DOC(
      CreateThreadedReader Operator

      This operator creates a threaded reader. A threaded reader's 
      'ReadNext()' can be invoked by several threads at the same 
      time. 
      When the attribute 'unsafe_mode' is false, the threaded reader's 
      'HasNext()' and 'ReInit()' will be disabled to avoid unexpected 
      bugs in multi-thread environment. If you really need them, you 
      can enable them by setting 'unsafe_mode' true. In this case, 
      'HasNext()' returning true only guarantees the safety of 
      invoking 'ReadNext()' in the same thread. Each thread must 
      invoke 'HasNext()' and 'ReadNext()' in pairs.
    )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = paddle::operators::reader;
REGISTER_FILE_READER_OPERATOR(create_threaded_reader,
                              reader::CreateThreadedReaderOp,
                              reader::CreateThreadedReaderOpMaker);
