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

#include <thread>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

static constexpr size_t kDoubleBufferSize = 2;

class DoubleBufferReader : public framework::DecoratedReader {
 public:
  explicit DoubleBufferReader(ReaderBase* reader)
      : DecoratedReader(reader),
        buffer_(framework::MakeChannel<std::vector<framework::LoDTensor>>(
            kDoubleBufferSize)) {
    std::thread prefetch(&DoubleBufferReader::PrefetchThreadFunc, this);
    prefetch.detach();
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  void ReInit() override;

  ~DoubleBufferReader() { buffer_->Close(); }

  bool HasNext() const override;

 private:
  void PrefetchThreadFunc();

  framework::Channel<std::vector<framework::LoDTensor>>* buffer_;
};

class CreateDoubleBufferReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& underlying_reader = scope.FindVar(Input("UnderlyingReader"))
                                        ->Get<framework::ReaderHolder>();
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new DoubleBufferReader(underlying_reader.Get()));
  }
};

class CreateDoubleBufferReaderOpMaker : public DecoratedReaderMakerBase {
 public:
  CreateDoubleBufferReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
    AddComment(R"DOC(
      CreateDoubleBufferReader Operator

      A double buffer reader takes another reader as its 'underlying reader'.
      It launches another thread to execute the 'underlying reader' asynchronously, 
      which prevents reading process from blocking subsequent training.
    )DOC");
  }
};

void DoubleBufferReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  out->clear();
  buffer_->Receive(out);
}

void DoubleBufferReader::ReInit() {
  reader_->ReInit();
  buffer_->Close();
  // The existing prefetch thread will terminate for the buffer_ is closed.
  buffer_ = framework::MakeChannel<std::vector<framework::LoDTensor>>(
      kDoubleBufferSize);
  std::thread prefetch(&DoubleBufferReader::PrefetchThreadFunc, this);
  prefetch.detach();
}

void DoubleBufferReader::PrefetchThreadFunc() {
  VLOG(5) << "A new prefetch thread starts.";
  while (true) {
    std::vector<framework::LoDTensor> batch;
    reader_->ReadNext(&batch);
    if (batch.empty()) {
      // EOF
      buffer_->Close();
      VLOG(5) << "Reached the end of the file. The prefetch thread terminates.";
      break;
    }
    if (!buffer_->Send(&batch)) {
      VLOG(5) << "WARNING: The double buffer channel has been closed. The "
                 "prefetch thread terminates.";
      break;
    }
  }
}

bool DoubleBufferReader::HasNext() const { PADDLE_THROW("Not Implemented"); }

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_double_buffer_reader,
                                   ops::CreateDoubleBufferReaderOp,
                                   ops::CreateDoubleBufferReaderOpMaker);
