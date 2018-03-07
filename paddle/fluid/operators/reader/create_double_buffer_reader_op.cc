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

#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

static constexpr size_t kDoubleBufferSize = 3;

class DoubleBufferReader : public framework::DecoratedReader {
 public:
  explicit DoubleBufferReader(ReaderBase* reader)
      : DecoratedReader(reader),
        buffer_(kDoubleBufferSize),
        write_pos_(0),
        read_pos_(0) {
    std::thread prefetch(
        std::bind(&DoubleBufferReader::PrefetchThreadFunc, this));
    prefetch.detach();
    // framework::Async(
    //      std::bind(&DoubleBufferReader::PrefetchThreadFunc, this));
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  bool HasNext() const override;

 private:
  void PrefetchThreadFunc();

  std::vector<std::vector<framework::LoDTensor>> buffer_;
  size_t write_pos_;
  size_t read_pos_;

  std::mutex mtx_;
  std::condition_variable buffer_not_full_;
  std::condition_variable buffer_not_empty_;
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
  std::unique_lock<std::mutex> lck(mtx_);
  while (write_pos_ == read_pos_) {
    buffer_not_empty_.wait(lck);
  }

  out->clear();
  out->reserve(buffer_[read_pos_].size());
  // TODO(fengjiayi): This copy shall be reduced.
  for (size_t i = 0; i < buffer_[read_pos_].size(); ++i) {
    framework::LoDTensor dst;
    TensorCopy(buffer_[read_pos_][i], platform::CPUPlace(), &dst);
    dst.set_lod(buffer_[read_pos_][i].lod());
    out->push_back(dst);
  }

  ++read_pos_;
  if (read_pos_ >= kDoubleBufferSize) {
    read_pos_ = 0;
  }
  buffer_not_full_.notify_all();
}

bool DoubleBufferReader::HasNext() const {
  return reader_->HasNext() || !buffer_.empty();
}

void DoubleBufferReader::PrefetchThreadFunc() {
  while (reader_->HasNext()) {
    std::unique_lock<std::mutex> lck(mtx_);
    while (((write_pos_ + 1) % kDoubleBufferSize) == read_pos_) {
      buffer_not_full_.wait(lck);
    }
    reader_->ReadNext(&buffer_[write_pos_]);
    ++write_pos_;
    if (write_pos_ >= kDoubleBufferSize) {
      write_pos_ = 0;
    }
    buffer_not_empty_.notify_all();
  }
}

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_double_buffer_reader,
                                   ops::CreateDoubleBufferReaderOp,
                                   ops::CreateDoubleBufferReaderOpMaker);
