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

class ShuffleReader : public framework::DecoratedReader {
 public:
  ShuffleReader(ReaderBase* reader, int buffer_size)
      : DecoratedReader(reader), buffer_size_(buffer_size), iteration_pos_(0) {
    buffer_.reserve(buffer_size);
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override;

 private:
  int buffer_size_;
  std::vector<std::vector<framework::LoDTensor>> buffer_;
  size_t iteration_pos_;
};

void ShuffleReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  if (iteration_pos_ >= buffer_.size()) {
    // Reload buffer with new data
    buffer_.clear();
    buffer_.reserve(buffer_size_);
    for (int i = 0; i < buffer_size_; ++i) {
      buffer_.push_back(std::vector<framework::LoDTensor>());
      reader_->ReadNext(&buffer_.back());
      if (buffer_.back().empty()) {
        buffer_.pop_back();
        break;
      }
    }
    // TODO(fengjiayi): 'std::random_shuffle' can be very slow. It needs to be
    // optimize.
    std::random_shuffle(buffer_.begin(), buffer_.end());
    iteration_pos_ = 0;
  }
  out->clear();
  if (!buffer_.empty()) {
    std::swap(*out, buffer_[iteration_pos_++]);
  }
  // if buffer_ is empty, the 'out' will return as an empty vector.
}

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
    out->Reset(
        new ShuffleReader(underlying_reader.Get(), Attr<int>("buffer_size")));
  }
};

class CreateShuffleReaderOpMaker : public DecoratedReaderMakerBase {
 public:
  CreateShuffleReaderOpMaker(OpProto* op_proto, OpAttrChecker* op_checker)
      : DecoratedReaderMakerBase(op_proto, op_checker) {
    AddAttr<int>("buffer_size", "The shuffle buffer size.").GreaterThan(0);
    AddComment(R"DOC(
      CreateShuffleReader Operator

      A shuffle reader takes another reader as its 'underlying reader'
      and yields the underlying reader's outputs in a shuffled order.
    )DOC");
  }
};
}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::reader;
REGISTER_DECORATED_READER_OPERATOR(create_shuffle_reader,
                                   ops::CreateShuffleReaderOp,
                                   ops::CreateShuffleReaderOpMaker);
