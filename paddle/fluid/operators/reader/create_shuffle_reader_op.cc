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

#include <random>
#include "glog/logging.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"

namespace paddle {
namespace operators {
namespace reader {

class ShuffleReader : public framework::DecoratedReader {
 public:
  ShuffleReader(const std::shared_ptr<ReaderBase>& reader, size_t buffer_size,
                size_t seed = 0)
      : DecoratedReader(reader), buffer_size_(buffer_size), seed_(seed) {
    VLOG(10) << "Create shuffle reader of " << reader_;
    if (seed_ == 0) {
      std::random_device device;
      seed_ = device();
    }
    ReloadBuffer();
  }

  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    out->clear();
    if (iteration_pos_ >= buffer_.size()) {
      VLOG(10) << "Resetting shuffle buffer";
      ReloadBuffer();
      if (buffer_.empty()) {
        return;
      }
    }
    *out = buffer_[iteration_pos_++];
  }

 private:
  void ReloadBuffer() {
    buffer_.clear();
    buffer_.reserve(buffer_size_);
    iteration_pos_ = 0;
    for (size_t i = 0; i < buffer_size_; ++i) {
      std::vector<framework::LoDTensor> ins;
      reader_->ReadNext(&ins);
      if (ins.empty()) {
        break;
      }
      buffer_.emplace_back(ins);
    }
    std::mt19937 g(seed_);
    std::shuffle(buffer_.begin(), buffer_.end(), g);
    seed_ = g();  // update seed_;
    VLOG(10) << "random buffer size = " << buffer_.size();
  }

  size_t buffer_size_;
  std::vector<std::vector<framework::LoDTensor>> buffer_;

  size_t iteration_pos_;
  size_t seed_;
};

class CreateShuffleReaderOp : public framework::OperatorBase {
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
    out->Reset(
        new ShuffleReader(underlying_reader.Get(),
                          static_cast<size_t>(Attr<int>("buffer_size"))));
  }
};

class CreateShuffleReaderOpMaker : public DecoratedReaderMakerBase {
 protected:
  void Apply() override {
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
